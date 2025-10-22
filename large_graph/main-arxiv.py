import argparse
import random
import numpy as np
import torch.nn as nn
from layers import NWGformerF,oursgat
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, SAGEConv
from lg_parse import parse_method, parser_add_main_args
from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from torch_scatter import scatter,scatter_add
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args([
    '--dataset', 'ogbn-arxiv',
    '--hidden_channels', '512',
    '--epochs', '1000',
    '--lr', '0.0005',
    '--runs', '5',
    '--local_layers', '4',
    '--bn',
    '--device', '0',
    '--res',
    '--weight_decay', '5e-4',
    '--display_step', '100'
])
print(args)
fix_seed(args.seed)
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,in_dropout=0.15, dropout=0.5, heads=1,
                 pre_ln=False, bn=True, local_attn=False, res=True, ln=False, jk=False, sage=False):
        super(MPNNs, self).__init__()
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.bn = bn
        self.res = res
        self.jk = jk
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.former1=NWGformerF(in_channels,hidden_channels)
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.bn:
            self.bns = torch.nn.ModuleList()
        ## first layer
        if local_attn:
            self.local_convs.append(oursgat(in_channels, hidden_channels))
        elif sage:
            self.local_convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.local_convs.append(GCNConv(hidden_channels, hidden_channels,cached=False, normalize=True))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.bn:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        ## following layers
        for _ in range(local_layers - 1):
            self.h_lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if local_attn:
                self.local_convs.append(oursgat(hidden_channels, hidden_channels))
            elif sage:
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
            if self.bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.bn:
            for p_bn in self.bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()
        self.former1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x=self.former1(x,edge_index)
        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
        x = self.pred_local(x_final)
        return x


### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers,
              in_dropout=args.in_dropout, dropout=args.dropout,
              heads=args.num_heads, pre_ln=args.pre_ln,
              bn=args.bn, local_attn=args.local_attn, res=args.res, ln=args.ln, jk=args.jk, sage=args.sage).to(device)

criterion = nn.NLLLoss()
eval_func = eval_acc
logger = Logger(args.runs, args)
model.train()
print('MODEL:', model)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
### Training loop ###
all_best_tests = []
for run in range(args.runs):
    print('runs:',run)
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    #scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-5, factor=0.9, patience=50, verbose=True)
    best_val = float('-inf')
    best_test = float('-inf')
    maxtest = 0.0
    best_tests_for_run = []
    if args.save_model:
        save_model(args, model, optimizer, run)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])
        #scheduler.step(result[3])
        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)
        if result[2] > maxtest:
            maxtest = result[2]
        best_tests_for_run.append(best_test)
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%, '
                  f'max Test: {100 * maxtest:.2f}%')
    all_best_tests.append(best_tests_for_run)



average_best_tests = np.mean(all_best_tests, axis=0)
average_best_tests_std = np.std(all_best_tests, axis=0)

for epoch, avg_best_test in enumerate(average_best_tests):
    print(f'Epoch {epoch}: Average Best Test: {100 * avg_best_test:.2f}%')

for epoch, avg_std in enumerate(average_best_tests_std):
    print(f'Epoch {epoch}: Average std: {100 * avg_std:.2f}%')


results = logger.print_statistics()
### Save results ###
save_result(args, results)
