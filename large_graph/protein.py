import argparse
import random
import numpy as np
from layers import oursgat
import torch.nn as nn
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph
from torch_geometric.nn import GCNConv, SAGEConv
from lg_parse import parse_method, parser_add_main_args
from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from torch.nn import LayerNorm, Linear

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NWGformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, droupout=0.3, P=2, M=2, bn=False, ln=False, res=False,use_weight=False, gnn='gat'):
        super(NWGformer, self).__init__()
        self.linq1 = torch.nn.Linear(input_dim, output_dim)
        self.linv1 = torch.nn.Linear(input_dim, output_dim)
        if gnn == 'gat' and P != 0:
            self.conv = oursgat(input_dim, output_dim)
        else:
            self.conv = GCNConv(input_dim, output_dim, cached=False, normalize=True)
        self.use_weight = use_weight
        if self.use_weight:
            self.w = torch.nn.Linear(input_dim, 1)
        self.P = P
        self.M = M
        self.bn = bn
        self.ln = ln
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)
        if self.ln:
            self.ln = torch.nn.LayerNorm(output_dim)
        self.res = res
        self.p = droupout
        self.eps = 1e-10

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.linq1.reset_parameters()
        self.linv1.reset_parameters()
        if self.use_weight:
            self.w.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()
        if self.ln:
            self.ln.reset_parameters()

    def forward(self, x, edge_index):
        Q = F.relu(self.linq1(x))
        V = self.linv1(x)
        if self.res:
            out = (self.former(Q, V) + self.conv(x, edge_index)) + V
        else:
            out = (self.former(Q, V) + self.conv(x, edge_index))
        if self.bn:
            out = self.bn(out)
        elif self.ln:
            out = self.ln(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.p, training=self.training)
        return out

    def former(self, Q, V):
        if self.use_weight:
            H = (self.w(Q)) ** self.P
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
        else:
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
            H = (torch.sum(Q, dim=1).unsqueeze(1)) ** self.P
        m = (torch.max(H) - torch.min(H)) * self.M
        H = (torch.pi / (2 * m)) * H
        cosQ = (Q * torch.cos(H))
        sinQ = (Q * torch.sin(H))
        out = torch.mm(cosQ, torch.mm(cosQ.T, V)) + torch.mm(sinQ, torch.mm(sinQ.T, V))
        norm = torch.mm(cosQ, (torch.sum(cosQ, dim=0)).unsqueeze(1)) + torch.mm(sinQ,(torch.sum(sinQ, dim=0)).unsqueeze(1)) + self.eps
        return (out / norm)

    def Softmax(self, Q, V):
        A = F.softmax(torch.mm(Q,Q.T), dim=1)
        out = torch.mm(A, V)
        return out


class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
                 in_dropout=0.1, dropout=0.3, heads=1,
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
        self.former1=NWGformer(in_channels,hidden_channels)
        #self.former2 = NWGformer(hidden_channels, hidden_channels)
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.bn:
            self.bns = torch.nn.ModuleList()
        ## first layer
        if local_attn:
            self.local_convs.append(oursgat(hidden_channels, hidden_channels))
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
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,cached=False, normalize=True))
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
        #self.former2.reset_parameters()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x=self.former1(x,edge_index)
        #x=self.former2(x,edge_index)
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



### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args([
    '--dataset', 'ogbn-proteins',
    '--hidden_channels', '128',
    '--epochs', '301',
    '--batch_size', '10000',
    '--lr', '0.01',
    '--runs', '5',
    '--local_layers', '6',
    '--local_attn',
    '--in_drop', '0.1',
    '--dropout',  '0.3',
    '--weight_decay', '0.0',
    '--bn',
    '--eval_step', '9',
    '--eval_epoch', '1000',
    '--device', '0',
    '--res'
])

print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)


# get the splits for all runs
if args.dataset in ('ogbn-proteins','ogbn-arxiv', 'ogbn-products'):
    split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)


### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

#dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

### Load method ###
model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers,
              in_dropout=args.in_dropout, dropout=args.dropout,
              heads=args.num_heads, pre_ln=args.pre_ln,
              bn=args.bn, local_attn=args.local_attn, res=args.res, ln=args.ln, jk=args.jk, sage=args.sage).to(device)

criterion = nn.BCEWithLogitsLoss()
eval_func = eval_rocauc
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label
else:
    true_label = dataset.label
all_best_tests = []
### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)
    num_batch = n // args.batch_size + 1

    best_tests_for_run = []
    for epoch in range(args.epochs):

        model.to(device)
        model.train()

        loss_train = 0
        idx = torch.randperm(n)
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            train_mask_i = train_mask[idx_i]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            optimizer.zero_grad()
            out_i = model(x_i, edge_index_i)
            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= num_batch
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss_train:.4f}')


        if epoch % 20==0 and epoch>99:
            result = evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device='cpu')

            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]
                if args.save_model:
                    save_model(args, model, optimizer, run)
            best_tests_for_run.append(best_test)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss_train:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%, '
                      f'Best Valid: {100 * best_val:.2f}%, '
                      f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)
    all_best_tests.append(best_tests_for_run)

average_best_tests = np.mean(all_best_tests, axis=0)

for epoch, avg_best_test in enumerate(average_best_tests):
    print(f'Epoch {epoch}: Average Best Test: {100 * avg_best_test:.2f}%')

results = logger.print_statistics()
### Save results ###
save_result(args, results)
