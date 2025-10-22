"""
Training on large dataset using neighbor sampling.
"""
import argparse
import copy
import os
import torch
import torch.nn as nn
from data_utils import load_fixed_splits
from dataset import load_dataset
from parse import parse_method, parser_add_main_args
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils import add_self_loops, to_undirected
from tqdm import tqdm
from SE import save_or_load_shortest_paths


def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def train(model, graph, loss_func, optimizer, batch_size, args):
    model.train()
    center_node_indices = torch.arange(batch_size, device=graph.x.device)

    if args.method == "nodeformer":
        output, loss_add = model(graph.x, graph.edge_index, center_node_indices)
        loss = loss_func(output, graph.y[:batch_size]) + loss_add[0]
    else:
        output = model(graph.x, graph.edge_index)[:batch_size]
        labels = graph.y[:batch_size]
        loss = loss_func(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, batch_size, args):
    model.eval()
    center_node_indices = torch.arange(batch_size, device=graph.x.device)
    if args.method == "nodeformer":
        output, loss_add = model(graph.x, graph.edge_index, center_node_indices)
    else:
        output = model(graph.x, graph.edge_index)[:batch_size]
    labels = graph.y[:batch_size]
    correct = (output.argmax(-1) == labels).sum().item()
    total = labels.size(0)
    return correct, total


def chunked_to_undirected(edge_index, num_nodes, chunk_size=10000000):
    num_edges = edge_index.shape[1]
    processed_chunks = []
    for i in range(0, num_edges, chunk_size):
        if i % (chunk_size * 10) == 0:
            print(f"doing: {i}/{num_edges} ({i / num_edges * 100:.1f}%)")
        chunk = edge_index[:, i:min(i + chunk_size, num_edges)]
        chunk_undirected = to_undirected(chunk)
        processed_chunks.append(chunk_undirected)
        del chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    edge_index_undirected = torch.cat(processed_chunks, dim=1)
    return edge_index_undirected

def chunked_add_self_loops(edge_index, num_nodes, chunk_size=1000000):
    print("add loop...")
    self_loops = torch.stack([
        torch.arange(num_nodes, device=edge_index.device),
        torch.arange(num_nodes, device=edge_index.device)
    ])

    self_loop_chunks = []
    for i in range(0, num_nodes, chunk_size):
        chunk = self_loops[:, i:min(i + chunk_size, num_nodes)]
        self_loop_chunks.append(chunk)

    self_loops_all = torch.cat(self_loop_chunks, dim=1)

    edge_index_with_loops = torch.cat([edge_index, self_loops_all], dim=1)

    print(f"add loop funish {edge_index_with_loops.shape[1]}")
    return edge_index_with_loops


def precompute_chunked_sp_features(edge_index, num_nodes, chunk_size=1000000, k=1, filename_prefix='paper100M'):
    num_chunks = (num_nodes + chunk_size - 1) // chunk_size
    all_sp_features = torch.zeros((num_nodes, k), dtype=torch.long)
    for chunk_idx in range(num_chunks):
        print(f"doing: {chunk_idx}/{num_chunks} ({chunk_idx / num_chunks * 100:.1f}%)")
        start_node = chunk_idx * chunk_size
        end_node = min((chunk_idx + 1) * chunk_size, num_nodes)
        mask = (edge_index[0] >= start_node) & (edge_index[0] < end_node) & \
               (edge_index[1] >= start_node) & (edge_index[1] < end_node)
        chunk_edges = edge_index[:, mask]
        if chunk_edges.shape[1] == 0:
            print(f"batch {chunk_idx} not edge")
            continue
        chunk_nodes = end_node - start_node
        remapped_edges = chunk_edges - start_node
        print(f"compute patch {chunk_idx} S: {chunk_nodes}")
        filename = f'{filename_prefix}_chunk_{chunk_idx}'
        try:
            S_chunk = save_or_load_shortest_paths(remapped_edges, chunk_nodes, k=k, filename=filename)
            all_sp_features[start_node:end_node] = S_chunk
        except Exception as e:
            print(f"compute patch {chunk_idx} s error: {e}")
        del chunk_edges, remapped_edges
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"patch shape: {all_sp_features.shape}")

    return all_sp_features

def process_graph_in_chunks(dataset, undirected_chunk_size=10000000, self_loop_chunk_size=1000000):
    print("start patch transform...")
    edge_index = dataset.graph["edge_index"]
    num_nodes = dataset.graph["num_nodes"]
    node_feat = dataset.graph["node_feat"]
    label = dataset.label
    print(f"v graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    edge_index_undirected = chunked_to_undirected(edge_index, num_nodes, undirected_chunk_size)
    del edge_index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    edge_index_final = chunked_add_self_loops(edge_index_undirected, num_nodes, self_loop_chunk_size)
    del edge_index_undirected
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #S = precompute_chunked_sp_features(edge_index_final,num_nodes,1000000)
    #node_feat = torch.cat((node_feat, S), dim=1)
    ones = torch.rand((num_nodes, 1))
    node_feat = torch.cat((node_feat, ones), dim=1)
    data = Data(x=node_feat, edge_index=edge_index_final, y=label)
    print("graph finish!")
    return data

def main():
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    print("Start loading dataset")
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    data = process_graph_in_chunks( dataset, undirected_chunk_size=20000000,  self_loop_chunk_size=2000000)
    if args.rand_split:
        split_idx_lst = [
            dataset.get_idx_split(
                train_prop=args.train_prop, valid_prop=args.valid_prop
            )
            for _ in range(1)
        ]
    elif args.rand_split_class:
        split_idx_lst = [
            dataset.get_idx_split(
                split_type="class", label_num_per_class=args.label_num_per_class
            )
            for _ in range(1)
        ]
    elif args.dataset in ["ogbn-papers100M"]:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(1)]
    else:
        split_idx_lst = load_fixed_splits(
            args.data_dir, dataset, name=args.dataset, protocol=args.protocol
        )

    split_idx = split_idx_lst[0]
    input_channels = data.x.shape[1]-1
    #input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    if len(data.y.shape) > 1:
        output_channels = max(output_channels, data.y.shape[1])
    data.y[data.y.isnan()] = 404.0
    data.y = data.y.view(-1)
    data.y = data.y.to(torch.long)
    device = torch.device(f"cuda:{args.device}")
    print("Finish loading dataset")
    print("Start sampling")
    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx["train"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )
    print("Finish train_loader")
    valid_loader = NeighborLoader(
        data,
        input_nodes=split_idx["valid"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
    )
    print("Finish valid_loader")
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx["test"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
    )
    print("Finish sampling")
    # --- init model --- #
    model = parse_method(args, output_channels, input_channels, device)
    loss_func = nn.CrossEntropyLoss()
    results = []
    best_model = None
    for run in range(args.runs):
        best_val_acc, best_test_acc, best_epoch, highest_test_acc = 0, 0, 0, 0
        if args.use_pretrained:
            print(f"Load pretrained model from {args.model_dir}.")
            model.load_state_dict(torch.load(args.model_dir, map_location=device))
        else:
            model.reset_parameters()
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print("Strat Traning")
        for e in range(args.epochs):
            # --- train --- #
            tot_loss = 0
            for graph in tqdm(train_loader, desc="Training"):
                graph = graph.to(device)
                loss = train(model, graph, loss_func, optimizer, graph.batch_size)
                tot_loss += loss
            # --- valid ---#
            valid_correct, valid_tot = 0, 0
            for graph in valid_loader:
                graph = graph.to(device)
                correct, tot = evaluate(model, graph, graph.batch_size,args)
                valid_correct += correct
                valid_tot += tot
            val_acc = valid_correct / valid_tot
            # --- test --- #
            test_correct, test_tot = 0, 0
            for graph in test_loader:
                graph = graph.to(device)
                correct, tot = evaluate(model, graph, graph.batch_size,args)
                test_correct += correct
                test_tot += tot
            test_acc = test_correct / test_tot
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = e + 1
                if args.save_model:
                    best_model = copy.deepcopy(model)
            if test_acc > highest_test_acc:
                highest_test_acc = test_acc
            if args.display_step > 0 and (e == 0 or (e + 1) % args.display_step == 0):
                print(
                    f"Epoch: {e+1:02d} "
                    f"Loss: {tot_loss:.4f} "
                    f"Valid acc: {val_acc * 100:.2f}% "
                    f"Test acc: {test_acc * 100:.2f}%"
                )

        print(f"Run {run+1:02d}")
        print(f"Best epoch: {best_epoch}")
        print(f"Highest test acc: {highest_test_acc * 100:.2f}%")
        print(f"Valid acc: {best_val_acc * 100:.2f}%")
        print(f"Test acc: {best_test_acc * 100:.2f}%")

        results.append([highest_test_acc, best_val_acc, best_test_acc])
    results = torch.as_tensor(results) * 100  # (runs, 3)
    print_str = f"{results.shape[0]} runs: "
    r = results[:, 0]
    print_str += f"Highest Test: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 1]
    print_str += f"Best Valid: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 2]
    print_str += f"Final Test: {r.mean():.2f} ± {r.std():.2f} "
    print_str += f"Best epoch: {best_epoch}"
    print(print_str)

    out_folder = "results"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if args.save_model:
        save_folder = "models"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        path = os.path.join(
            save_folder, f"{args.dataset}_{args.method}_{args.epochs}.pt"
        )
        torch.save(best_model.state_dict(), path)
if __name__ == "__main__":
    main()

