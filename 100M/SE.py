import torch

from collections import deque
from torch_geometric.utils import to_undirected, remove_self_loops


def compute_shortest_paths(edge_index, num_nodes, k):
    components = find_connected_components(edge_index, num_nodes)
    print(f"fund {len(components)} Conjugate Component")

    if len(components) > 1:
        print("add not connect subgraph...")
        edge_index = connect_components(edge_index, components)

    all_nodes = edge_index.view(-1)
    degrees = torch.bincount(all_nodes, minlength=num_nodes)
    min_degree = degrees.min().item()

    candidates = (degrees == min_degree).nonzero(as_tuple=True)[0]
    if len(candidates) < k:
        additional_nodes = (degrees != min_degree).nonzero(as_tuple=True)[0][:k - len(candidates)]
        candidates = torch.cat((candidates, additional_nodes))

    V = candidates[:k]

    adj_list = build_adjacency_list(edge_index, num_nodes)

    distances_matrix = torch.zeros((num_nodes, k), dtype=torch.long)

    for idx, node in enumerate(V):
        print('node:',node)
        distances = bfs(adj_list, node.item(), num_nodes)
        distances_matrix[:, idx] = distances

    return distances_matrix

def find_connected_components(edge_index, num_nodes):
    adj_list = build_adjacency_list(edge_index, num_nodes)
    visited = [False] * num_nodes
    components = []
    for node in range(num_nodes):
        if not visited[node]:
            component = []
            queue = deque([node])
            visited[node] = True
            while queue:
                n = queue.popleft()
                component.append(n)
                for neighbor in adj_list[n]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)
    return components

def connect_components(edge_index, components):
    new_edges = []
    for i in range(len(components) - 1):
        u = components[i][0]
        v = components[i+1][0]
        new_edges.append([u, v])
        new_edges.append([v, u])
    new_edges = torch.tensor(new_edges, dtype=edge_index.dtype).t()
    return torch.cat([edge_index, new_edges], dim=1)

def build_adjacency_list(edge_index, num_nodes):
    adj_list = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if v not in adj_list[u]:
            adj_list[u].append(v)
        if u not in adj_list[v]:
            adj_list[v].append(u)
    return adj_list



def bfs(adj_list, start_node, num_nodes):
    distances = torch.full((num_nodes,), float('inf'))
    distances[start_node] = 0
    queue = deque([start_node])
    visited_order = []

    while queue:
        node = queue.popleft()
        visited_order.append(node)
        for neighbor in adj_list[node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    sorted_distances = torch.zeros_like(distances)
    for i, node in enumerate(visited_order):
        sorted_distances[node] = distances[node]

    return sorted_distances


def save_or_load_shortest_paths(edge_index, n,k, filename):
    S = compute_shortest_paths(edge_index, n, k)

    return S
