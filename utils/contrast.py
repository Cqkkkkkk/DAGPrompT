import torch
import numpy as np
from copy import deepcopy
from torch_geometric.data import Data


def generate_corrupted_graph_via_drop_node(data, aug_ratio=0.1):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]

    new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
    new_x = data.x[idx_nondrop]
    return Data(x=new_x, edge_index=new_edge_index)


def generate_corrupted_graph_via_drop_edge(data, aug_ratio):
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    idx_delete = np.random.choice(
        edge_num, (edge_num - permute_num), replace=False)
    new_edge_index = data.edge_index[:, idx_delete]

    return Data(x=data.x, edge_index=new_edge_index)


def genereate_corrupted_graph_via_shuffle_X(data):
    """
        Perturb one graph by row-wise shuffling X (node features) without changing the A (adjacency matrix).
    """
    node_num = data.x.shape[0]
    idx = np.random.permutation(node_num)
    new_x = data.x[idx, :]
    return Data(x=new_x, edge_index=data.edge_index)


def generate_corrupted_graph(graph_data, aug='dropE', aug_ratio=0.1):
    if aug == 'dropN':
        return generate_corrupted_graph_via_drop_node(graph_data, aug_ratio)
    elif aug == 'dropE':
        return generate_corrupted_graph_via_drop_edge(graph_data, aug_ratio)
    elif aug == 'shuffleX':
        return genereate_corrupted_graph_via_shuffle_X(graph_data)
    else:
        raise KeyError("[Pretrain] Encounter unsupported corruption method")
