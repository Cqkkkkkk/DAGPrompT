import os
import pdb
import torch
import pickle
import numpy as np
import pickle as pkl
import torch.nn as nn
import networkx as nx
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling, subgraph, k_hop_subgraph
from config import cfg


def edge_index_to_sparse_matrix(edge_index: torch.LongTensor, num_node: int):
    node_idx = torch.LongTensor([i for i in range(num_node)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    sp_adj = torch.sparse.FloatTensor(
        edge_index,
        torch.ones(edge_index.size(1)),
        torch.Size((num_node, num_node))
    )

    return sp_adj


def prepare_structured_data(graph_data: Data):
    r"""Prepare structured <i,k,j> format link prediction data"""
    node_idx = torch.LongTensor([i for i in range(graph_data.num_nodes)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([graph_data.edge_index, self_loop], dim=1)
    v, a, b = structured_negative_sampling(edge_index, graph_data.num_nodes)
    data = torch.stack([v, a, b], dim=1)

    # (num_edge, 3)
    #   for each entry (i,j,k) in data, (i,j) is a positive sample while (i,k) forms a negative sample
    return data



def induced_graphs(data, smallest_size=10, largest_size=30):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                         edge_index=data.edge_index, relabel_nodes=True)

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                             edge_index=data.edge_index)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label))
            candidate_nodes = torch.from_numpy(
                np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(
                candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat(
                [torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(
                subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(
                torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(
            subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)
        # print(index)
    return induced_graph_list


def split_induced_graphs(name, data, smallest_size=10, largest_size=30):
    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                         edge_index=data.edge_index, relabel_nodes=True)

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                             edge_index=data.edge_index)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label))
            candidate_nodes = torch.from_numpy(
                np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(
                candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat(
                [torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(
                subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(
                torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(
            subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]
        if cfg.model.adaptive_adj:
            induced_graph = Data(x=x, edge_index=sub_edge_index,
                                y=current_label, index=index, node_index_saves=subset)
        else:
            induced_graph = Data(x=x, edge_index=sub_edge_index,
                                y=current_label, index=index)
        induced_graph_list.append(induced_graph)
        if index % 50 == 0:
            print(index)

    dir_path = f'./datasets/induced_graph/{name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, 'induced_graph.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        pickle.dump(induced_graph_list, f)


def node_sample_and_save(data, num_shot, folder, num_classes):
    labels = data.y.to('cpu')
    train_index = []
    for i in range(num_classes):
        class_indices = torch.where(labels == i)[0]
        class_indices = class_indices[torch.randperm(class_indices.size(0))]
        train_index.extend(class_indices[:num_shot])
    
    train_index = torch.tensor(train_index)
    train_labels = labels[train_index]

    # remaining_index = torch.tensor([x for x in torch.arange(data.num_nodes) if x not in train_index])
    # num_validation = int(remaining_index.size(0) * 0.5)
    # val_index = remaining_index[torch.randperm(num_validation)]
    # test_index = torch.tensor([x for x in remaining_index if x not in val_index])
        
    test_index = torch.tensor([x for x in torch.arange(data.num_nodes) if x not in train_index])
    test_labels = labels[test_index]

    # 保存文件
    torch.save(train_index, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    # torch.save(val_index, os.path.join(folder, 'val_idx.pt'))
    # torch.save(val_labels, os.path.join(folder, 'val_labels.pt'))
    torch.save(test_index, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))



def graph_sample_and_save(dataset, k, folder, num_classes):

    # 计算测试集的数量（例如90%的图作为测试集）
    num_graphs = len(dataset)
    num_test = int(0.8 * num_graphs)
    labels = torch.tensor([graph.y.item() for graph in dataset])

    # 随机选择测试集的图索引
    all_indices = torch.randperm(num_graphs)
    test_indices = all_indices[:num_test]
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    test_labels = labels[test_indices]
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_indices = all_indices[num_test:]

    # 从剩下的10%的图中为训练集选择每个类别的k个样本
    train_indices = []
    for i in range(num_classes):
        # 选出该类别的所有图
        class_indices = [idx for idx in remaining_indices if labels[idx].item() == i]
        # 如果选出的图少于k个，就取所有该类的图
        selected_indices = class_indices[:k] 
        train_indices.extend(selected_indices)

    # 随机打乱训练集的图索引
    train_indices = torch.tensor(train_indices)
    shuffled_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_indices]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    train_labels = labels[train_indices]
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))


def graph_sample_and_save(dataset, k, folder, num_classes):

    # 计算测试集的数量（例如90%的图作为测试集）
    num_graphs = len(dataset)
    num_test = int(0.8 * num_graphs)
    labels = torch.tensor([graph.y.item() for graph in dataset])

    # 随机选择测试集的图索引
    all_indices = torch.randperm(num_graphs)
    test_indices = all_indices[:num_test]
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    test_labels = labels[test_indices]
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_indices = all_indices[num_test:]

    # 从剩下的10%的图中为训练集选择每个类别的k个样本
    train_indices = []
    for i in range(num_classes):
        # 选出该类别的所有图
        class_indices = [
            idx for idx in remaining_indices if labels[idx].item() == i]
        # 如果选出的图少于k个，就取所有该类的图
        selected_indices = class_indices[:k]
        train_indices.extend(selected_indices)

    # 随机打乱训练集的图索引
    train_indices = torch.tensor(train_indices)
    shuffled_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_indices]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    train_labels = labels[train_indices]
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form


def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix(
            (np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/Planetoid/Cora/raw/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "./data/Planetoid/Cora/raw/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack(
                (np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# def standardize_data(f, train_mask):
#     """Standardize feature matrix and convert to tuple representation"""
#     # standardize data
#     f = f.todense()
#     mu = f[train_mask == True, :].mean(axis=0)
#     sigma = f[train_mask == True, :].std(axis=0)
#     f = f[:, np.squeeze(np.array(sigma > 0))]
#     mu = f[train_mask == True, :].mean(axis=0)
#     sigma = f[train_mask == True, :].std(axis=0)
#     f = (f - mu) / sigma
#     return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
