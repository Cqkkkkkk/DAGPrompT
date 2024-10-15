import torch
from torch_geometric.data import Data, Batch, ClusterData
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, TUDataset, Flickr, WikipediaNetwork, WebKB
from torch_geometric.utils import negative_sampling, to_undirected

from config import cfg
from data.arxiv_year import ArxivYear
import pdb

def load_base_dataset(dataname):
    """
    Load the base dataset
    """
    if dataname in ['pubmed', 'citeseer', 'cora']:
        dataset = Planetoid(root=cfg.dataset.root,
                            name=dataname, transform=NormalizeFeatures())
    elif dataname in ['computers', 'photo']:
        dataset = Amazon(root=cfg.dataset.root, name=dataname)
    elif dataname in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root=cfg.dataset.root, name=dataname)
    elif dataname in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=cfg.dataset.root, name=dataname)
    elif dataname == 'arxiv-year':
        dataset = ArxivYear()
    elif dataname == 'Reddit':
        raise NotImplementedError
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        raise NotImplementedError
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        raise NotImplementedError
        dataset = Flickr(root='data/Flickr')
    elif dataname in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
        # raise NotImplementedError
        dataset = TUDataset(root='data/TUDataset', name=dataname)

    return dataset


def load4link_prediction_single_graph(dataname, num_per_samples=1):
    """
    Data loader for link prediction task on single graph dataset
    """
    dataset = load_base_dataset(dataname)
    data = dataset[0]

    input_dim = dataset.num_features
    output_dim = dataset.num_classes

    # Perform negative sampling to generate negative neighbor samples
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat(
        [torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))],
        dim=0
    )

    return data, edge_label, edge_index, input_dim, output_dim


def load4link_prediction_multi_graph(dataname, num_per_samples=1):
    """
    Data loader for link prediction task on multi-graph dataset
    """
    dataset = load_base_dataset(dataname)

    input_dim = dataset.num_features
    output_dim = 2  # link prediction的输出维度应该是2，0代表无边，1代表右边
    data = Batch.from_data_list(dataset)

    # Perform negative sampling to generate negative neighbor samples
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat(
        [torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))],
        dim=0
    )

    return data, edge_label, edge_index, input_dim, output_dim


def load4node(dataname, num_shot=10):
    """
    Data loader for downstream node classification task
    """
    dataset = load_base_dataset(dataname)
    data = dataset[0]  # Get the first graph object.

    # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for label in data.y.unique():
        label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

        label_indices = label_indices[torch.randperm(len(label_indices))]
        train_indices = label_indices[:num_shot]
        train_mask[train_indices] = True
        remaining_indices = label_indices[100:]

        test_indices = remaining_indices

        # val_mask[val_indices] = True
        test_mask[test_indices] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    # data.val_mask = val_mask

    return data, dataset


def load4nodeClsuter(dataname='CiteSeer', num_parts=200):
    """
    Data loader for node-level pretrain strategies (GraphCL and SimGrace) with clustering
    """
    dataset = load_base_dataset(dataname)
    data = dataset[0]

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    graph_list = list(ClusterData(data=data, num_parts=num_parts))

    return graph_list, input_dim


def load4graph(dataset_name, pretrained=False):
    """
    For GraphCL with multi-graph dataset. 
    A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    graph_list = [data for data in dataset]

    input_dim = dataset.num_features
    out_dim = dataset.num_classes

    if pretrained == True:
        return input_dim, out_dim, graph_list
    else:
        return input_dim, out_dim, dataset

