import pdb
import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Dataset


class ArxivYear(Dataset):
    def __init__(self, root='../datasets', nclass=5):
        super(ArxivYear, self).__init__(root)
        self.data = load_arxiv_year_dataset(root=root, nclass=nclass)
        self.num_nodes = self.data.x.shape[0]
        self.nclass = nclass
        
    def len(self):
        return len(self.data.y)
    
    def get(self, idx):
        return self.data

    @property
    def num_classes(self) -> int:
        return self.nclass



def load_arxiv_year_dataset(root, nclass):
    dataset = PygNodePropPredDataset(root=root, name='ogbn-arxiv') 
    data = dataset[0]
    label = even_quantile_labels(data.node_year.flatten().numpy(), nclass, verbose=False)
    data.y = torch.as_tensor(label)

    return data


def even_quantile_labels(vals, nclasses, verbose=True):
    """ 
    partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

if __name__ == '__main__':
    dataset = ArxivYear()
    print(dataset[0], dataset.num_nodes, dataset.num_features, dataset.num_classes)