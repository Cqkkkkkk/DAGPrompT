import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from model.layers import GCNConvLoRA, GATConvLoRA
import pdb


class BaseGNN(torch.nn.Module):
    def __init__(self, GraphConv, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0.5, pool='mean'):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """

        if hidden_dim is None:
            hidden_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hidden_dim

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        if num_layer < 2:
            raise ValueError(
                'GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.layers = torch.nn.ModuleList(
                [GraphConv(hidden_dim, hidden_dim), GraphConv(hidden_dim, out_dim)])
        else:
            layers = [GraphConv(hidden_dim, hidden_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hidden_dim, hidden_dim))
            layers.append(GraphConv(hidden_dim, out_dim))
            self.layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.act = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        x = self.input_proj(x)
        h_list = [x]
        for idx, conv in enumerate(self.layers[0:-1]):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]
       
        # For pre-train there is no batching nore prompting, directly returns the node embedding
        if batch == None:
            return node_emb
        else: # For downstream tasks
            if prompt_type == 'dagprompt':
                node_embeddings = prompt(h_list)
                graph_embeddings = [self.pool(node_embedding, batch.long()) for node_embedding in node_embeddings]
                return torch.stack(graph_embeddings).to(node_emb.device)
            else:
                if prompt_type == 'gprompt':
                    node_emb = prompt(node_emb)
                graph_emb = self.pool(node_emb, batch.long())
                return graph_emb
           

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()



class BaseGNNLoRA(torch.nn.Module):
    def __init__(self, GraphConv, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0.5, pool='mean', r=0):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """

        if hidden_dim is None:
            hidden_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hidden_dim

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        if num_layer < 2:
            raise ValueError(
                'GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.layers = torch.nn.ModuleList(
                [GraphConv(hidden_dim, hidden_dim, r=r), GraphConv(hidden_dim, out_dim, r=r)])
        else:
            layers = [GraphConv(hidden_dim, hidden_dim, r=r)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hidden_dim, hidden_dim, r=r))
            layers.append(GraphConv(hidden_dim, out_dim, r=r))
            self.layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.act = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        x = self.input_proj(x)
        h_list = [x]
        for idx, conv in enumerate(self.layers[0:-1]):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]
       
        # For pre-train there is no batching nore prompting, directly returns the node embedding
        if batch == None:
            return node_emb
        else: # For downstream tasks
            if prompt_type == 'dagprompt':
                node_embeddings = prompt(h_list)
                graph_embeddings = [self.pool(node_embedding, batch.long()) for node_embedding in node_embeddings]
                return torch.stack(graph_embeddings).to(node_emb.device)
            else:
                if prompt_type == 'gprompt':
                    node_emb = prompt(node_emb)
                graph_emb = self.pool(node_emb, batch.long())
                return graph_emb
           

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()




class GCN(BaseGNN):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__(GCNConv, input_dim, hidden_dim,
                         out_dim, num_layer, JK, drop_ratio, pool)


class GCNLoRA(BaseGNNLoRA):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean', r=0, edge_index=None):
        super().__init__(GCNConvLoRA, input_dim, hidden_dim,
                         out_dim, num_layer, JK, drop_ratio, pool, r) 
        if edge_index is not None:
            self.global_edge_weights = nn.ParameterDict({
                f"{edge[0]}_{edge[1]}": nn.Parameter(torch.tensor([1.0]))
                for edge in edge_index.t().tolist()
            })
            self.edge_index_rec = edge_index
        else:
            self.global_edge_weights = None
    
    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None, node_index_saves=None):
        x = self.input_proj(x)
        h_list = [x]
        edge_weight = self.get_batch_edge_weights(edge_index, node_index_saves) \
            if self.global_edge_weights is not None else None

        # pdb.set_trace()
        for idx, conv in enumerate(self.layers[0:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.layers[-1](x, edge_index)
        h_list.append(x)
        node_emb = h_list[-1]
      
        # For pre-train there is no batching nore prompting, directly returns the node embedding
        if batch == None:
            return node_emb
        else: # For downstream tasks
            node_embeddings = prompt(h_list)
            graph_embeddings = [self.pool(node_embedding, batch.long()) for node_embedding in node_embeddings]
            return torch.stack(graph_embeddings).to(node_emb.device)

    def get_batch_edge_weights(self, edge_index_batch, node_index_saves):
        # Fetch edge weights for the current batch from the global dictionary
        # pdb.set_trace()
        edge_weights = []
        # cnt = 0
        # acc = 0
        for edge in edge_index_batch.t().tolist():
            key = f"{node_index_saves[edge[0]]}_{node_index_saves[edge[1]]}"
            if key in self.global_edge_weights:
                edge_weights.append(self.global_edge_weights[key])
                # acc += 1
            else:
                edge_weights.append(torch.tensor([1.0]).to(edge_index_batch.device))
            # cnt += 1
        # print(acc / cnt)
        return torch.cat(edge_weights)


class GAT(BaseGNN):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__(GATConv, input_dim, hidden_dim,
                         out_dim, num_layer, JK, drop_ratio, pool)

class GATLoRA(BaseGNNLoRA):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean', r=0):
        super().__init__(GATConvLoRA, input_dim, hidden_dim,
                         out_dim, num_layer, JK, drop_ratio, pool, r)    

class GraphSAGE(BaseGNN):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__(SAGEConv, input_dim, hidden_dim,
                         out_dim, num_layer, JK, drop_ratio, pool)