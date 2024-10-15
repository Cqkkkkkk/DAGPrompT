import pdb
import torch
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader

from config import cfg
from pretrain_strategy.base import PreTrainBase
from utils.loss import DAGPromptLinkPredictionLoss
from utils.data_process import edge_index_to_sparse_matrix, prepare_structured_data
from utils.data_loader import load4link_prediction_single_graph, load4link_prediction_multi_graph


class EdgepredDAGPromptPretrain(PreTrainBase):
    """
    Pretraining strategy (edge prediction) for Distribution-aware Graph Prompting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_type = 'dagprompt'
        self.dataloader = self.generate_loader_data()

        self.initialize_gnn(self.input_dim, self.hidden_dim)
        self.fc = Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.initialize_optimizer()

    def initialize_optimizer(self):
        parameters_group = list(self.gnn.parameters()) + list(self.fc.parameters())
        self.optimizer = torch.optim.Adam(parameters_group, lr=self.lr, weight_decay=self.wd)
        

    def generate_loader_data(self):
        if self.dataset_name in ['pubmed', 'citeseer', 'cora', 'computers', 'photo', 'texas', 'wisconsin', 'cornell', 'chameleon', 'squirrel', 'arxiv-year']:
            load_funcion = load4link_prediction_single_graph
        elif self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
            load_funcion = load4link_prediction_multi_graph
        else:
            raise NotImplementedError

        self.data, _, _, self.input_dim, self.output_dim = load_funcion(
            self.dataset_name)
        self.adj = edge_index_to_sparse_matrix(
            self.data.edge_index,
            self.data.x.shape[0]
        ).to(self.device)
        data = prepare_structured_data(self.data)
        return DataLoader(TensorDataset(data), batch_size=cfg.optim.batch_size, shuffle=True)

    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        self.gnn.train()
        for _, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            batch = batch[0]
            batch = batch.to(self.device)

            out = self.gnn(self.data.x.to(self.device),
                           self.data.edge_index.to(self.device))

            all_node_emb = self.fc(out)

            all_node_emb = torch.sparse.mm(self.adj, all_node_emb)

            # Embeddings of triplet (target_node, positive_node, negative_node)
            node_emb = all_node_emb[batch[:, 0]]
            pos_emb = all_node_emb[batch[:, 1]]
            neg_emb =  all_node_emb[batch[:, 2]]

            loss = DAGPromptLinkPredictionLoss(node_emb, pos_emb, neg_emb)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step
