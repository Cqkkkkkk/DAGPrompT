import torch
import pdb
from utils.center_embedding import center_embedding_multihop
from config import cfg


class DAGPrompt(torch.nn.Module):
    def __init__(self, input_dim, hop_range, alpha=0.5):
        super(DAGPrompt, self).__init__()
        self.alpha = alpha
        self.hop_range = hop_range
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(1, input_dim)) for _ in range(hop_range)])
        # self.gamma = torch.nn.Parameter([alpha * (1 - alpha) ** i for i in range(hop_range)])
        if hop_range >= 2:
            gamma = alpha * torch.pow((1 - alpha), torch.arange(hop_range))
            # gamma[-1] = torch.pow((1 - alpha), torch.tensor(hop_range - 1))
        else:
            gamma = torch.tensor([1.0])
        self.gamma = torch.nn.Parameter(gamma)

        self.max_n_num = input_dim
        self.hop_range = hop_range
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.xavier_uniform_(weight)
        if self.hop_range >= 2:
            gamma = self.alpha * torch.pow((1 - self.alpha), torch.arange(self.hop_range))
            # gamma[-1] = torch.pow((1 - alpha), torch.tensor(hop_range - 1))
        else:
            gamma = torch.tensor([1.0])
        self.gamma = torch.nn.Parameter(gamma)

    def forward(self, node_embeddings):
        # Element-wise product
        for i in range(self.hop_range):
            if cfg.model.lg:
                node_embeddings[i] = node_embeddings[i] * self.weights[i] * self.gamma[i]
            else:
                node_embeddings[i] = node_embeddings[i] * self.weights[i]
            # node_embeddings[i] = node_embeddings[i] * self.weights[i] # * self.gamma[i]
        return node_embeddings


class ParameterizedMultiHopCenterEmbedding(torch.nn.Module):
    def __init__(self, hop_num, label_num, hidden_dim) -> None:
        super().__init__()
        self.label_num = label_num
        self.hop_num = hop_num
        self.weight = torch.nn.Parameter(torch.Tensor(hop_num, label_num, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embeddings, labels):
        centers, class_counts = center_embedding_multihop(
            input=node_embeddings, 
            index=labels,
            label_num=self.label_num,
            hop_num=self.hop_num
        )
        return centers + self.weight, class_counts