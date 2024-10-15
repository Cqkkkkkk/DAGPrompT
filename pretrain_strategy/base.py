import os
import torch
from tqdm import tqdm
from model.backbones import GCN, GAT, GraphSAGE


class PreTrainBase(torch.nn.Module):
    def __init__(self, gnn_type, dataset_name, hidden_dim, num_layer, epochs, lr=1e-03, wd=5e-6, device=1):
        super().__init__()
        self.gnn_type = gnn_type
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.wd = wd

    def initialize_gnn(self, input_dim, hid_dim):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=input_dim, hidden_dim=hid_dim,
                           num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=input_dim, hidden_dim=hid_dim,
                           num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=input_dim,
                                 hidden_dim=hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)
        # self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

    # def initialize_optimizer(self):
    #     self.optimizer = Adam(self.gnn.parameters(),
    #                           lr=self.lr, weight_decay=self.wd)
    def initialize_optimizer(self):
        raise NotImplementedError

    def pretrain_one_epoch(self):
        raise NotImplementedError
    
    def pretrain(self):
        train_loss_min = 1000000
        with tqdm(range(self.epochs)) as tq:
            for epoch in tq:
                self.optimizer.zero_grad()
                train_loss = self.pretrain_one_epoch()
                infos = {
                    'epoch': epoch,
                    'train_loss': train_loss
                }
                tq.set_postfix(infos)

                if train_loss_min > train_loss:
                    train_loss_min = train_loss
                    ckpt_path = self.save_model(self.dataset_name, self.gnn_type, self.pretrain_type, self.hidden_dim)
        print(f"Best ckpt saved at {ckpt_path}")
            

    def save_model(self, dataset_name, gnn_type, pretrain_type, hidden_dim):
        folder_path = f"./ckpts/{dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        ckpt_path = os.path.join(folder_path, f"{pretrain_type}-{gnn_type}-h{hidden_dim}.pth")
        torch.save(self.gnn.state_dict(), ckpt_path)
        return ckpt_path