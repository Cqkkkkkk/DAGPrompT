import os
import pdb
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from tasker.base import BaseTask
from utils.loss import DAGPromptTuningLoss
from utils.data_loader import load4node
from utils.data_process import split_induced_graphs, node_sample_and_save
from prompts.eval import DAGPromptEvaluator
from model.lora import mark_only_lora_as_trainable

from config import cfg


class NodeTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = 'node'
        # pdb.set_trace()
        self.dataset_name = kwargs['dataset_name']
        self.load_data()
        self.answering = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.output_dim),
                                             torch.nn.Softmax(dim=1)).to(self.device)

        self.generate_few_shot_data()
        self.initialize_gnn()
        self.initialize_prompt()
        self.initialize_optimizer()
        torch.nn.init.xavier_uniform_(self.answering[0].weight)

    def generate_few_shot_data(self):
        for k in range(1, cfg.repeat + 1):
            k_shot_folder = f'./datasets/sample_data/Node/{self.dataset_name}/{self.num_shot}_shot/{k}'
            os.makedirs(k_shot_folder, exist_ok=True)
            node_sample_and_save(self.data, self.num_shot,
                                 k_shot_folder, self.output_dim)

    def load_induced_graph(self):
        self.data, self.dataset = load4node(
            self.dataset_name, num_shot=self.num_shot)
        # self.data.to('cpu')
        self.input_dim = self.dataset.num_features
        self.output_dim = self.dataset.num_classes
        file_path = './datasets/induced_graph/' + \
            self.dataset_name + '/induced_graph.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        else:
            print('Begin split_induced_graphs.')
            split_induced_graphs(self.dataset_name, self.data,
                                 smallest_size=10, largest_size=30)
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        return graphs_list

    def load_data(self):
        self.data, self.dataset = load4node(
            self.dataset_name, num_shot=self.num_shot)
        self.data.to(self.device)
        self.input_dim = self.dataset.num_features
        self.output_dim = self.dataset.num_classes

 
    def DAGPromptTrain(self, train_loader):
        self.prompt.train()
        self.gnn.train()
        if not cfg.model.adaptive_adj:   
            mark_only_lora_as_trainable(self.gnn)
        # pdb.set_trace()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        label_num = self.output_dim
        for batch in train_loader:
            
            node_index_saves = batch.node_index_saves if cfg.model.adaptive_adj else None

            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch,
                                      prompt=self.prompt, prompt_type='dagprompt', 
                                      node_index_saves=node_index_saves)
            # )

            # centers, class_counts = center_embedding_multihop(out_embeddings, batch.y, label_num, self.num_layer + 1)
            centers, class_counts = self.param_center_embeddings(
                out_embeddings, batch.y)
            # For each class, calculate the average embedding across the batchs
            if accumulated_centers is None:
                accumulated_centers = centers
                accumulated_counts = class_counts
            else:
                accumulated_centers += centers * class_counts
                accumulated_counts += class_counts
            criterion = DAGPromptTuningLoss()
            loss = criterion(out_embeddings, centers, batch.y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        mean_centers = accumulated_centers / (accumulated_counts + 1e-8)
        # pdb.set_trace()
        return total_loss / len(train_loader), mean_centers

    def run(self):
        test_accs = []
        for i in range(1, cfg.repeat + 1):
            # Load data
            edge_index_init = self.data.edge_index if cfg.model.adaptive_adj else None
            self.initialize_gnn(edge_index_init)
            # self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            data_dir = "./datasets/sample_data/Node/{}/{}_shot/{}/".format(
                self.dataset_name, self.num_shot, i)
            train_index = torch.load(os.path.join(data_dir, "train_idx.pt")).type(
                torch.long).to(self.device)
            test_index = torch.load(os.path.join(data_dir, "test_idx.pt")).type(
                torch.long).to(self.device)

            # for all-in-one and Gprompt we use k-hop subgraph
            if self.prompt_type in ['gprompt', 'allinone', 'gpf', 'gpf-plus', 'dagprompt']:
                graphs_list = self.load_induced_graph()
                train_graphs = []
                test_graphs = []

                for graph in graphs_list:
                    if graph.index in train_index:
                        train_graphs.append(graph)
                    elif graph.index in test_index:
                        test_graphs.append(graph)

                train_loader = DataLoader(
                    train_graphs, batch_size=cfg.optim.batch_size, shuffle=True)
                eval_batch_size = cfg.optim.eval_batch_size if cfg.optim.eval_batch_size > 0 else cfg.optim.batch_size
                test_loader = DataLoader(
                    test_graphs, batch_size=eval_batch_size, shuffle=False)
                # print("prepare induce graph data is finished!")

            patience = cfg.optim.patience
            best_loss = 1e9
            cnt_wait = 0
            with tqdm(range(1, self.epochs)) as tq:
                for epoch in tq:
                    if self.prompt_type == 'none':
                        loss = self.GNNtrain(self.data, train_index)
                    elif self.prompt_type == 'gppt':
                        loss = self.GPPTtrain(self.data, train_index)
                    elif self.prompt_type == 'allinone':
                        loss = self.AllInOneTrain(train_loader)
                    elif self.prompt_type in ['gpf', 'gpf-plus']:
                        loss = self.GPFTrain(train_loader)
                    elif self.prompt_type == 'gprompt':
                        loss, center = self.GpromptTrain(train_loader)
                    elif self.prompt_type == 'dagprompt':
                        loss, center = self.DAGPromptTrain(train_loader)
                    
                    else:
                        raise NotImplementedError
                    if loss < best_loss:
                        best_loss = loss
                        cnt_wait = 0
                    else:
                        cnt_wait += 1
                        if cnt_wait == patience:
                            break
                    infos = {
                        'epoch': epoch,
                        'loss': loss
                    }
                    tq.set_postfix(infos)

            if self.prompt_type == 'dagprompt':
                test_acc = DAGPromptEvaluator(
                    test_loader, self.gnn, self.prompt, center, self.device)
            # print("Test accuracy: {:.2f} ".format(test_acc * 100))
            test_accs.append(test_acc)

        mean_test_acc = np.mean(test_accs) * 100
        std_test_acc = np.std(test_accs) * 100
        print(
            f"Dataset {cfg.dataset.name}: {mean_test_acc:.2f} | std {std_test_acc:.2f}")
        