import os
import pdb
import time
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from tasker.base import BaseTask
from utils.loss import DAGPromptTuningLoss
from utils.data_loader import load4graph
from utils.data_process import graph_sample_and_save
from prompts.eval import DAGPromptEvaluator

from config import cfg


class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = 'graph'
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
            k_shot_folder = f'./datasets/sample_data/Graph/{self.dataset_name}/{self.num_shot}_shot/{k}'
            os.makedirs(k_shot_folder, exist_ok=True)
            graph_sample_and_save(self.dataset, self.num_shot,
                                 k_shot_folder, self.output_dim)

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
            self.input_dim, self.output_dim, self.dataset = load4graph(self.dataset_name, self.num_shot)

    def DAGPromptTrain(self, train_loader):
        self.prompt.train()
        self.gnn.train()
        # mark_only_lora_as_trainable(self.gnn)
        # pdb.set_trace()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        label_num = self.output_dim
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            out_embeddings = self.gnn(batch.x, batch.edge_index, batch.batch,
                                      prompt=self.prompt, prompt_type='dagprompt')

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
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            data_dir = "./datasets/sample_data/Graph/{}/{}_shot/{}/".format(
                self.dataset_name, self.num_shot, i)
            train_index = torch.load(os.path.join(data_dir, "train_idx.pt")).type(
                torch.long).to(self.device)
            test_index = torch.load(os.path.join(data_dir, "test_idx.pt")).type(
                torch.long).to(self.device)


            if self.prompt_type in ['gprompt', 'allinone', 'gpf', 'gpf-plus', 'dagprompt']:
                train_dataset = self.dataset[train_index]
                test_dataset = self.dataset[test_index]
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

                train_loader = DataLoader(train_dataset, batch_size=cfg.optim.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=cfg.optim.batch_size, shuffle=False)

        
            patience = cfg.optim.patience
            best_loss = 1e9
            cnt_wait = 0
            with tqdm(range(1, self.epochs)) as tq:
                for epoch in tq:
                    if self.prompt_type == 'dagprompt':
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

            # exit(1)
            if self.prompt_type == 'dagprompt':
                test_acc = DAGPromptEvaluator(
                    test_loader, self.gnn, self.prompt, center, self.device)
            # print("Test accuracy: {:.2f} ".format(test_acc * 100))
            test_accs.append(test_acc)

        mean_test_acc = np.mean(test_accs) * 100
        std_test_acc = np.std(test_accs) * 100
        print(
            f"Dataset {cfg.dataset.name}: {mean_test_acc:.2f} | std {std_test_acc:.2f}")
        