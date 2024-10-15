import pdb
import torch
from torch.optim import Adam
from model.backbones import GCN, GAT, GraphSAGE, GCNLoRA, GATLoRA
from model.lora import filter_lora_parameters
from prompts.dagprompt import DAGPrompt, ParameterizedMultiHopCenterEmbedding
from config import cfg


class BaseTask:
    def __init__(
        self,
        pre_train_model_path,
        dataset_name,
        gnn_type='GCN',
        num_layers=2,
        hidden_dim=128,
        prompt_type='none',
        num_shot=10,
        epochs=100,
        lr=1e-3,
        wd=5e-6,
        device=1,
        r=0
    ):
        self.pre_train_model_path = pre_train_model_path
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layers
        self.hidden_dim = hidden_dim
        self.prompt_type = prompt_type
        self.num_shot = num_shot
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = torch.device(device)
        self.r = r
        self.initialize_lossfn()

    def initialize_optimizer(self):
        if self.prompt_type == 'none':
            param_group = []
            param_group.append({"params": self.gnn.parameters()})
            param_group.append({"params": self.answering.parameters()})
            # self.optimizer = Adam(model_param_group, lr=0.005, weight_decay=5e-4)
            self.optimizer = Adam(param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type == 'allinone':
            prompt_parameters = filter(
                lambda p: p.requires_grad, self.prompt.parameters()
            )
            # lr=0.001, weight_decay=0.00001
            self.prompt_optimizer = Adam(prompt_parameters, lr=self.lr, weight_decay=self.wd)
            answer_parameters = filter(
                lambda p: p.requires_grad, self.answering.parameters()
            )
            # lr=0.001, weight_decay=0.00001
            self.answer_optimizer = Adam(answer_parameters, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['gpf', 'gpf-plus']:
            param_group = []
            param_group.append({"params": self.prompt.parameters()})
            param_group.append({"params": self.answering.parameters()})
            # lr=0.005, weight_decay=5e-4
            self.optimizer = Adam(param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['gprompt', 'gppt']:
            # lr=0.01, weight_decay=5e-4
            param_group = []
            param_group.append({"params": self.prompt.parameters()})
            # param_group.append({"params": self.gnn.parameters()})
            self.prompt_optimizer = Adam(param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['dagprompt']:
            param_group = []
            param_group.append({"params": filter_lora_parameters(self.gnn)})
            param_group.append({"params": self.prompt.parameters()})
            param_group.append({"params": self.param_center_embeddings.parameters()})
            if self.gnn.global_edge_weights is not None:
                # pdb.set_trace()
                param_group.append({"params": self.gnn.global_edge_weights.parameters()})
            self.optimizer = Adam(param_group, lr=self.lr, weight_decay=self.wd)

        # Calculate the total number of tunable parameters
        total_tunable_params = 0
        for param in param_group:
            for p in param['params']:
                total_tunable_params += sum(p.numel() for p in p if p.requires_grad)
        print(f'Total tunable parameters: {total_tunable_params}')
        # exit()

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def initialize_prompt(self):
        if self.prompt_type == 'dagprompt':
            self.prompt = DAGPrompt(
                self.hidden_dim, 
                hop_range=self.num_layer+1,
                alpha=cfg.model.alpha
            ).to(self.device)
            self.param_center_embeddings = ParameterizedMultiHopCenterEmbedding(
                hop_num=self.num_layer+1, 
                label_num=self.output_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)
        else:
            raise NotImplementedError

    def initialize_gnn(self, edge_index=None):
        if self.gnn_type == 'GAT':
            if self.prompt_type == 'dagprompt':
                self.gnn = GATLoRA(input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim, num_layer=self.num_layer, r=self.r)
            else:
                self.gnn = GAT(input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            if self.prompt_type == 'dagprompt':
                self.gnn = GCNLoRA(input_dim=self.input_dim,
                                   hidden_dim=self.hidden_dim, num_layer=self.num_layer, r=self.r, edge_index=edge_index)
                print('In adaptive adj:' + "True" if edge_index is not None else "False") 
            else:
                self.gnn = GCN(input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)

        if 'none' not in self.pre_train_model_path:
            # pdb.set_trace()
            if self.gnn_type not in self.pre_train_model_path:
                raise ValueError(
                    f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path:
                raise ValueError(
                    f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")
            # pdb.set_trace()
            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device), strict=False)
            print("Successfully loaded pre-trained weights!")
