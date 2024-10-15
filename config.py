import logging
import os
from yacs.config import CfgNode as CN


# Global config object
cfg = CN()

def set_cfg(cfg):
    r'''
     This function sets the default config value.
     1) Note that for an experiment, only part of the arguments will be used
     The remaining unused arguments won't affect anything.
     2) We support *at most* two levels of configs, e.g., cfg.dataset.name
     '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    
    # Select the device, cpu or cuda 
    cfg.device = 'cuda:2'
    
    # Random seed for data split
    cfg.seed = 42

    # Random seed for model initialization
    cfg.model_seed = 42
    
    # Num splits for few-shot learning
    cfg.num_split = 5

    # Num labels per class for few-shot learning
    cfg.num_shot = 20

    # Ratio of nodes for validation (exclude train nodes) for few-shot learning
    cfg.val_ratio = 0.5

    # Path to the pre-trained model, only effective for downstream tasks
    cfg.pre_train_model_path = None

    # Repeat experitment times
    cfg.repeat = 5

    # Node classification, or graph classification task?
    cfg.task = 'node' # node or graph

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #

    cfg.dataset = CN()

    cfg.dataset.name = 'pubmed'

    # Modified automatically by code, no need to set
    cfg.dataset.num_nodes = -1

    # Modified automatically by code, no need to set
    cfg.dataset.num_classes = -1

    # Dir to load the dataset. If the dataset is downloaded, it is in root
    cfg.dataset.root = '../datasets'

    # ------------------------------------------------------------------------ #
    # Optimization options
    # ------------------------------------------------------------------------ #

    cfg.optim = CN()

    # Maximal number of epochs
    cfg.optim.epochs = 200

    cfg.optim.patience = 200
   

    # Base learning rate
    cfg.optim.lr = 0.01

    # L2 regularization
    cfg.optim.wd = 5e-4

    # Batch size, only works in minibatch mode
    cfg.optim.batch_size = 128
    cfg.optim.eval_batch_size = -1


    # ------------------------------------------------------------------------ #
    # Model options 
    # ------------------------------------------------------------------------ #
    
    cfg.model = CN()

    # Backbone model to use 
    cfg.model.backbone = 'GCN'

    cfg.model.pretrain_type = 'DGI'

    # Prompt type, in ['gppt', 'gprpmpt', ]
    cfg.model.prompt_type = 'none'
    
    # Hidden layer dim 
    cfg.model.hidden_dim = 128

    # Number of attetnion heads
    cfg.model.num_heads = 8

    # Layer number
    cfg.model.num_layers = 3

    # Dropout rate
    cfg.model.dropout = 0.5

    # Pooling method, sum, mean, max
    cfg.model.pool = 'mean'

    # JK method: how the node features across layers are combined. last, sum, max or concat
    cfg.model.JK = 'last'

    cfg.model.alpha = 0.5

    cfg.model.r = 0

    cfg.model.lg = False
    
    cfg.model.adaptive_adj = False

set_cfg(cfg)