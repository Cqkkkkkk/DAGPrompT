import torch
import random
import numpy as np

def set_seed_global(seed: int, force_deter=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if force_deter:
        torch.use_deterministic_algorithms(True)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'