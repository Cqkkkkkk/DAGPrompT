import os
import sys
import argparse
from config import cfg
from tasker.node import NodeTask
from tasker.graph import GraphTask
from utils.seed import set_seed_global


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Main entry")
    parser.add_argument('--cfg', dest='cfg_file', default='configs/base.yaml',
                        help='Config file path', type=str)

    if len(sys.argv) == 1:
        print('Now you are using the default configs.')
        parser.print_help()

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.pre_train_model_path = os.path.join(
        f"./ckpts/{cfg.dataset.name}", f"{cfg.model.pretrain_type}-{cfg.model.backbone}-h{cfg.model.hidden_dim}.pth")

    if cfg.model.backbone == 'GAT':
        force_deter = False
    else:
        force_deter = True    
    set_seed_global(cfg.seed, force_deter=force_deter)

    if cfg.task == 'node':
        tasker = NodeTask(
            pre_train_model_path=cfg.pre_train_model_path,
            dataset_name=cfg.dataset.name,
            gnn_type=cfg.model.backbone,
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            prompt_type=cfg.model.prompt_type,
            num_shot=cfg.num_shot,
            epochs=cfg.optim.epochs,
            lr=cfg.optim.lr,
            wd=cfg.optim.wd,
            device=cfg.device,
            r=cfg.model.r
        )
        tasker.run()
    elif cfg.task == 'graph':
        tasker = GraphTask(
            pre_train_model_path=cfg.pre_train_model_path,
            dataset_name=cfg.dataset.name,
            gnn_type=cfg.model.backbone,
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            prompt_type=cfg.model.prompt_type,
            num_shot=cfg.num_shot,
            epochs=cfg.optim.epochs,
            lr=cfg.optim.lr,
            wd=cfg.optim.wd,
            device=cfg.device,
            r=cfg.model.r
        )
        tasker.run()
