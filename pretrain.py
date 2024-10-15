import sys
import argparse
from config import cfg
from utils.seed import set_seed_global
from pretrain_strategy.edgepred_dagprompt import EdgepredDAGPromptPretrain


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
    set_seed_global(cfg.seed)
    pretrain_mapper = {
        'dagprompt': EdgepredDAGPromptPretrain,
    }

    pretrain = pretrain_mapper[cfg.model.pretrain_type](
        gnn_type=cfg.model.backbone,
        dataset_name=cfg.dataset.name,
        hidden_dim=cfg.model.hidden_dim,
        num_layer=cfg.model.num_layers,
        epochs=cfg.optim.epochs,
        lr=cfg.optim.lr,
        wd=cfg.optim.wd,
        device=cfg.device
    )
    pretrain.pretrain()
