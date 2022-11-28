import os
import wandb
import hydra
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf

from source.utils import initialize_wandb, Experiment


@hydra.main(version_base='1.2.0', config_path='config', config_name='default')
def main(cfg: DictConfig):

    distributed = (torch.cuda.device_count() > 1)
    torch.manual_seed(0)
    np.random.seed(0)

    # set up wandb
    key = os.environ.get('WANDB_API_KEY')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _ = initialize_wandb(cfg.wandb.project, cfg.wandb.username, cfg.exp_name, dir=cfg.wandb.dir, config=config, key=key)
    wandb.define_metric('epoch', summary='max')
    wandb.define_metric('train/lr', step_metric='epoch')

    print(f'Starting {cfg.local_rank} of n gpus: {torch.cuda.device_count()}')

    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    exp = Experiment(cfg, distributed, torch.cuda.device_count())
    exp.run_experiment()


if __name__ == "__main__":

    main()
