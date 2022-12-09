import os
import hydra
import torch
import numpy as np

from omegaconf import DictConfig

from source.utils import initialize_wandb, hydra_argv_remapper, Experiment


@hydra.main(version_base='1.2.0', config_path='config', config_name='debug')
def main(cfg: DictConfig):

    distributed = (torch.cuda.device_count() > 1)
    torch.manual_seed(0)
    np.random.seed(0)

    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print(f'Starting {cfg.local_rank} of n gpus: {torch.cuda.device_count()}')

    # set up wandb
    if cfg.local_rank == 0:
        key = os.environ.get('WANDB_API_KEY')
        run = initialize_wandb(cfg, key=key)
        run.define_metric('epoch', summary='max')
        run.define_metric('train/lr', step_metric='epoch')

    exp = Experiment(cfg, distributed, torch.cuda.device_count())
    exp.run_experiment()


if __name__ == "__main__":

    if torch.cuda.device_count() > 1:
        m = {}
        for i in range(torch.cuda.device_count()):
            m_i = {f'--local_rank={i}': 'local_rank'}
            m.update(m_i)
        hydra_argv_remapper(m)

    main()
