import os

from omegaconf import OmegaConf

from scripts.train_utils import get_curr_time_w_random_shift
from utils.utils import cfg_sanity_check_and_patch

def set_env_variables():
    # checks if not run with torchrun or torch.launch.distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return
    # otherwise checks if on slurm cluster
    elif 'SLURM_JOB_ID' in os.environ:
        # run sbatch with `--ntasks-per-node=GPUs`; MASTER_ADDR is expected to be `export`ed in sbatch file
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NPROCS']

def get_config():
    cfg_cli = OmegaConf.from_cli()
    cfg_yml = OmegaConf.load(cfg_cli.config)
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    if 'start_time' not in cfg or cfg.start_time is None:
        cfg.start_time = get_curr_time_w_random_shift()
    # adds support for special resolve function in config eg `param: ${add:0,True,2,3}` will be resolved to 6
    OmegaConf.register_new_resolver('add', lambda *args: sum(args))
    OmegaConf.resolve(cfg)  # things like "${model.size}" in cfg will be resolved into values
    return cfg


def main(cfg):
    if cfg.action == 'train_avclip':
        from model.modules.feat_extractors.train_clip_src.training.train_clip import main as train
    elif cfg.action in ['train_avsync_model', 'ft_avsync_model_for_syncability']:
        from scripts.train_sync import train as train
    else:
        raise NotImplementedError('cfg.action', cfg.action)
    cfg_sanity_check_and_patch(cfg)
    train(cfg)


if __name__ == '__main__':
    cfg = get_config()
    set_env_variables()
    main(cfg)
