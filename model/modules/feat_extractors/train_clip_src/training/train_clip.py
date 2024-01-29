'''
v-iashin: adapted from `open_clip/src/training/main.py`
'''
import glob
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import random

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from model.modules.feat_extractors.train_clip_src.open_clip.factory import create_model
from scripts.train_utils import EarlyStopper, get_curr_time_w_random_shift, get_transforms

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None


from open_clip import trace_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate_on_sync_w_shifts
from training.file_utils import pt_load, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    '''If the `model` object is wrapped in `torch.nn.parallel.DistributedDataParallel` we have
    to use `model.modules` to get access to methods of the model. This wrapper allows
    to avoid using `if ddp: model.module.* else: model.*`. Used during `evaluate_on_sync_w_shifts`.'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main(cfg):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(cfg)

    # get the name of the experiments
    date_str = cfg.get('start_time', get_curr_time_w_random_shift())
    if cfg.distributed:
        # sync date_str from master to all ranks
        date_str = broadcast_object(cfg, date_str)
    cfg.name = date_str

    resume_latest = cfg.training.resume == 'latest'
    log_base_path = os.path.join(cfg.logging.logdir, cfg.name)
    cfg.log_path = None
    if is_master(cfg, local=cfg.logging.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{cfg.rank}' if cfg.logging.log_local else 'out.log'
        cfg.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(cfg.log_path) and not resume_latest and cfg.train_data is not None:
        #     print(f"Warning. Experiment already exists. Resuming from {cfg.log_path}, perhaps not latest")
        #     return -1

    # Setup text logger
    cfg.log_level = logging.DEBUG if cfg.debug else logging.INFO
    setup_logging(cfg.log_path, cfg.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    cfg.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(cfg):
        cfg.tensorboard_path = log_base_path
        for dirname in [cfg.tensorboard_path, cfg.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        cfg.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        # checkpoint_path = cfg.training.resume
        checkpoint_path = os.path.join(cfg.logging.logdir, cfg.name, "checkpoints")
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if cfg.training.remote_sync is not None:
            checkpoint_path = os.path.join(cfg.training.remote_sync, cfg.name, "checkpoints")
            if cfg.logging.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if cfg.training.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(cfg):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if cfg.logging.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=cfg.training.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if cfg.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(cfg, resume_from)
        cfg.training.resume = resume_from

    if cfg.logging.log_code_state and is_master(cfg):
        copy_codebase(cfg)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(cfg) and cfg.training.remote_sync is not None:
        # FIXME: I did one reading, but only a run should raise errors if any
        logging.warning('v-iashin: have not debugged this bit')
        logging.warning('v-iashin: have not debugged this bit')
        logging.warning('v-iashin: have not debugged this bit')
        logging.warning('v-iashin: have not debugged this bit')
        logging.warning('v-iashin: have not debugged this bit')
        # first make sure it works
        result = remote_sync(
            os.path.join(cfg.logging.logdir, cfg.name),
            os.path.join(cfg.training.remote_sync, cfg.name),
            cfg.training.remote_sync_protocol
        )
        if result:
            logging.info('remote_sync successful.')
        else:
            logging.error('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            cfg.training.remote_sync_frequency,
            os.path.join(cfg.logging.logdir, cfg.name),
            os.path.join(cfg.training.remote_sync, cfg.name),
            cfg.training.remote_sync_protocol
        )
        remote_sync_process.start()

    if cfg.training.precision == 'fp16':
        logging.warning('It is recommended to use AMP mixed-precision instead of FP16. '
                        'FP16 support needs further verification and tuning, especially for train.')

    if cfg.distributed:
        logging.info(f'Running in distributed mode with multiple processes. Device: {cfg.device}.'
                     f'Process (global: {cfg.rank}, local {cfg.local_rank}), total {cfg.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {cfg.device}.')

    dist_model = None
    cfg.distill = cfg.training.distill_model is not None and cfg.training.distill_pretrained is not None
    if cfg.distill:
        # FIXME: support distillation with grad accum.
        assert cfg.accum_freq == 1
        # FIXME: support distillation with coca.; FIXME: cfg.model is expected to be different
        assert 'coca' not in cfg.model.lower()

    if isinstance(cfg.training.force_image_size, (tuple, list)) and len(cfg.training.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        cfg.training.force_image_size = cfg.training.force_image_size[0]

    random_seed(cfg.training.seed, 0)

    model = create_model(cfg, device)
    transforms = get_transforms(cfg)
    if cfg.distill:
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model = create_model(cfg, device)

    random_seed(cfg.training.seed, cfg.rank)

    if cfg.training.trace:
        model = trace_model(model, batch_size=cfg.base.batch_size, device=device)

    if cfg.training.lock_rgb:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_rgb_tower(unlocked_groups=cfg.training.lock_rgb_unlocked_groups,
                             freeze_bn_stats=cfg.training.lock_rgb_freeze_bn_stats)
    if cfg.training.lock_audio:
        model.lock_audio_tower(unlocked_layers=cfg.training.lock_audio_unlocked_layers,
                               freeze_layer_norm=cfg.training.lock_audio_freeze_layer_norm)

    if cfg.training.grad_checkpointing:
        raise NotImplementedError('grad checkpointing is not supported yet')
        model.set_grad_checkpointing()

    if is_master(cfg):
        # if resuming, making a copy of the resumed config; if a new experiment, using the exp name
        exp_name = get_curr_time_w_random_shift() if cfg.training.resume else cfg.name
        cfg_fname = f'cfg-{exp_name}.yaml'
        cfg_path = os.path.join(cfg.logging.logdir, cfg.name, cfg_fname)
        OmegaConf.save(cfg, cfg_path)
        logging.info(OmegaConf.to_yaml(cfg))

    if cfg.distributed:
        if cfg.training.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if cfg.training.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if cfg.distill:
            dist_model = DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if cfg.data.vids_path or cfg.data.dataset_type == "synthetic":
        assert not cfg.training.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_param = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_param = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW([{"params": gain_or_bias_param, "weight_decay": 0.},
                                 {"params": rest_param, "weight_decay": cfg.training.optimizer.weight_decay}],
                                lr=cfg.training.learning_rate,
                                betas=cfg.training.optimizer.betas,
                                eps=1e-08,)

        scaler = GradScaler() if cfg.training.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if cfg.training.resume is not None:
        checkpoint = pt_load(cfg.training.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not cfg.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{cfg.training.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{cfg.training.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(cfg, transforms, epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * cfg.training.num_epochs
        # scale the num of warmup updates by world size
        warmup = int(cfg.training.lr_scheduler.get('warmup', 0) // cfg.world_size)
        if cfg.training.lr_scheduler.name == "cosine":
            scheduler = cosine_lr(optimizer, cfg.training.learning_rate, warmup, total_steps)
        elif cfg.training.lr_scheduler.name == "const":
            scheduler = const_lr(optimizer, cfg.training.learning_rate, warmup, total_steps)
        elif cfg.training.lr_scheduler.name == "const-cooldown":
            assert cfg.training.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = data["train"].dataloader.num_batches * cfg.training.epochs_cooldown
            scheduler = const_lr_cooldown(optimizer, cfg.lr, warmup, total_steps,
                                          cooldown_steps, cfg.training.lr_cooldown_power,
                                          cfg.training.lr_cooldown_end)
        else:
            logging.error(f'Unknown scheduler, {cfg.training.lr_scheduler.name}.' \
                          + 'Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    cfg.save_logs = cfg.logging.logdir and cfg.logging.logdir.lower() != 'none' and is_master(cfg)
    writer = None
    if cfg.save_logs and cfg.logging.use_tboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(cfg.tensorboard_path)

    if cfg.logging.use_wandb and is_master(cfg):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        cfg.train_sz = data["train"].dataloader.num_samples
        if cfg.data.vids_path is not None:
            cfg.val_sz = data['valid'].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=f'avsync-{cfg.action}',
            name=cfg.name,
            notes=cfg.logging.wandb_notes,
            tags=[],
            dir=cfg.logging.logdir,
            config=OmegaConf.to_container(cfg, resolve=[True | False]),
        )
        if cfg.debug:
            wandb.watch(model, log='all')
        wandb.save(cfg_path)
        logging.debug('Finished loading wandb.')

    if 'train' not in data and cfg.training.run_shifted_win_val:
        # change this with coneectin to other mentions of evaluate
        evaluate_on_sync_w_shifts(model, data, 'valid', start_epoch, cfg, writer)
        return

    loss = None

    ### This does not work with ddp :( in our case TODO: check with upcoming newer versions of pytorch
    # if cfg.training.compile:
    #     logging.info('Started model compilation...')
    #     model = torch.compile(model)
    #     logging.info('Model has been compiled')
    early_stopper = EarlyStopper(cfg.training.patience, cfg.training.to_max_metric, cfg.training.metric_name)
    if cfg.training.resume is not None and 'metrics' in checkpoint:
        # FIXME: this will use sync_w_shifts performance to decide if the model has improved
        best_metrics = checkpoint['metrics']['sync_w_shifts']
        early_stopper.set_best_metrics(best_metrics)

    # print the number of parameters  (NOTE: this includes the projection layers)
    if is_master(cfg):
        logging.info(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        logging.info(f'Segment queue size: {getattr(model, "segment_queue_size", None)}')
        logging.info(f'Global queue size: {getattr(model, "global_queue_size", None)}')

    for epoch in range(start_epoch, cfg.training.num_epochs):
        if is_master(cfg):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, cfg, writer)
        completed_epoch = epoch + 1

        if (
            cfg.training.run_shifted_win_val \
                and cfg.training.val_frequency > 0 \
                and ((epoch % cfg.training.val_frequency) == 0 or epoch == cfg.epochs)
        ):
            sync_w_shifts_metrics = evaluate_on_sync_w_shifts(model, data, 'valid', completed_epoch,
                                                              cfg, writer, loss)

        # Saving checkpoints.
        if is_master(cfg) and cfg.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": cfg.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'metrics': {
                    'sync_w_shifts': sync_w_shifts_metrics,
                },
                'args': cfg,
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == cfg.training.num_epochs or (
                cfg.logging.save_frequency > 0 and (completed_epoch % cfg.logging.save_frequency) == 0
            ):
                save_path = os.path.join(cfg.checkpoint_path, f"epoch_{completed_epoch}.pt")
                torch.save(checkpoint_dict, save_path)
                logging.info(f"=> saved checkpoint at epoch {completed_epoch} to {save_path}")
            if cfg.logging.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(cfg.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
                    logging.info(f"=> deleted checkpoint from {previous_checkpoint}")
            # save best
            if early_stopper.is_new_model_better_than_curr(sync_w_shifts_metrics):
                early_stopper.reset_patience(cfg.rank, sync_w_shifts_metrics)
                # try not to corrupt the latest checkpoint if save fails
                tmp2_save_path = os.path.join(cfg.checkpoint_path, "tmp2.pt")
                best_save_path = os.path.join(cfg.checkpoint_path, 'epoch_best.pt')
                torch.save(checkpoint_dict, tmp2_save_path)
                logging.info(f"=> saved best checkpoint at epoch {completed_epoch} to {tmp2_save_path}")
                os.replace(tmp2_save_path, best_save_path)
                logging.info(f"=> moved best checkpoint from {tmp2_save_path} to {best_save_path}")
            else:
                early_stopper.increment_patience(cfg.rank)

            if cfg.logging.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(cfg.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(cfg.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                logging.info(f"=> saved latest checkpoint at epoch {completed_epoch} to {tmp_save_path}")
                os.replace(tmp_save_path, latest_save_path)
                logging.info(f"=> moved latest checkpoint from {tmp_save_path} to {latest_save_path}")

    if cfg.logging.use_wandb and is_master(cfg):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(cfg.logging.logdir, cfg.name),
            os.path.join(cfg.training.remote_sync, cfg.name),
            cfg.training.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logging.logdir, args.name, f"code-{args.name}")
    if os.path.exists(new_code_path):
        logging.warning(f'Code for the experiment already exists at {new_code_path}. Skipping copy.')
        return -1
    logging.info(f"Copying codebase to {new_code_path}")
    # current_code_path = os.path.realpath(__file__)
    # current_code_path = os.path.dirname(current_code_path)
    current_code_path = os.getcwd()
    patterns_to_ignore = args.logging.patterns_to_ignore
    copytree(current_code_path, new_code_path, ignore=ignore_patterns(*patterns_to_ignore))
    logging.info("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
