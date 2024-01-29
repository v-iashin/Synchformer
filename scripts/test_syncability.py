import numpy as np
from datetime import timedelta
import os
import math
import time
import shutil
import logging
from pathlib import Path
import pickle

import sys
sys.path.insert(0, '.')  # nopep8

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from model.modules.feat_extractors.train_clip_src.training.train import AverageMeter

from scripts.train_utils import (broadcast_obj, calc_cls_metrics, gather_dict, get_batch_sizes,
                                 get_curr_time_w_random_shift, get_datasets,
                                 get_device, get_loaders, get_model, get_transforms, is_master,
                                 prepare_inputs, set_seed)
from utils.utils import cfg_sanity_check_and_patch
from sklearn.metrics import roc_curve, roc_auc_score


def setup_logging(cfg, log_dir, to_resume, save_in_file=True):
    logging_level = logging.INFO
    # seting up logging
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H-%M-%S')
    logging.root.setLevel(logging_level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if logger.name.startswith(('transformers', )):  # these guys are too verbose at INFO
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging_level)
    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if len(logging.root.handlers) > 0:
        logging.root.handlers.clear()
    logging.root.addHandler(stream_handler)

    # log file
    if save_in_file:
        log_file = os.path.join(log_dir, f'log-{cfg.start_time}.log')
        if log_file:
            # if resumed, appends to the same log file (`a`)
            file_handler = logging.FileHandler(log_file, mode='a' if to_resume else 'w')
            file_handler.setFormatter(formatter)
            logging.root.addHandler(file_handler)


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


def init_ddp(cfg_sync, cfg_off=None):
    local_rank = os.environ.get('LOCAL_RANK')

    if local_rank is not None:
        print('NUM_GPUS', os.environ.get("WORLD_SIZE"),
              '- LOCAL RANK', os.environ.get('LOCAL_RANK'),
              '- GLOBAL RANK', os.environ.get('RANK'))
        print('HOST:', os.environ.get('HOSTNAME'),
              'MASTER:', os.environ.get('MASTER_ADDR'), ':', os.environ.get('MASTER_PORT'))
        dist.init_process_group(cfg_sync.training.dist_backend, timeout=timedelta(0, 600))
        cfg_sync.training.local_rank = int(os.environ['LOCAL_RANK'])
        cfg_sync.training.global_rank = int(os.environ['RANK'])
        cfg_sync.training.world_size = dist.get_world_size()
        if cfg_off is not None:
            cfg_off.training.local_rank = int(os.environ['LOCAL_RANK'])
            cfg_off.training.global_rank = int(os.environ['RANK'])
            cfg_off.training.world_size = dist.get_world_size()
    else:
        cfg_sync.training.local_rank = 0
        cfg_sync.training.global_rank = 0
        cfg_sync.training.world_size = 1
        if cfg_off is not None:
            cfg_off.training.local_rank = 0
            cfg_off.training.global_rank = 0
            cfg_off.training.world_size = 1


def patch_cfg(cfg):
    cfg.logging.logdir = cfg.logging.logdir.replace('//', '/')\
                            .replace('/scratch/project_2000936/vladimir/logs/sync/', './logs/')\
                            .replace('/scratch/project_462000293/vladimir/logs/sync/', './logs/')\
                            .replace('/flash/project_2000936/vladimir/logs/sync/', './logs/')
    cfg.data.vids_path = cfg.data.vids_path.replace('//', '/')\
                            .replace('/scratch/project_2000936/vladimir/', '/home/nvme/data/')\
                            .replace('/scratch/project_462000293/vladimir/', '/home/nvme/data/')\
                            .replace('/flash/project_2000936/vladimir/', '/home/nvme/data/')
    if 'ckpt_path' in cfg:
        cfg.ckpt_path = cfg.ckpt_path.replace('//', '/')\
                           .replace('/scratch/project_2000936/vladimir/logs/sync/', './logs/')\
                           .replace('/scratch/project_462000293/vladimir/logs/sync/', './logs/')\
                           .replace('/flash/project_2000936/vladimir/logs/sync/', './logs/')
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    return cfg


def get_set_of_video_ids_shorter_than_9_6_sec():
    return {
        '-7tYmeOmsRg_180000_190000.mp4',
        '1_Q80fDGLRM_10000_20000.mp4',
        '8qsCZLEoA1Q_4000_14000.mp4',
        'F9bJVVYgFl4_73000_83000.mp4',
        'KQAR_64a35I_11000_21000.mp4',
        'TgJHM5oSWio_8000_18000.mp4',
        'U9PyY8Ldf9A_5000_15000.mp4',
        'aUfDxRelPHg_22000_32000.mp4',
        'cLpDBj--as0_8000_18000.mp4',
        'cRT5SWbyA54_4000_14000.mp4'
    }


def test_syncability():
    target_key_sync = 'sync_target'
    target_key_off = 'offset_target'
    phase = 'test'

    # loading configs
    cfg_cli = OmegaConf.from_cli()
    cfg_sync_yml = OmegaConf.load(cfg_cli.config_sync)

    # deciding if we need to load the offset pred model,
    # i.e. if the user wants to evaluate it across various thresholds)
    config_off_path = cfg_cli.get('config_off', None)
    if config_off_path is None:
        assert cfg_cli.get('ckpt_path_off', None) is None, '`config_off` is not specified, but `ckpt_path_off` is'
        do_tier_offset_preds_by_sync = False
    else:
        assert cfg_cli.get('ckpt_path_off', None) is not None, '`config_off` is specified, but `ckpt_path_off` is not'
        do_tier_offset_preds_by_sync = True

    if do_tier_offset_preds_by_sync:
        cfg_off_yml = OmegaConf.load(cfg_cli.config_off)

    # the latter arguments are prioritized
    cfg_sync = OmegaConf.merge(cfg_sync_yml, cfg_cli)
    cfg_sync.ckpt_path = cfg_cli.ckpt_path_sync
    # TODO: replace this with making s1 ckpt paths to be None
    if 'SLURM_JOB_ID' not in os.environ:
        cfg_sync = patch_cfg(cfg_sync)
    if 'start_time' not in cfg_sync or cfg_sync.start_time is None:
        cfg_sync.start_time = get_curr_time_w_random_shift()
    if do_tier_offset_preds_by_sync:
        cfg_off = OmegaConf.merge(cfg_off_yml, cfg_cli)
        cfg_off.ckpt_path = cfg_cli.ckpt_path_off
        if 'SLURM_JOB_ID' not in os.environ:
            cfg_off = patch_cfg(cfg_off)
        if 'start_time' not in cfg_off or cfg_off.start_time is None:
            cfg_off.start_time = cfg_sync.start_time

    # since offset pred model predict on 14 n_segments and sync on 13, we need to make sure that
    # the synchability transforms will make inputs 14-segments long
    # the following patches all occurencies (hard coded for now)
    n_segments_sync = cfg_sync_yml.data.n_segments
    if do_tier_offset_preds_by_sync:
        for i, t_cfg in enumerate(cfg_sync.transform_sequence_train):
            if t_cfg.target.endswith(('TemporalCropAndOffsetForSyncabilityTraining', 'GenerateMultipleSegments')):
                cfg_sync.transform_sequence_train[i].params.n_segments = cfg_off_yml.data.n_segments
        for i, t_cfg in enumerate(cfg_sync.transform_sequence_test):
            if t_cfg.target.endswith(('TemporalCropAndOffsetForSyncabilityTraining', 'GenerateMultipleSegments')):
                cfg_sync.transform_sequence_test[i].params.n_segments = cfg_off_yml.data.n_segments

    # adds support for special resolve function in config eg `param: ${add:0,True,2,3}` will be resolved to 6
    OmegaConf.register_new_resolver('add', lambda *args: sum(args))
    OmegaConf.resolve(cfg_sync)  # things like "${model.size}" in cfg_sync will be resolved into values
    cfg_sanity_check_and_patch(cfg_sync)

    if do_tier_offset_preds_by_sync:
        OmegaConf.resolve(cfg_off)
        cfg_sanity_check_and_patch(cfg_off)

    set_env_variables()

    init_ddp(cfg_sync=cfg_sync, cfg_off=cfg_off if do_tier_offset_preds_by_sync else None)
    global_rank = cfg_sync.training.global_rank
    world_size = cfg_sync.training.world_size
    # FIXME: only earlier models don't have this key
    cfg_sync.logging.log_frequency = cfg_sync.logging.get('log_frequency', 5)

    if is_master(global_rank):
        # self.logdir!
        logdir = os.path.join(cfg_sync.logging.logdir, cfg_sync.start_time)
        setup_logging(cfg_sync, logdir, to_resume=cfg_sync.training.resume)
        logging.info(f'Config (Sync): \n{OmegaConf.to_yaml(cfg_sync)}')
        if do_tier_offset_preds_by_sync:
            logging.info(f'Config (Off): \n{OmegaConf.to_yaml(cfg_off)}')

    device, num_gpus = get_device(cfg_sync)

    # ckpt_path was created only for the master (to keep it the same), now we broadcast it to each worker
    cfg_sync.ckpt_path = broadcast_obj(cfg_sync.ckpt_path, global_rank, device)
    # making sure each worker has the same ckpt path as the master
    assert hasattr(cfg_sync, 'ckpt_path'), f'I AM AT RANK: {global_rank}'
    if do_tier_offset_preds_by_sync:
        cfg_off.ckpt_path = broadcast_obj(cfg_off.ckpt_path, global_rank, device)
        assert hasattr(cfg_off, 'ckpt_path'), f'I AM AT RANK: {global_rank}'

    set_seed(cfg_sync.training.seed)  # same seed for all workers for model init
    transforms = get_transforms(cfg_sync)  # getting away with only sync transforms for both
    model_sync, model_sync_without_ddp = get_model(cfg_sync, device)
    if do_tier_offset_preds_by_sync:
        model_off, model_off_without_ddp = get_model(cfg_off, device)
    set_seed(cfg_sync.training.seed + global_rank)
    batch_sizes = get_batch_sizes(cfg_sync, num_gpus)
    datasets = get_datasets(cfg_sync, transforms)

    ignore_vids = get_set_of_video_ids_shorter_than_9_6_sec()
    # filter dataset['test'] for these vids
    datasets[phase].dataset = [p for p in datasets[phase].dataset if Path(p).name not in ignore_vids]
    logging.info(f'Filtered {len(ignore_vids)} videos from the {phase} set')
    loaders = get_loaders(cfg_sync, datasets, batch_sizes)

    ckpt_sync = torch.load(cfg_sync.ckpt_path, map_location=torch.device('cpu'))
    model_sync_without_ddp.load_state_dict(ckpt_sync['model'])
    ckpt_sync_epoch = ckpt_sync['epoch']
    best_metric_val_sync = ckpt_sync['metrics'][cfg_sync.training.metric_name]
    model_sync.eval()
    if do_tier_offset_preds_by_sync:
        ckpt_off = torch.load(cfg_off.ckpt_path, map_location=torch.device('cpu'))
        model_off_without_ddp.load_state_dict(ckpt_off['model'])
        ckpt_off_epoch = ckpt_off['epoch']
        best_metric_val_off = ckpt_off['metrics'][cfg_off.training.metric_name]
        model_off.eval()

    if is_master(global_rank):
        logging.info(f'Loading the best syncability model from {cfg_sync.ckpt_path}')
        logging.info(f'Best metric (syncablity): {best_metric_val_sync}')
        logging.info((f'The syncability model was trained for {ckpt_sync_epoch} epochs.'))
        logging.info(f'Copying the checkpoint to *_eXX.pt from {cfg_sync.ckpt_path}')
        if do_tier_offset_preds_by_sync:
            logging.info(f'Loading the best offset pred model from {cfg_off.ckpt_path}')
            logging.info(f'Best metric (offset pred): {best_metric_val_off}')
            logging.info((f'The offset pred model was trained for {ckpt_off_epoch} epochs.'))
            logging.info(f'Copying the checkpoint to *_eXX.pt from {cfg_off.ckpt_path}')
        p = Path(cfg_sync.ckpt_path)
        curr_t = get_curr_time_w_random_shift()
        shutil.copyfile(str(p), str(
            (p.parent / f'{p.stem}_e{ckpt_sync_epoch}_t{curr_t}').with_suffix(p.suffix)))

    # init runnining results
    running_results = dict(logits_sync=[], targets_sync=[], )
    if do_tier_offset_preds_by_sync:
        running_results['logits_off'] = []
        running_results['targets_off'] = []

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(ckpt_sync_epoch)

    # how many times to iterate through a evaluation dataset (makes estimates more robust for small datasets)
    iter_times = cfg_sync.data.dataset.params.get('iter_times', 1)
    for it in range(iter_times):

        # resetting batch / data time meters per log window
        batch_time_m.reset()
        data_time_m.reset()

        num_samples = 0
        for i, batch in enumerate(loaders[phase]):
            aud, vid, targets = prepare_inputs(batch, device, phase)
            data_time_m.update(time.time() - end)
            with torch.set_grad_enabled(False):
                with torch.autocast('cuda', enabled=cfg_sync.training.use_half_precision):
                    loss_sync, logits_sync = model_sync(vid[:, :n_segments_sync].clone(),
                                                        aud[:, :n_segments_sync].clone(),
                                                        targets[target_key_sync])
                    if do_tier_offset_preds_by_sync:
                        loss_off, logits_off = model_off(vid, aud, targets[target_key_off])

            num_samples += len(vid) * cfg_sync.training.world_size
            batch_time_m.update(time.time() - end)
            end = time.time()

            # gathering results in one place to iterate on this later
            iter_results = dict(
                logits_sync=[logits_sync.detach().cpu()],
                targets_sync=[targets[target_key_sync].cpu()],
            )
            if do_tier_offset_preds_by_sync:
                iter_results['logits_off'] = [logits_off.detach().cpu()]
                iter_results['targets_off'] = [targets[target_key_off].cpu()]

            for k in running_results.keys():
                running_results[k] += iter_results[k]

            if is_master(global_rank) and i % cfg_sync.logging.log_frequency == 0:
                batch_size = len(vid)
                samples_per_epoch = len(loaders[phase].dataset)
                ratio_done = (i+1) / len(loaders[phase])
                sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
                msg = f'{phase} ({ckpt_sync_epoch}) [{num_samples:>{sample_digits}}/{samples_per_epoch} ({ratio_done:.0%})]'
                if data_time_m is not None:
                    msg += f' Data (t): {data_time_m.avg:.3f}'
                if batch_time_m is not None:
                    samples_per_s = batch_size * world_size / batch_time_m.val
                    samples_per_s_per_gpu = batch_size / batch_time_m.val
                    msg += f' Batch (t): {batch_time_m.avg:.3f}, {samples_per_s:.1f}/s, {samples_per_s_per_gpu:.1f}/s/gpu'
                logging.info(msg)

        if is_master(global_rank):
            logging.info(f'Done iterations {it+1} / {iter_times}')

        if dist.is_initialized():
            dist.barrier()

        if is_master(global_rank):
            logging.info(f'Passed Barrier {it+1} / {iter_times}')

    running_results = gather_dict(running_results)

    if is_master(global_rank):
        logits_sync = torch.cat(running_results['logits_sync']).float()
        targets_sync = torch.cat(running_results['targets_sync']).long()
        if do_tier_offset_preds_by_sync:
            logits_off = torch.cat(running_results['logits_off']).float()
            targets_off = torch.cat(running_results['targets_off']).long()

        # roc curve score
        dataset_size, num_cls = logits_sync.shape
        targets_1hot = torch.nn.functional.one_hot(targets_sync, num_classes=num_cls)
        roc_aucs = [roc_auc_score(targets_1hot[:, c], torch.softmax(logits_sync, dim=1)[:, c], average=None) for c in range(num_cls)]
        roc_curve_sc = np.mean(roc_aucs)

        # roc curve
        fpr, tpr, thresholds = roc_curve(targets_sync, torch.softmax(logits_sync, dim=1)[:, 1], pos_label=1)
        save_path = os.path.join(logdir, f'roc_{phase}_e{ckpt_sync_epoch}_{get_curr_time_w_random_shift()}.pkl')
        roc_curve_vals = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_curve_sc': roc_curve_sc}
        with open(save_path, 'wb') as f:
            pickle.dump(roc_curve_vals, f)
        logging.info(f'Saved metrics to {save_path}')

        metrics_sync = calc_cls_metrics(targets_sync, logits_sync,
                                        add_doubt_cls=cfg_sync.training.get('add_doubt_cls', False))
        metrics_sync = {k: round(v, 4) for k, v in metrics_sync.items()}
        logging.info(f'Synchability metrics on {phase} set: {metrics_sync}')

        if do_tier_offset_preds_by_sync:
            # compute metrics for the each confidence threshold
            # 0.8 means that if the model is less than 80% confident in its prediction, it will be discarded
            conf_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            thresh2metrics = {t: None for t in conf_thresholds}
            for t in conf_thresholds:
                syncable_mask = logits_sync.softmax(dim=-1)[:, 1] > t  # those the sync model preds are syncable
                logits_sync_masked = logits_sync[syncable_mask]
                targets_sync_masked = targets_sync[syncable_mask]
                targets_off_masked = targets_off[syncable_mask]
                logits_off_masked = logits_off[syncable_mask]
                # if the synchability model is confident that the video is syncable, but it is not,
                # the offset prediction cannot be right. However, it is not always the case. Thus,
                # we need to make sure that it is not counted as a true positive (0 and 20 classes).
                # We need to swap the target on these.

                # those that the sync model thinks are syncable, but they are not
                incorrect_sync_mask = (logits_sync_masked.argmax(dim=-1) != targets_sync_masked)
                # the swapping values: (pred_off_cls + 5) % num_cls; (5 is to make sure tolerane wont catch it)
                fake_targets_off_masked = (logits_off_masked.argmax(dim=-1) + 5) % logits_off_masked.shape[-1]
                # swap the targets with the fake ones (the ones that the model is not predicting)
                targets_off_masked[incorrect_sync_mask] = fake_targets_off_masked[incorrect_sync_mask]

                metrics_off_t = calc_cls_metrics(targets_off_masked, logits_off_masked,
                                                 add_doubt_cls=cfg_off.training.get('add_doubt_cls', False))
                thresh2metrics[t] = {k: round(v, 4) for k, v in metrics_off_t.items()}
                logging.info(f'Metrics on offset prediction ({phase}) with conf thresh {t}: {thresh2metrics[t]}')

            # saving the metrics
            save_path = os.path.join(logdir, f'metrics_{phase}_e{ckpt_sync_epoch}_{get_curr_time_w_random_shift()}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(thresh2metrics, f)
            logging.info(f'Saved metrics to {save_path}')

    logging.info('Finished the experiment')


if __name__ == '__main__':
    test_syncability()
