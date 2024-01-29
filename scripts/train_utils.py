import itertools
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import scipy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
try:
    import wandb
except ImportError:
    wandb = None

from matplotlib import pyplot as plt
from sklearn.metrics import (top_k_accuracy_score, average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler

from utils.utils import (fix_prefix, get_obj_from_str, get_transform_instance_from_compose,
                         instantiate_from_config, show_cfg_diffs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_ddp(cfg):
    local_rank = os.environ.get('LOCAL_RANK')

    if local_rank is not None:
        print('NUM_GPUS', os.environ.get("WORLD_SIZE"),
              '- LOCAL RANK', os.environ.get('LOCAL_RANK'),
              '- GLOBAL RANK', os.environ.get('RANK'))
        print('HOST:', os.environ.get('HOSTNAME'),
              'MASTER:', os.environ.get('MASTER_ADDR'), ':', os.environ.get('MASTER_PORT'))
        dist.init_process_group(cfg.training.dist_backend, timeout=timedelta(0, 600))
        cfg.training.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.training.global_rank = int(os.environ['RANK'])
        cfg.training.world_size = dist.get_world_size()
        # disable_print_if_not_master(is_master(dist.get_rank()))
    else:
        cfg.training.local_rank = 0
        cfg.training.global_rank = 0
        cfg.training.world_size = 1


def is_master(global_rank):
    return global_rank == 0


def get_curr_time_w_random_shift():
    # shifting for a random number of seconds so that exp folder names coincide less often
    now = datetime.now() - timedelta(seconds=np.random.randint(60))
    return now.strftime('%y-%m-%dT%H-%M-%S')


def broadcast_obj(object, global_rank, device):
    if dist.is_initialized():
        objects = [object if is_master(global_rank) else None for _ in range(dist.get_world_size())]
        dist.broadcast_object_list(objects, src=0, device=device)
        object = objects[0]
    return object


def get_device(cfg):
    device = torch.device(cfg.training.local_rank)
    torch.cuda.set_device(device)
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    return device, num_gpus


def get_transforms(cfg, which_transforms=['train', 'test']):
    transforms = {}
    for mode in which_transforms:
        ts_cfg = cfg.get(f'transform_sequence_{mode}', None)
        ts = [lambda x: x] if ts_cfg is None else [instantiate_from_config(c) for c in ts_cfg]
        transforms[mode] = torchvision.transforms.Compose(ts)
    return transforms


def get_datasets(cfg, transforms, which_datasets=['train', 'valid', 'test']):
    if not isinstance(which_datasets, (list, tuple)):
        which_datasets = [which_datasets]
    DatasetClass = get_obj_from_str(cfg.data.dataset.target)
    load_fixed_offsets_on = cfg.data.dataset.params.load_fixed_offsets_on
    vis_load_backend = cfg.data.dataset.params.vis_load_backend
    if 'size_ratios' in cfg.data.dataset.params:
        size_ratios = cfg.data.dataset.params.size_ratios
    else:
        size_ratios = {d: cfg.data.dataset.params.size_ratio for d in which_datasets}
    vids_path = cfg.data.vids_path
    attr_annot_path = cfg.data.dataset.params.get('attr_annot_path', None)
    max_attr_per_vid = cfg.data.dataset.params.get('max_attr_per_vid', None)

    datasets = dict()
    if 'train' in which_datasets:
        datasets['train'] = DatasetClass(
            split='train', vids_dir=vids_path, transforms=transforms['train'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['train'], attr_annot_path=attr_annot_path,
            max_attr_per_vid=max_attr_per_vid)
        logging.info(f'Loaded {len(datasets["train"])} train samples')
    if 'valid' in which_datasets:
        datasets['valid'] = DatasetClass(
            split='valid', vids_dir=vids_path, transforms=transforms['test'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['valid'], attr_annot_path=attr_annot_path,
            max_attr_per_vid=max_attr_per_vid)
        logging.info(f'Loaded {len(datasets["valid"])} valid samples')
    if 'test' in which_datasets:
        datasets['test'] = DatasetClass(
            split='test', vids_dir=vids_path, transforms=transforms['test'],
            load_fixed_offsets_on=load_fixed_offsets_on, vis_load_backend=vis_load_backend,
            size_ratio=size_ratios['test'], attr_annot_path=attr_annot_path,
            max_attr_per_vid=max_attr_per_vid)
        logging.info(f'Loaded {len(datasets["test"])} test samples')
    return datasets


def get_datasets_v2(cfg, which_datasets=['train', 'valid', 'test']):
    if not isinstance(which_datasets, (list, tuple)):
        which_datasets = [which_datasets]
    datasets = {}
    for phase in which_datasets:
        # for each phase, we instantiate transforms and dataset separately to avoid instantiating them
        # inside the dataset class (which is not always desirable)
        ts_cfgs = cfg.transform_sequence_train if phase == 'train' else cfg.transform_sequence_test
        ts = [lambda x: x] if ts_cfgs is None else [instantiate_from_config(c) for c in ts_cfgs]
        transforms = torchvision.transforms.Compose(ts)
        obj_name = cfg.data.dataset[phase].name
        obj_args = cfg.data.dataset[phase].args
        datasets[phase] = get_obj_from_str(obj_name)(transforms=transforms, **obj_args)
        logging.info(f'Loaded {len(datasets[phase])} {phase} samples')
    return datasets


def get_batch_sizes(cfg, num_gpus=None):
    return {'train': cfg.training.base_batch_size, 'test': cfg.training.base_batch_size, }


def get_loaders(cfg, datasets, batch_sizes):
    loaders = dict()
    for phase, dataset in datasets.items():
        if dist.is_initialized():
            sampler = DistributedSampler(datasets[phase], shuffle=phase == 'train')
        else:
            sampler = None

        if phase == 'train':
            # NOTE: don't change this to True, as it is used in `sampler`
            loaders[phase] = DataLoader(dataset, batch_sizes['train'], shuffle=sampler is None,
                                        sampler=sampler, num_workers=cfg.training.num_workers)
        else:
            loaders[phase] = DataLoader(dataset, batch_sizes['test'], shuffle=False,
                                        sampler=sampler, num_workers=cfg.training.num_workers)
    return loaders


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    '''If the `model` object is wrapped in `torch.nn.parallel.DistributedDataParallel` we have
    to use `model.modules` to get access to methods of the model. This wrapper allows
    to avoid using `if ddp: model.module.* else: model.*`. Used during `evaluate_on_sync_w_shifts`.'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def get_model(cfg, device):
    model = instantiate_from_config(cfg.model)

    # TODO: maybe in the module
    if cfg.model.params.vfeat_extractor.is_trainable is False:
        for params in model.vfeat_extractor.parameters():
            params.requires_grad = False
    if cfg.model.params.afeat_extractor.is_trainable is False:
        for params in model.afeat_extractor.parameters():
            params.requires_grad = False

    model = model.to(device)
    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[cfg.training.local_rank])
        # any mistaken calls on `model_without_ddp` (=None) will likely raise an error
        model_without_ddp = model.module

    return model, model_without_ddp


def get_optimizer(cfg, model, num_gpus):
    learning_rate = cfg.training.base_learning_rate * num_gpus
    # TODO: instantiate (but we need to pass params as well - fix the intantiate fn)
    # TODO: consider uncommenting this line and optimize only these tensors (from torchvision reference)
    # params = [p for p in model.parameters() if p.requires_grad]
    # avoiding NaN during half precision training
    eps = 1e-7 if cfg.training.use_half_precision else 1e-8
    if cfg.training.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, cfg.training.optimizer.betas,
                                     eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate, cfg.training.optimizer.betas,
                                      eps, cfg.training.optimizer.weight_decay)
    elif cfg.training.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, cfg.training.optimizer.momentum,
                                    weight_decay=cfg.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer: "{cfg.training.optimizer.name}" is not implemented')
    return optimizer


def get_lr_scheduler(cfg, optimizer):
    if cfg.training.lr_scheduler.name == 'constant_with_warmup':
        assert 'warmup' in cfg.training.lr_scheduler, f'{cfg.training.lr_scheduler}'
        warmup = cfg.training.lr_scheduler.warmup
        lr_sched = lr_scheduler.SequentialLR(optimizer, schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=warmup),
            lr_scheduler.ConstantLR(optimizer, factor=1),
        ], milestones=[warmup])
    elif cfg.training.lr_scheduler.name == 'constant':
        lr_sched = lr_scheduler.ConstantLR(optimizer, factor=1)
    return lr_sched


def load_ckpt(cfg, model_wo_ddp, optimizer=None, scaler=None, lr_scheduler=None):
    ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    ckpt_cfg = ckpt['args']
    if not cfg.training.finetune:
        model_wo_ddp.load_state_dict(ckpt['model'])
    else:
        _ckpt_load_status = model_wo_ddp.load_state_dict(ckpt['model'], strict=False)
        if len(_ckpt_load_status.missing_keys) > 0 or len(_ckpt_load_status.unexpected_keys) > 0:
            logging.warning(f'ckpt load status: {_ckpt_load_status}')
            logging.warning('Check if the above missing keys are expected. If so, ignore this warning.'
                            'Otherwise (e.g. lots of unmatched keys), fix the ckpt or the model.'
                            'Expectable keys could be heads during fine-tuning')

    if cfg.training.resume and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if cfg.training.resume and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if cfg.training.resume and 'lr_scheduler' in ckpt:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch = ckpt['epoch']
    # restarting training counters if the ckpt is used to init weights rather than continuing training
    if cfg.training.finetune:
        old_vid_dir = f'{Path(ckpt_cfg.data.vids_path).parent.stem}/{Path(ckpt_cfg.data.vids_path).stem}'
        new_vid_dir = f'{Path(cfg.data.vids_path).parent.stem}/{Path(cfg.data.vids_path).stem}'
        # assert old_vid_dir != new_vid_dir, f'ft on same data? old: {old_vid_dir}; new: {new_vid_dir}'
        # assert ckpt_cfg.data.dataset.target != cfg.data.dataset.target, ckpt_cfg.data.dataset.target
        if old_vid_dir == new_vid_dir:
            logging.warning(f'Finetuning on the same data: {old_vid_dir}')
        if ckpt_cfg.data.dataset.target == cfg.data.dataset.target:
            logging.warning(f'Finetuning on the same dataset: {ckpt_cfg.data.dataset.target}')
        logging.info(f'Finetuning from: {ckpt_cfg.ckpt_path} on {cfg.data.dataset.target}')
        ckpt['metrics'][cfg.training.metric_name] = 0 if cfg.training.to_max_metric else float('inf')
        start_epoch = 0
        show_cfg_diffs(ckpt_cfg, cfg, Path(cfg.ckpt_path).parent / 'cfg_diffs.diff')
    elif cfg.training.resume:
        start_epoch += 1
    # a bit ugly but it patches checkpoints produced by the 'old' code
    metrics = ckpt['metrics']
    metrics = metrics.get('off', metrics)
    return start_epoch, metrics


class EarlyStopper(object):

    def __init__(self, patience: int, to_max: bool, metric_name: str) -> None:
        '''E.g. If loss is the trackable metric, to_max=False; if accuracy, to_max=True'''
        self.no_change_epochs = 0
        self.to_max = to_max
        self.best_metric = 0.0 if to_max else float('inf')
        self.best_metrics = {metric_name: self.best_metric}
        self.metric_name = metric_name
        self.triggered = False
        self.patience = patience

    def reset_patience(self, global_rank, metrics):
        self.no_change_epochs = 0
        self.set_best_metrics(metrics)
        if is_master(global_rank):
            logging.info(f'New best {self.metric_name}: {self.best_metric:.5f}')

    def increment_patience(self, global_rank):
        self.no_change_epochs += 1
        if self.no_change_epochs >= self.patience:
            self.triggered = True
        if is_master(global_rank):
            logging.info(
                f'{self.metric_name} ({self.best_metric:.5f}) hasnt changed for {self.no_change_epochs} '
                f'patience: {self.patience}'
            )

    def is_new_model_better_than_curr(self, metrics: dict):
        new_metric_val = metrics[self.metric_name]
        return self.best_metric < new_metric_val if self.to_max else self.best_metric > new_metric_val

    def set_best_metrics(self, metrics):
        self.best_metric = metrics[self.metric_name]
        self.best_metrics = metrics


def toggle_mode(cfg, model, phase):
    if phase == 'train':
        model.train()
        if cfg.model.params.afeat_extractor.is_trainable is False:
            if dist.is_initialized():
                model.module.afeat_extractor.eval()
            else:
                model.afeat_extractor.eval()
        if cfg.model.params.vfeat_extractor.is_trainable is False:
            if dist.is_initialized():
                model.module.vfeat_extractor.eval()
            else:
                model.vfeat_extractor.eval()
    else:
        model.eval()


def apply_fn_recursive(obj, fn: callable):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, dict):
        return {k: apply_fn_recursive(v, fn) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [apply_fn_recursive(v, fn) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([apply_fn_recursive(v, fn) for v in obj])
    else:
        raise NotImplementedError(f'obj type: {type(obj)}')

def prepare_inputs(batch, device, phase=None, get_targets=True):
    targets = None
    if get_targets:
        targets = batch['targets']
        # targets = {k: targets[k].to(device, non_blocking=True) for k in targets if 'target' in k}
        for k, v in targets.items():
            if 'target' in k:
                targets[k] = apply_fn_recursive(v, lambda x: x.to(device))

    aud = batch['audio'].to(device)
    vid = batch['video'].to(device)

    return aud, vid, targets

def make_backward_and_optim_step(cfg, loss, model, optimizer, scaler, lr_scheduler):
    # without half precision training:
    # loss.backward()
    # if cfg.get('max_clip_norm', None) is not None:
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_clip_norm)
    # optimizer.step()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    max_clip_norm = cfg.training.get('max_clip_norm', None)
    if max_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()


def verbose_epoch_progress(global_rank, logger, running_results: dict, phase, epoch, train_dataset,
                           save_csv_with_preds=False):
    running_results = gather_dict(running_results)
    # logging loss values
    if is_master(global_rank):
        logger.log_epoch_loss(running_results['loss_total'], epoch, phase, prefix='total')
        if 'loss_off' in running_results:
            logger.log_epoch_loss(running_results['loss_off'], epoch, phase, prefix='off')
        if 'loss_doubt' in running_results:
            logger.log_epoch_loss(running_results['loss_doubt'], epoch, phase, prefix='doubt')

    # logging metrics
    logits = torch.cat(running_results['logits']).float()
    targets = torch.cat(running_results['targets']).long()
    metrics = calc_cls_metrics(targets, logits, only_accuracy=True)

    epoch_loss = running_results['loss_total']

    # a shortcut for binary tasks
    is_task_binary = logits.shape[-1] == 2  # during fine-tuning for syncability the task is binary
    if is_task_binary:
        if is_master(global_rank):
            logger.log_epoch_metrics(metrics, epoch, phase)
            # log to wandb
            if wandb.run is not None:
                wandb.log({'epoch': epoch})
    else:
        off_transf = get_transform_instance_from_compose(train_dataset.transforms, 'TemporalCropAndOffset')
        metrics = calc_cls_metrics(targets, logits)
        # make a map from targets (int) to labels (human-readable class names)
        label_grid = [str(round(c, 3)) for c in off_transf.class_grid.tolist()]
        target2label = {target: label for target, label in enumerate(label_grid)}
        label2metrics = calc_performance_per_class(target2label, 'off', logits, targets,
                                                   train_dataset=train_dataset)
        preds = logits.argmax(dim=1)  # id of max logit

        # compute median accuracy for k=1 and k=k
        k = min(logits.shape[-1], 5)  # top-5 is not always available (e.g. for 3 classes)
        if not all(['accuracy_1' in v for v in label2metrics.values()]):
            logging.warning('Not all offset classes have predictions. Median accs of those are replaced with 0.0')
        metrics['accuracy_1_median'] = np.median([v.get('accuracy_1', 0.0) for v in label2metrics.values()])
        metrics[f'accuracy_{k}_median'] = np.median([v.get(f'accuracy_{k}', 0.0) for v in label2metrics.values()])

        if is_master(global_rank):
            plots_and_log_perf_per_cls(logger, phase, epoch, train_dataset, label2metrics, targets, preds)
            logger.log_epoch_metrics(metrics, epoch, phase)

    return epoch_loss, metrics


# NOTE: wrap the call in `is_master` to avoid calling `plt.show()` on all processes
def plot_n_log_perf(logger, phase, epoch, train_dataset, oos_targets, off_preds_sec, off_targets):
    # make plots: 2 x 3:
    #   1st col: prediction error vs target offset (for in-sync and out-of-sync ranges)
    #   2nd col: prediction distribution (for in-sync and out-of-sync ranges)
    #   3rd col: target distribution (for in-sync and out-of-sync ranges)
    ins_mask = oos_targets == 0
    off_targets_ins = off_targets[ins_mask]
    off_preds_sec_4ins = off_preds_sec[ins_mask].squeeze(1)
    off_targets_oos = off_targets[~ins_mask]
    off_preds_sec_4oos = off_preds_sec[~ins_mask].squeeze(1)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f'{logger.start_time} | {phase} (e{epoch}) | {train_dataset.__class__.__name__}')

    # 1st row:
    ax[0, 0].set_title('prediction error vs target offset (in-sync targets)')
    ax[0, 0].set_xlabel('target offset')
    ax[0, 0].set_ylabel('prediction error (mse)')
    ax[0, 0].set_xlim(*train_dataset.itu_t_range)
    ax[0, 0].scatter(off_targets_ins, (off_preds_sec_4ins - off_targets_ins) ** 2, s=1, alpha=0.5)

    ax[0, 1].set_title('prediction distribution (in-sync targets)')
    ax[0, 1].set_xlabel('prediction')
    ax[0, 1].set_ylabel('count')
    # ax[0, 1].set_xlim(*train_dataset.itu_t_range)
    # draw two vertical lines at train_dataset.itu_t_range[0] and [1]
    ax[0, 1].axvline(train_dataset.itu_t_range[0], color='r', linestyle='--')
    ax[0, 1].axvline(train_dataset.itu_t_range[1], color='r', linestyle='--')
    ax[0, 1].hist(off_preds_sec_4ins, bins=50)

    ax[0, 2].set_title('target distribution (in-sync targets)')
    ax[0, 2].set_xlabel('target')
    ax[0, 2].set_ylabel('count')
    ax[0, 2].set_xlim(*train_dataset.itu_t_range)
    ax[0, 2].hist(off_targets_ins, bins=50)

    # 2nd row:
    ax[1, 0].set_title('prediction error vs target offset (out-of-sync targets)')
    ax[1, 0].set_xlabel('target offset')
    ax[1, 0].set_ylabel('prediction error (mse)')
    ax[1, 0].set_xlim(-train_dataset.max_off_sec, train_dataset.max_off_sec)
    ax[1, 0].scatter(off_targets_oos, (off_preds_sec_4oos - off_targets_oos) ** 2, s=1, alpha=0.5)

    ax[1, 1].set_title('prediction distribution (out-of-sync targets)')
    ax[1, 1].set_xlabel('prediction')
    ax[1, 1].set_ylabel('count')
    ax[1, 1].set_xlim(-train_dataset.max_off_sec, train_dataset.max_off_sec)
    ax[1, 1].hist(off_preds_sec_4oos, bins=50)

    ax[1, 2].set_title('target distribution (out-of-sync targets)')
    ax[1, 2].set_xlabel('target')
    ax[1, 2].set_ylabel('count')
    ax[1, 2].set_xlim(-train_dataset.max_off_sec, train_dataset.max_off_sec)
    ax[1, 2].hist(off_targets_oos, bins=50)

    fig.tight_layout()
    # save to disk
    save_dir = Path(logger.logdir) / 'viz' / 'perf_per_class'
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f'{phase}_e{epoch:04d}.png')

    # log to wandb
    if wandb.run is not None:
        wandb.log({f"{phase}/perf_per_target": wandb.Image(fig), 'epoch': epoch})


# NOTE: wrap the call in `is_master` to avoid calling `plt.show()` on all processes
def plots_and_log_perf_per_cls(logger, phase, epoch, train_dataset, label2metrics, targets, preds,
                               metric='accuracy_1'):
    # adaptive hight
    fig, ax = plt.subplots(1, 3, figsize=(13, 7*len(label2metrics)/41))
    fig.suptitle(f'{logger.start_time} | {phase} (e{epoch}) | {train_dataset.__class__.__name__}')
    fontsize = None
    width = 0.60
    # NOTE: if metric is not in label2metrics, set it to 0.0
    num_cls = len(label2metrics)

    # all sorted by label
    label_list = list(label2metrics.keys())
    target_list = list(range(len(label2metrics)))
    pred_count_list = [preds.tolist().count(target) for target in target_list]
    target_count_list = [targets.tolist().count(target) for target in target_list]
    perf_list = [label2metrics[label].get(metric, 0.0) for label in label_list]
    if all([v == 0.0 for v in perf_list]):
        logging.warning(f'All {metric} values are 0.0')

    # distribution of predictions
    _dict = {'perf_per_cls': perf_list, 'preds': pred_count_list, 'targets': target_count_list}
    for i, (name, vals) in enumerate(_dict.items()):
        ax[i].set_title(name, fontsize=fontsize)
        ax[i].set_axisbelow(True)
        ax[i].xaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
        # mean line
        ax[i].vlines(sum(vals) / len(vals), 0, num_cls, color='black', linestyles='dashed', alpha=0.7)
        y = list(range(num_cls))
        # Add some text for labels, title and custom x-axis tick labels, etc.
        x_axis_name = metric if name == 'perf_per_cls' else 'Count'
        ax[i].set_xlabel(x_axis_name, fontsize=fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].set_yticks(target_list)
        ax[i].set_yticklabels(label_list, fontsize=fontsize)
        if name == 'perf_per_cls':
            ax[i].set_xlim(left=0.0, right=1.0)

        for rect in ax[i].barh(y, vals, width, alpha=0.5, color='C0', lw=1, align='center'):
            ax[i].annotate(f'{rect.get_width():.2f}',
                           xy=(rect.get_width(), rect.get_y()),
                           xytext=(10, 1),
                           textcoords="offset points",
                           ha='center',
                           va='center',
                           fontsize=fontsize,)

        ax[i].autoscale(enable=True, axis='y', tight=True)  # tight layout inside of the figure

    fig.tight_layout()
    # save to disk
    save_dir = Path(logger.logdir) / 'viz' / 'perf_per_class'
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f'{phase}_e{epoch:04d}.png')

    # log to wandb
    if wandb.run is not None:
        wandb.log({f"{phase}/perf_per_cls_{metric}": wandb.Image(fig), 'epoch': epoch})


def calc_performance_per_class(target2label, class_category, logits_off, targets_off, targets_cls=None,
                               train_dataset=None, add_doubt_cls=False):
    logits_and_targets = [logits_off, targets_off]
    if class_category == 'cls':
        assert targets_cls is not None, 'targets_cls is None'
        logits_and_targets += [targets_cls]

    target_cls2preds = {target_cls: {'logits_off': [], 'targets_off': []} for target_cls in target2label.keys()}

    for logits_targets_i in zip(*logits_and_targets):
        if class_category == 'off':
            logits_off, target_off = logits_targets_i
            target_off = target_off.item()
            key = target_off
        elif class_category == 'cls':
            raise NotImplementedError('performance per data class is not implemented (see running_results)')
            logits_off, target_off, target_cls = logits_targets_i
            target_cls = target_cls.item()
            target_off = target_off.item()
            key = target_cls
        target_cls2preds[key]['logits_off'] += [logits_off]
        target_cls2preds[key]['targets_off'] += [target_off]

    label2metrics = {}
    for key, preds in target_cls2preds.items():
        if len(preds['logits_off']) > 0 and len(preds['targets_off']) > 0:
            # if key target class was never predicted, calc_cls_metrics will throw some warnings
            metric_name2val = calc_cls_metrics(torch.tensor(preds['targets_off']),
                                               torch.stack(preds['logits_off']),
                                               only_accuracy=True, verbose=False, add_doubt_cls=add_doubt_cls)
        else:
            metric_name2val = dict()
        label2metrics[target2label[key]] = metric_name2val
    return label2metrics


def verbose_test_progress(global_rank, logger, cfg, running_results, ckpt_epoch):
    running_results = gather_dict(running_results)

    if is_master(global_rank):
        logits = torch.cat(running_results['logits']).float()
        targets = torch.cat(running_results['targets']).long()
        metrics = calc_cls_metrics(targets, logits, add_doubt_cls=cfg.training.get('add_doubt_cls', False))
        metrics['loss'] = running_results['loss_total']
        logger.log_test_metrics(metrics, dict(cfg), ckpt_epoch)

    logging.info('Finished the experiment')


def gather_dict(dct):
    if dist.is_initialized():
        dist.barrier()
        for k, v in dct.items():
            gather_buffer = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_buffer, v)
            if isinstance(v, list):
                # flattens a list of lists into one list
                dct[k] = list(itertools.chain(*gather_buffer))
            elif isinstance(v, float):
                # average
                dct[k] = sum(gather_buffer) / len(gather_buffer)
            else:
                raise NotImplementedError(f'{type(v)}, \n{v}')
    return dct


def calc_cls_metrics(targets, outputs: torch.FloatTensor, topk=(1, 5), only_accuracy=False, prefix='',
                     verbose=True, add_doubt_cls: bool = False, calc_tol_accuracy=True,
                     softmaxed_outputs=False, calc_pr_rec_f1=False):
    """
    Adapted from https://github.com/hche11/VGGSound/blob/master/utils.py

    Calculate statistics including mAP, AUC, and d-prime.
        Args:
            targets: 1d tensors, (dataset_size, )
            output: 2d tensors, (dataset_size, classes_num) - before softmax
            topk: tuple
            only_accuracy: bool - if True, only accuracy@k will be calculated
            prefix: str - prefix for the metric names
            verbose: bool - if True, will log warnings i.e if some classes never occured in targets or outputs
        Returns:
            metric_dict: a dict of metrics
    """
    prefix = fix_prefix(prefix)
    metrics_dict = dict()

    dataset_size, num_cls = outputs.shape
    topk = [min(k, num_cls) for k in topk]

    if softmaxed_outputs:
        targets_pred = outputs.clone()
    else:
        targets_pred = torch.softmax(outputs, dim=1)

    if verbose and not torch.isfinite(outputs).all():
        # could raise an error but keeping it a warning for potential debugging runs
        outputs = torch.rand_like(outputs)
        logging.warning('infinity or loss was nan. Replacing with random values.')

    # ids of the predicted classes (same as softmax)
    _, preds = torch.topk(outputs, k=max(topk), dim=1)
    unique_preds = sorted(list(set(preds[:, 0].tolist())))  # picking only the top prediction
    if verbose and (len(unique_preds) < num_cls):
        logging.warning(f'Some classes never occured in _outputs_. {prefix} pred classes: {unique_preds}')

    # accuracy@k
    for k in topk:
        if num_cls == 2:  # binary classification
            if k == 2:  # silence the warning
                continue
            metrics_dict[f'{prefix}accuracy_{k}'] = top_k_accuracy_score(targets, targets_pred[:, 1], k=k,
                                                                         labels=range(num_cls))
        else:
            metrics_dict[f'{prefix}accuracy_{k}'] = top_k_accuracy_score(targets, targets_pred, k=k,
                                                                         labels=range(num_cls))

    # accuracy@k_tol
    if calc_tol_accuracy:
        # NOTE: we ignore the performance on items that have targets that correspond to conf class
        if add_doubt_cls:
            num_off_cls = num_cls - 1  # we don't need conf cls for metrics with tolerance
            # mask out items with doubt cls target
            doubt_items_mask = targets != num_cls - 1
            targets_for_tol = targets.clone()[doubt_items_mask]
            preds_for_tol = preds.clone()[doubt_items_mask]
        else:
            num_off_cls = num_cls
            targets_for_tol = targets.clone()
            preds_for_tol = preds.clone()
        targets_as_preds = targets_for_tol.unsqueeze(-1).expand_as(preds_for_tol)
        targets_left_tol = (targets_as_preds - 1).clamp(0, num_off_cls-1)
        targets_right_tol = (targets_as_preds + 1).clamp(0, num_off_cls-1)
        targets_for_acc_w_tol = torch.stack([targets_left_tol, targets_as_preds, targets_right_tol])
        correct_for_maxtopk_w_tol = (preds_for_tol == targets_for_acc_w_tol).any(dim=0)
        for k in topk:
            # there might be more than one `True` per item (15, 16 in top2). Preventing overcounting w/ any()
            TPs_w_tol = correct_for_maxtopk_w_tol[:, :k].any(dim=1).sum().item()
            # adding 1e-7 to avoid division by 0 which occurs when all items have confidence class target
            metrics_dict[f'{prefix}accuracy_{k}_tol1'] = TPs_w_tol / (len(correct_for_maxtopk_w_tol) + 1e-7)

        if num_off_cls == 3 and dataset_size > 100:  # 100 is to avoid verbosity on iteration level
            logging.warning('Accuracy with tolerance is not reliable as num of offset classes is 3.')

    if only_accuracy:
        return metrics_dict

    # if there are no targets of some classes, the metrics will be wrong, replacing with dummy values
    unique_targets = sorted(list(set(targets.tolist())))
    if len(unique_targets) < num_cls:
        logging.warning(f'Some classes never occured in targets. {prefix} target classes: {unique_targets}')
        # some dummy values just for the sake of error prevention
        metrics_dict[f'{prefix}mAP'] = 0.0
        metrics_dict[f'{prefix}mROCAUC'] = 0.5
        metrics_dict[f'{prefix}dprime'] = 0.0
        return metrics_dict

    # avg precision, average roc_auc, and dprime
    targets_1hot = torch.nn.functional.one_hot(targets, num_classes=num_cls)

    targets_1hot = targets_1hot.numpy()
    targets_pred = targets_pred.numpy()

    # one-vs-rest
    avg_p = [average_precision_score(targets_1hot[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]
    roc_aucs = [roc_auc_score(targets_1hot[:, c], targets_pred[:, c], average=None) for c in range(num_cls)]

    metrics_dict[f'{prefix}mAP'] = np.mean(avg_p)
    metrics_dict[f'{prefix}mROCAUC'] = np.mean(roc_aucs)
    # Percent point function (ppf) (inverse of cdf — percentiles).
    metrics_dict[f'{prefix}dprime'] = scipy.stats.norm().ppf(metrics_dict[f'{prefix}mROCAUC'])*np.sqrt(2)

    if calc_pr_rec_f1:
        metrics_dict[f'{prefix}precision'] = precision_score(targets, preds[:, 0], zero_division=0.0)
        metrics_dict[f'{prefix}recall'] = recall_score(targets, preds[:, 0], zero_division=0.0)
        metrics_dict[f'{prefix}f1'] = f1_score(targets, preds[:, 0], zero_division=0.0)

    return metrics_dict
