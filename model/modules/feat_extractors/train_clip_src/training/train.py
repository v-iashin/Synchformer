import json
import logging
import math
import os
from pathlib import Path
import time
import signal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import distributed as dist
from model.modules.feat_extractors.train_clip_src.open_clip.model import MultilevelMoCoCLIP
from matplotlib import pyplot as plt
import torchaudio

from scripts.train_utils import gather_dict
from utils.utils import get_param_by_name_from_transform_cfg

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_cast_dtype
from .distributed import is_master
from .precision import get_autocast


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


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    for_loop_segment_fwd = args.training.get('for_loop_segment_fwd', False)
    device = torch.device(args.device)
    autocast = get_autocast(args.training.precision)
    cast_dtype = get_cast_dtype(args.training.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if dist.is_initialized():
        to_add_global_repr = getattr(model.module, 'to_add_global_repr', None)
    else:
        to_add_global_repr = getattr(model, 'to_add_global_repr', None)

    losses_m = dict()
    metrics_m = dict()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        rgb = batch['video'].to(device, dtype=cast_dtype, non_blocking=True)
        audio = batch['audio'].to(device, dtype=cast_dtype, non_blocking=True)

        train_step = num_batches_per_epoch * epoch + i

        if not args.training.skip_scheduler:
            scheduler(train_step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # saving shapes because we will flatten batch dimension with positive and negative samples
        assert rgb.shape[:2] == audio.shape[:2], f'rgb.shape: {rgb.shape}, audio.shape: {audio.shape}'
        B, S, C, Tv, H, W = rgb.shape
        B, S, Ta, F = audio.shape

        # as in ALBEF
        alpha = args.training.alpha * min(1, i / len(dataloader)) if epoch == 0 else args.training.alpha

        # saves recontructed input to the model during the first iteration (detects bugs)
        if is_master(args) and train_step == 0:
            logging.info('Saving reconstructed input to the model.')
            vis_inputs(args, rgb, audio, batch, train_step, args.transform_sequence_train)

        with autocast():
            model_out = model(rgb, audio, alpha, for_loop_segment_fwd, args.world_size)
            logit_scale = model_out['logit_scales']
            if args.distill:
                with torch.no_grad():
                    dist_model_out = dist_model(rgb, audio, alpha, for_loop_segment_fwd)
                model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})

            if 'losses' in model_out:
                losses = model_out['losses']
            else:
                losses = loss(**model_out, output_dict=True)

            # weight the losses
            losses['segment_contrastive_loss'] *= args.training.segment_loss_weight
            if to_add_global_repr:
                losses['global_contrastive_loss'] *= args.training.global_loss_weight
            # compute the global loss
            total_loss = sum(losses.values())
            losses["loss"] = total_loss

        backward(total_loss, scaler)

        if scaler is not None:
            if args.training.max_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.max_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.training.max_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.max_clip_norm, norm_type=2.0)
            optimizer.step()

        #### done during the model.forward()
        # # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # with torch.no_grad():
        #     unwrap_model(model).segment_logit_scale.clamp_(0, math.log(100))
        #     if model.to_add_global_repr:
        #         unwrap_model(model).global_logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        # NOTE loss and metrics are coarsely sampled, just master node and per log update
        if is_master(args) and (i % args.logging.log_frequency == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(rgb)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            seen_samples = samples_per_epoch * epoch + num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            with autocast():
                model.eval()
                _, metrics = eval_one_example(args, model, rgb, audio, 'train', train_step, i)
                model.train()

            for k, v in losses.items():
                if k not in losses_m:
                    losses_m[k] = AverageMeter()
                losses_m[k].update(v.item(), batch_size)
            for k, v in metrics.items():
                if k not in metrics_m:
                    metrics_m[k] = AverageMeter()
                metrics_m[k].update(v.item(), batch_size)

            segment_logit_scale_scalar = logit_scale[0].item()
            if to_add_global_repr:
                global_logit_scale_scalar = logit_scale[1].item()

            loss_log = ' '.join([f"{k.capitalize()}: {v.val:#.5g} ({v.avg:#.5g})" for k, v in losses_m.items()])
            metric_log = ' '.join([f"{k.capitalize()}: {v.val:#.5g} ({v.avg:#.5g})" for k, v in metrics_m.items()])
            samples_per_s = batch_size * args.world_size / batch_time_m.val
            samples_per_s_per_gpu = batch_size / batch_time_m.val
            msg = f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
            msg += f"Data (t): {data_time_m.avg:.3f} "
            msg += f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_s:#g}/s, {samples_per_s_per_gpu:#g}/s/gpu "
            msg += f"LR: {optimizer.param_groups[0]['lr']:5f} "
            msg += f'{loss_log} '
            msg += f'{metric_log} '
            msg += f"Logit Scale (local): {segment_logit_scale_scalar:.3f} "
            msg += f"Logit Scale (global): {global_logit_scale_scalar:.3f} " if to_add_global_repr else ''
            logging.info(msg)

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_s": samples_per_s,
                "samples_per_s_per_gpu": samples_per_s_per_gpu,
                "segment_scale": segment_logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "alpha": alpha,
            }
            if to_add_global_repr:
                log_data.update({'global_scale': global_logit_scale_scalar})
            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data.update({name: val.val for name, val in metrics_m.items()})

            log_data = {f'train/{name}': val for name, val in log_data.items()}
            for name, val in log_data.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, train_step)
            if args.logging.use_wandb:
                assert wandb is not None, 'Please install wandb.'
                wandb.log({**log_data, 'train_step': train_step, 'seen_samples': seen_samples})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    # NOTE loss and metrics are coarsely sampled, just master node and per log update
    if is_master(args) and args.save_logs:
        metrics_m_repr = '\t'.join([f'{k}: {round(v.avg, 4):.4f}' for k, v in metrics_m.items()])
        losses_m_repr = '\t'.join([f'{k}: {round(v.avg, 4):.4f}' for k, v in losses_m.items()])
        metrics_ep = {name: v.avg for name, v in metrics_m.items()}  # extracting vals from AverageMeters as floats
        losses_ep = {name: v.avg for name, v in losses_m.items()}
        logging.info(f'Train Epoch: {epoch} (Worker #{args.rank}) {metrics_m_repr} {losses_m_repr}')
        dct_to_log = {**metrics_ep, **losses_ep}  # merge two dicts in one dict, and take .items()
        dct_to_log = {f'train/{name}': val for name, val in dct_to_log.items()}
        for name, val in dct_to_log.items():
            if tb_writer is not None:
                tb_writer.add_scalar(name, val, epoch)
        if args.logging.use_wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({**dct_to_log, 'epoch': epoch})

        with open(os.path.join(args.checkpoint_path, "results_train.jsonl"), "a+") as f:
            f.write(json.dumps({'epoch': epoch, 'metrics': metrics_ep}))
            f.write("\n")

    if args.distributed:
        # logging.info(f'Train (Worker #{args.rank}): reached the end of train.')
        dist.barrier()
        if is_master(args):
            logging.info('All workers reached the end of the train epoch.')


def evaluate_on_sync_w_shifts(model: MultilevelMoCoCLIP, data, phase, epoch, args, tb_writer=None, loss=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.training.precision)
    cast_dtype = get_cast_dtype(args.training.precision)

    model.eval()

    data[phase].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data[phase].dataloader
    num_batches_per_epoch = dataloader.num_batches
    samples_per_epoch = dataloader.num_samples

    sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))

    if dist.is_initialized():
        to_add_global_repr = getattr(model.module, 'to_add_global_repr', None)
    else:
        to_add_global_repr = getattr(model, 'to_add_global_repr', None)

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    num_samples = 0
    metrics_loc_m = dict()
    losses_loc_m = dict()

    for i, batch in enumerate(dataloader):
        rgb = batch['video'].to(device, dtype=cast_dtype, non_blocking=True)
        audio = batch['audio'].to(device, dtype=cast_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)

        B, S, C, Tv, H, W = rgb.shape
        B, S, Ta, F = audio.shape

        # save recontructed model input during the first iteration (detects bugs) must be out of no_grad
        valid_step = num_batches_per_epoch * epoch + i
        if i == 0:
            if is_master(args):
                vis_inputs(args, rgb, audio, batch, valid_step, args.transform_sequence_test)
            if args.distributed:
                dist.barrier()

        with autocast(), torch.no_grad():
            losses_sample, metrics_sample = eval_one_example(args, model, rgb, audio, phase, valid_step, i)

        # weight the losses
        losses_sample['segment_contrastive_loss'] *= args.training.segment_loss_weight
        if to_add_global_repr:
            losses_sample['global_contrastive_loss'] *= args.training.global_loss_weight
        # compute the global loss
        total_loss = sum(losses_sample.values())
        losses_sample['loss'] = total_loss

        # update meters
        for k, v in losses_sample.items():
            if k not in losses_loc_m:
                losses_loc_m[k] = AverageMeter()
            losses_loc_m[k].update(v.item(), B)
        for k, v in metrics_sample.items():
            if k not in metrics_loc_m:
                metrics_loc_m[k] = AverageMeter()
            metrics_loc_m[k].update(v.item(), B)

        batch_time_m.update(time.time() - end)
        end = time.time()
        num_samples += B * args.world_size
        batch_count = i + 1
        percent_complete = 100.0 * batch_count / num_batches_per_epoch

        samples_per_s = B * args.world_size / batch_time_m.val
        samples_per_s_per_gpu = B / batch_time_m.val

        # NOTE logging is coarsely sampled, just master node
        if i % args.logging.log_frequency == 0:
            if is_master(args):
                metric_log = " ".join([f"{t.capitalize()}: {m.val:#.5g} ({m.avg:#.5g})" for t, m in metrics_loc_m.items()])
                loss_log = " ".join([f"{t.capitalize()}: {m.val:#.5g} ({m.avg:#.5g})" for t, m in losses_loc_m.items()])
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_s:#g}/s, {samples_per_s_per_gpu:#g}/s/gpu\t"
                    f'{metric_log}\t'
                    f'{loss_log}'
                )
                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_s": samples_per_s,
                    "samples_per_s_per_gpu": samples_per_s_per_gpu,
                }
                log_data.update({name: val.val for name, val in losses_loc_m.items()})
                log_data.update({name: val.val for name, val in metrics_loc_m.items()})
                # adding phase name to the keys
                log_data = {f'{phase}/{name}': val for name, val in log_data.items()}
                for name, val in log_data.items():
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, valid_step)
                if args.logging.use_wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({**log_data, f'{phase}_step': valid_step})

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
            if args.distributed:
                dist.barrier()

    metrics_m_repr = '\t'.join([f'{k}: {round(v.avg, 4):.4f}' for k, v in metrics_loc_m.items()])
    logging.info(f'Eval Epoch: {epoch} (Worker #{args.rank}) {metrics_m_repr}')
    metrics_local = {name: v.avg for name, v in metrics_loc_m.items()}  # extracting vals from AverageMeters as floats
    losses_local = {name: v.avg for name, v in losses_loc_m.items()}  # extracting vals from AverageMeters as floats
    metrics_global = gather_dict(metrics_local)  # if in ddp, gathers the object from workers, mean for vals, cat for lists
    losses_global = gather_dict(losses_local)
    # logging.info(f'Eval (Worker #{args.rank}): gathered metrics.')

    if is_master(args) and args.save_logs:
        metrics_m_repr = '\t'.join([f'{k}: {v:.4f}' for k, v in metrics_global.items()])
        losses_m_repr = '\t'.join([f'{k}: {v:.4f}' for k, v in losses_global.items()])
        logging.info(f'Eval Epoch: {epoch} (All workers) {metrics_m_repr} {losses_m_repr}')
        dct_to_log = {**metrics_global, **losses_global}  # merge two dicts in one dict
        dct_to_log = {f'{phase}/{name}': val for name, val in dct_to_log.items()}  # add phase to the key
        for name, val in dct_to_log.items():
            if tb_writer is not None:
                tb_writer.add_scalar(name, val, epoch)
        if args.logging.use_wandb:
            assert wandb is not None, 'Please install wandb.'
            wandb.log({**dct_to_log, 'epoch': epoch})

        with open(os.path.join(args.checkpoint_path, f"results_{phase}.jsonl"), "a+") as f:
            f.write(json.dumps({'epoch': epoch, 'metrics': metrics_global}))
            f.write("\n")

    if args.distributed:
        # logging.info(f'Eval (Worker #{args.rank}): reached the end of eval.')
        dist.barrier()
        if is_master(args):
            logging.info('All workers reached the end of the eval epoch.')

    return metrics_global

@torch.no_grad()
def eval_one_example(args, model: MultilevelMoCoCLIP, rgb: torch.Tensor, audio: torch.Tensor, phase: str,
                     global_step: int, local_step: int):
    for_loop_segment_fwd = args.training.get('for_loop_segment_fwd', False)
    assert not model.training, 'Model should be in eval mode.'
    B, S, C, Tv, H, W = rgb.shape
    B, S, Ta, F = audio.shape
    device = torch.device(rgb.device)

    out = model.forward_for_logging(rgb, audio, for_momentum=False, for_loop=for_loop_segment_fwd, do_norm=True)
    segment_afeat, segment_vfeat = out['segment_afeat'], out['segment_vfeat']
    segment_sim_v2a, segment_sim_a2v = out['segment_sim_v2a'], out['segment_sim_a2v']
    segment_sim_v2v, segment_sim_a2a = out['segment_sim_v2v'], out['segment_sim_a2a']
    losses = {'segment_contrastive_loss': out['segment_contrastive_loss']}
    if 'global_contrastive_loss' in out:
        losses['global_contrastive_loss'] = out['global_contrastive_loss']

    # show the similarity matrix on the image
    # NOTE: visualizing coarsely, just master node
    if is_master(args) and (local_step == 0 or (global_step % (args.logging.log_frequency * 20)) == 0):
        log_sim_matrices(args, segment_sim_v2a, segment_sim_a2v, segment_sim_v2v, segment_sim_a2a, phase,
                         global_step)

    # now, compute the predictions for the shifted segments
    D = segment_afeat.shape[-1]
    # (B, S, D) <- (B*S, D)
    segment_afeat = segment_afeat.view(B, S, D)
    segment_vfeat = segment_vfeat.view(B, S, D)
    # window size in segments and if on test, then use the validation win size
    W = args.training.get(f'run_shifted_win_val_winsize_{phase.replace("test", "valid")}')
    assert W < segment_vfeat.shape[-2], f'Win size ({W}) should be < than the number of segments.'
    # for each window of size W in A, returns the index of the closest representation among all
    # shifted windows of size W in V
    preds_a, preds_v = shift_and_get_preds(segment_afeat, segment_vfeat, W)  # (B, n_shifts)
    # get the ground truth for each window
    n_shifts = preds_a.shape[-1]
    gt = get_gt(n_shifts, device)
    # get the metrics (precision: how many shifts were guessed correctly)
    metrics = calc_cls_metrics((preds_a, preds_v), gt, ['precision'])  # {str: (B, )}
    metrics = {k: v.mean() for k, v in metrics.items()}  # {str: tensor}  average over batch
    return losses, metrics

def log_sim_matrices(args, sim_v2a, sim_a2v, sim_v2v, sim_a2a, phase: str, global_step: int):
    # to be used during both Stages (I and II), hence adapting to different configs
    log_dir = args.logging.logdir
    exp_name = args.get('name', args.start_time)
    to_use_wandb = args.logging.use_wandb

    ## Put the above on one plot (2x2):
    save_dir = Path(log_dir) / exp_name / 'viz' / 'sim_matrices'
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 10))
    for i, (name, sim) in enumerate(zip(['v2a', 'a2v', 'v2v', 'a2a'], [sim_v2a, sim_a2v, sim_v2v, sim_a2a])):
        plt.subplot(2, 2, i + 1)
        sim = min_max_scale_rowwise(sim)  # some rows have very low values than others, so we re-scale them
        plt.imshow(sim.cpu().numpy())
        plt.title(name)
    plt.tight_layout()
    if to_use_wandb:
        wandb.log({f'{phase}/chart': plt, f'{phase}_step': global_step})
    plt.savefig(save_dir / f'{phase}_{str(global_step).zfill(6)}.png')
    plt.close()

    logging.info(f'Saved similarity matrices to {str(save_dir)}')

def vis_inputs(args, vis, aud, batch, global_step, transforms, max_vids_per_batch=2):
    '''
    Args:
        vis (torch.Tensor): RGB frames (B, S, C, Tv, H, W)
        aud (torch.Tensor): audio spectrograms (B, S, Ta, F)
        batch (dict): batch
        global_iter (int): batch index (across all epochs, ie i + num_batches_per_epoch * epoch)
        transforms (omegaconf): transforms config
    '''
    aud_rec = aud.clone().cpu()
    vis_rec = vis.clone().cpu()
    B, S, C, Tv, H, W = vis_rec.shape
    B, S, Ta, F = aud_rec.shape

    save_dir = Path(args.logging.logdir) / args.name / 'viz' / 'inputs'
    save_dir.mkdir(exist_ok=True, parents=True)
    for b in range(min(B, max_vids_per_batch)):  # vids
        for s in range(len(aud_rec[b])):  # segments
            aud_rec_bs = aud_rec[b, s]
            vis_rec_bs = vis_rec[b, s]
            # dataset.transforms.PermuteStreams
            aud_rec_bs = aud_rec_bs.mT  # (F, Ta); .mT is like .T but for the last 2 dims
            vis_rec_bs = vis_rec_bs.permute(1, 2, 3, 0)  # (Tv, H, W, C)
            # dataset.transforms.AudioNormalizeAST
            spec_means = batch['meta']['audio']['norm_stats']['mean'][b]
            spec_stds = batch['meta']['audio']['norm_stats']['std'][b]
            aud_rec_bs = aud_rec_bs * spec_stds * 2 + spec_means
            spec_to_show = aud_rec_bs.clone()  # to be saved as a picture
            # dataset.transforms.AudioLog
            aud_rec_bs = aud_rec_bs.exp()
            # dataset.transforms.AudioMelSpectrogram
            # reconstruct the full spectrogram from mel spectrogram
            n_fft = get_param_by_name_from_transform_cfg(transforms, 'AudioMelSpectrogram', 'n_fft')
            win_length = get_param_by_name_from_transform_cfg(transforms, 'AudioMelSpectrogram', 'win_length')
            hop_length = get_param_by_name_from_transform_cfg(transforms, 'AudioMelSpectrogram', 'hop_length')
            afps = batch['meta']['audio']['framerate'][0][b].item()
            vfps = batch['meta']['video']['fps'][0][b].item()
            spec_recon = torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=F,
                                                               sample_rate=afps)
            waveform_recon = torchaudio.transforms.GriffinLim(n_fft=n_fft, win_length=win_length,
                                                              hop_length=hop_length)

            # save path
            split = batch['split'][b]
            path = Path(batch['path'][b])
            dataset = args.data.dataset.target.split('.')[-1]
            vid_id = f"{path.parent.stem}-{path.stem}" if 'LRS3' in dataset else path.stem
            save_path_stem = Path(save_dir) / f'{split}_{str(global_step).zfill(6)}_{vid_id}_seg{s}'

            # `spec_recon` may stuck when the spec is bad. We time it out and replace with random noise
            try:
                signal.signal(signal.SIGALRM, lambda signum, frame: print(f'WARNING: {signum} {frame}'))
                signal.alarm(10)  # 10 sec timeout
                aud_rec_bs = spec_recon(aud_rec_bs)  # (n_fft // 2 + 1, Ta)
                signal.alarm(0)
            except Exception as e:
                aud_rec_bs = torch.rand(n_fft // 2 + 1, aud_rec_bs.shape[-1])  # random noise
                logging.warning(f'InverseMelScale failed, replaced with random noise at {save_path_stem} {e}')
            waveform_bs = waveform_recon(aud_rec_bs)  # (T,)
            waveform_bs = waveform_bs.unsqueeze(0)  # (1, T)

            # dataset.transforms.RGBNormalize
            rgb_means = batch['meta']['video']['norm_stats']['mean'][b].view(1, 1, 1, 3)
            rgb_stds = batch['meta']['video']['norm_stats']['std'][b].view(1, 1, 1, 3)
            vis_rec_bs = vis_rec_bs * rgb_stds + rgb_means
            # dataset.transforms.RGBToFloatToZeroOne
            vis_rec_bs = (vis_rec_bs * 255).short()

            # save video
            torchvision.io.write_video(str(save_path_stem.with_suffix('.mp4')), vis_rec_bs, vfps,
                                       audio_array=waveform_bs, audio_fps=afps, audio_codec='aac')
            # save the spectrogram
            fig = plt.Figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(spec_to_show, cmap='gist_gray', origin='lower')
            fig.savefig(save_path_stem.with_suffix('.jpg'), bbox_inches='tight')
            plt.close(fig)
    # save reconstructed input
    logging.info('Saved reconstructed inputs to ' + str(save_dir))

def shift_and_get_preds(a: torch.Tensor, v: torch.Tensor, W: int) -> torch.Tensor:
    '''
    Inputs:
        a: torch.Tensor (B, S, D) - logits from each segment (S) of audio;
        v: torch.Tensor (B, S, D) - logits from each segment (S) of rgb;
        W: int - window size
    Returns:
        tuple of torch.Tensor (B, n_shifts) - most similar window (of size W) from A in V and vice versa
    '''
    assert a.shape == v.shape, f'{a.shape} != {v.shape}'
    B, S, D = a.shape

    # .unfold makes a sliding window of size W over the segment dimension, makes shifted copies
    a_folds = a.unfold(dimension=-2, size=W, step=1)  # (B, n_shifts, D, W)
    v_folds = v.unfold(dimension=-2, size=W, step=1)

    _, n_shifts, _, _ = a_folds.shape
    assert n_shifts == S - W + 1, f'{n_shifts} != {S - W + 1}'

    # assuming that aggregation of elements in a window is sum (could be mean) -> W is latent dim as well
    a_folds = a_folds.contiguous().view(B, n_shifts, D*W)  # (B, n_shifts, D*W)
    v_folds = v_folds.contiguous().view(B, n_shifts, D*W)  # (B, n_shifts, D*W)

    # pairwise similarity matrix beteen all windows in V vs A
    sim = a_folds @ v_folds.mT  # (B, n_shifts, n_shifts); .mT is like .T but for last 2 dims (batch-safe)

    # get top-1 predictions (along cols and rows of the sim matrix for each element in the batch)
    # intuitivelly, for each window in A, finds the most similar window in V, and vice-versa
    preds_a, preds_v = torch.argmax(sim, dim=-2), torch.argmax(sim, dim=-1)  # (B, n_shifts)

    return preds_a, preds_v  # (B, n_shifts)

def get_gt(n_shifts: int, device: torch.device) -> torch.Tensor:
    '''
    Assumes that the segments are in-sync, i.e. the first segment in A corresponds to the first
    segment in V. Hence, n_shifts for A is the same as for V.

    inputs:
        n_shifts: int - number of shifts
        device: torch.device - device to put the tensor on
    returns:
        gt: tuple of torch.Tensor (1, n_shifts) - ground truth (batch-broadcastable)
    '''
    return torch.arange(n_shifts, device=device).view(1, n_shifts)  # (B, n_shifts), (B, n_shifts)

def calc_cls_metrics(preds: tuple, gt: torch.Tensor, types: list) -> dict:
    '''
    inputs:
        preds: tuple of torch.Tensor (B, n_shifts) - top-1 predictions for A and V
        gt: torch.Tensor (1, n_shifts) - ground truth
        types: list of str - metrics to calculate
    returns:
        out: dict mapping strings to torch.Tensor (B, ) - metrics for each modality and element
                                                            in the batch
    '''
    (preds_a, preds_v) = preds
    B, n_shifts = gt.shape

    out = dict()
    if 'precision' in types:
        # count true positives
        out['precision_a'] = torch.sum(preds_a == gt, dim=-1) / n_shifts  # (B)
        out['precision_v'] = torch.sum(preds_v == gt, dim=-1) / n_shifts
        out['precision'] = (out['precision_a'] + out['precision_v']) / 2  # (B)
    return out

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def min_max_scale_rowwise(x: torch.Tensor) -> torch.Tensor:
    '''x is (BS, BS), scales such that min(x[i, :]) = 0 and max(x[i, :]) = 1'''
    assert len(x.shape) == 2, f'x should be 2D, got {x.shape}'
    mins, _ = x.min(dim=-1, keepdim=True)  # (BS, 1), indices are ignored (_)
    maxs, _ = x.max(dim=-1, keepdim=True)
    return (x - mins) / (maxs - mins)

def softmax_rowwise(x: torch.Tensor) -> torch.Tensor:
    '''x is (BS, BS), applies softmax along rows'''
    assert len(x.shape) == 2, f'x should be 2D, got {x.shape}'
    return torch.softmax(x, dim=-1)


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
