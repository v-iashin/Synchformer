from matplotlib import pyplot as plt
import logging
import os
from pathlib import Path
from shutil import copytree, ignore_patterns, copy

import torch
from torchaudio.transforms import Spectrogram, GriffinLim, InverseMelScale
import torchvision
import wandb
from omegaconf import OmegaConf
from model.modules.feat_extractors.train_clip_src.training.train import log_sim_matrices
from scripts.train_utils import get_curr_time_w_random_shift, is_master
from torch.utils.tensorboard import SummaryWriter, summary

from utils.utils import fix_prefix, get_param_by_name_from_transform_cfg


class LoggerWithTBoard(SummaryWriter):

    def __init__(self, global_rank, cfg):
        self.start_time = cfg.start_time
        self.logdir = os.path.join(cfg.logging.logdir, self.start_time)

        # setup logging for all workers but log to a file only on the master
        self.setup_logging(to_resume=cfg.training.resume, global_rank=global_rank)

        # if not the master process, be silent and fail if mistakingly called
        if not is_master(global_rank):
            # just making a placeholder to broadcast to
            cfg.ckpt_path = None
            return None

        if not any([cfg.training.run_test_only, cfg.training.resume, cfg.training.finetune]):
            # self.logdir!
            cfg.ckpt_path = os.path.join(self.logdir, f'{self.start_time}.pt')

        self.ckpt_path = cfg.ckpt_path

        # weights and biases
        self.use_wandb = cfg.logging.use_wandb
        if self.use_wandb:
            wandb.init(
                dir=cfg.logging.logdir,
                name=cfg.start_time,
                project=f'avsync-{cfg.action}',
                config=OmegaConf.to_container(cfg, resolve=[True | False]),
                sync_tensorboard=True,
                id=cfg.start_time,
                resume='must' if cfg.training.resume else None,
            )
            wandb_patterns_to_ignore = tuple(x.replace('*', '') for x in cfg.logging.patterns_to_ignore)
            wandb.run.log_code('.', exclude_fn=lambda x: x.endswith(wandb_patterns_to_ignore))

        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        # this will initialize the SummaryWriter (tboard)
        super().__init__(self.logdir)

        if cfg.training.finetune:
            cfg.ckpt_path = copy(self.ckpt_path, os.path.join(self.log_dir, f'{self.start_time}.pt'))
            self.ckpt_path = cfg.ckpt_path
            print(f'Finetuning. The ckpt is copied to {self.ckpt_path}')

        now = get_curr_time_w_random_shift()

        # backup the cfg
        cfg_path = Path(self.log_dir) / f'cfg-{self.start_time}.yaml'
        # if exists, the fname will have the current time stamp
        if cfg_path.exists():
            cfg_path = cfg_path.parent / cfg_path.name.replace(self.start_time, now)
        OmegaConf.save(cfg, cfg_path)
        # backup the code state
        if cfg.logging.log_code_state:
            dest_dir = os.path.join(self.logdir, f'code-{self.start_time}')
            if not os.path.exists(dest_dir):
                copytree(os.getcwd(), dest_dir, ignore=ignore_patterns(*cfg.logging.patterns_to_ignore))

    def setup_logging(self, to_resume, save_to_file=True, global_rank=0):
        logging_level = logging.INFO if is_master(global_rank) else logging.WARNING
        # seting up logging
        fmt = f'%(asctime)s @ {global_rank:2d} | %(levelname)s | %(message)s'
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d,%H-%M-%S')
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
        # log file, if run for the second time, should append new logs
        if save_to_file:
            Path(self.logdir).mkdir(parents=True, exist_ok=True)
            log_file = os.path.join(self.logdir, f'log-{self.start_time}.log')
            file_handler = logging.FileHandler(log_file, mode='a' if to_resume else 'w')
            file_handler.setFormatter(formatter)
            logging.root.addHandler(file_handler)

    def log_param_num(self, global_rank, model):
        if global_rank == 0:
            # for name, param in model.named_parameters():
            # if param.requires_grad:
            #     print(name, param.data.numel())
            param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f'The number of parameters: {param_num/1e+6:.3f} mil')
            self.add_scalar('num_params', param_num, 0)
            return param_num

    def log_iter_loss(self, loss, iter, phase, prefix: str = ''):
        self.add_scalar(f'{phase}/{fix_prefix(prefix)}loss_iter', loss, iter)

    def log_epoch_loss(self, loss, epoch, phase, prefix: str = ''):
        self.add_scalar(f'{phase}/{fix_prefix(prefix)}loss', loss, epoch)
        logging.info(f'{phase} ({epoch}): {fix_prefix(prefix)}loss {loss:.3f};')

    def log_epoch_metrics(self, metrics_dict, epoch, phase, prefix: str = ''):
        for metric, val in metrics_dict.items():
            self.add_scalar(f'{phase}/{fix_prefix(prefix)}{metric}', val, epoch)
        metrics_dict = {k: round(v, 4) for k, v in metrics_dict.items()}
        logging.info(f'{phase} ({epoch}) {fix_prefix(prefix)}metrics: {metrics_dict};')

    def log_test_metrics(self, metrics_dict, hparams_dict, best_epoch, prefix: str = ''):
        allowed_types = (int, float, str, bool, torch.Tensor)
        hparams_dict = {k: v for k, v in hparams_dict.items() if isinstance(v, allowed_types)}
        metrics_dict = {f'test/{fix_prefix(prefix)}{k}': round(v, 4) for k, v in metrics_dict.items()}
        exp, ssi, sei = summary.hparams(hparams_dict, metrics_dict)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metrics_dict.items():
            self.add_scalar(k, v, best_epoch)
        logging.info(f'test ({best_epoch}) {fix_prefix(prefix)}metrics: {metrics_dict};')

    def log_model(self, model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg, suffix):
        checkpoint = {
            'args': cfg,
            'loss': loss,
            'metrics': metrics_dict,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'scaler': scaler.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'model_type': model.__class__.__name__,
        }
        save_path = Path(self.ckpt_path).parent / f'{Path(self.ckpt_path).stem}{suffix}.pt'
        torch.save(checkpoint, str(save_path))
        logging.info(f'Saved {suffix} model in {str(save_path)}')

    def log_best_model(self, model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg):
        self.log_model(model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg, '')
        self.log_model(model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg, '_best')

    def log_latest_model(self, model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg):
        self.log_model(model, scaler, loss, epoch, optimizer, lr_scheduler, metrics_dict, cfg, '_latest')

    def vizualize_input(self, vid: torch.Tensor, aud: torch.Tensor, batch, global_iter: int, phase: str,
                        cfg: OmegaConf, max_vids_per_batch=2):
        ''' [B, (S, ) Tv, C, H, W] [B, (S, ) 1, F, Ta] - either with segment or not '''
        orig_B = aud.shape[0]
        Tv, C, H, W = vid.shape[-4:]  # works for both with and without segments
        F, Ta = aud.shape[-2:]
        aud_rec = aud.cpu()[:max_vids_per_batch]
        vid_rec = vid.cpu()[:max_vids_per_batch]
        B = vid_rec.shape[0]

        # if segments, concat them along T. Remember that audio is padded -> rec audio is weird
        is_input_segmented = len(vid_rec.shape) == 6
        if is_input_segmented:
            assert cfg.data.get('step_size_seg', 1.0) in [1.0, 0.5], 'only 1.0 and 0.5 are supported for now'
            if cfg.data.get('step_size_seg', 1.0) == 0.5:
                # drop every second segment
                vid_rec = vid_rec[:, ::2].contiguous()
                aud_rec = aud_rec[:, ::2].contiguous()
            S = vid_rec.shape[1]
            vid_rec = vid_rec.view(B, S * Tv, C, H, W)
            aud_rec = aud_rec.permute(0, 1, 4, 2, 3).contiguous().view(B, S * Ta, 1, F).permute(0, 2, 3, 1)

        a_means = batch['meta']['audio']['norm_stats']['mean'].view(orig_B, 1, -1, 1)[:max_vids_per_batch]
        a_stds = batch['meta']['audio']['norm_stats']['std'].view(orig_B, 1, -1, 1)[:max_vids_per_batch]
        v_means = batch['meta']['video']['norm_stats']['mean'].view(orig_B, 1, 3, 1, 1)[:max_vids_per_batch]
        v_stds = batch['meta']['video']['norm_stats']['std'].view(orig_B, 1, 3, 1, 1)[:max_vids_per_batch]

        # AudioStandardNormalize  (AST normaliization is done by (x - m) / (2*s) )
        if is_input_segmented:
            aud_rec = aud_rec * a_stds * 2 + a_means
        else:
            aud_rec = aud_rec * a_stds + a_means
        # AudioLog
        aud_rec = torch.exp(aud_rec)
        # AudioSpectrogram
        if not hasattr(self, 'griffinlim'):
            if is_input_segmented:
                t_cfg = cfg.transform_sequence_test if phase == 'test' else cfg.transform_sequence_train
                n_fft = get_param_by_name_from_transform_cfg(t_cfg, 'AudioMelSpectrogram', 'n_fft')
                win_length = get_param_by_name_from_transform_cfg(t_cfg, 'AudioMelSpectrogram', 'win_length')
                hop_length = get_param_by_name_from_transform_cfg(t_cfg, 'AudioMelSpectrogram', 'hop_length')
                afps = batch['meta']['audio']['framerate'][0][0].item()  # assuming all the same in the batch
                vfps = batch['meta']['video']['fps'][0][0].item()  # assuming all the same in the batch
                self.griffinlim = torchvision.transforms.Compose([
                    InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=F, sample_rate=afps,
                                    tolerance_change=0.00001),
                    GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length),
                ])
                self.wav2spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            else:
                self.griffinlim = GriffinLim(n_fft=512, hop_length=128)
                self.wav2spec = Spectrogram(n_fft=512, hop_length=128)
        aud_rec = aud_rec.float()
        logging.info('Converting spec to audio...')
        aud_rec = self.griffinlim(aud_rec)
        logging.info('Done converting spec to audio.')

        # RGBNormalize
        vid_rec = vid_rec * v_stds + v_means
        # RGBToFloatToZeroOne
        vid_rec = (vid_rec * 255).short()

        vid_rec = vid_rec.permute(0, 1, 3, 4, 2)

        save_dir = os.path.join(self.logdir, 'viz')
        os.makedirs(save_dir, exist_ok=True)
        for b in range(min(B, max_vids_per_batch)):
            offset_sec = batch['targets'].get('offset_sec', [0.0] * len(aud_rec))[b]
            vfps = batch['meta']['video']['fps'][0][b]
            audio_fps = batch['meta']['audio']['framerate'][0][b]
            vid_id = Path(batch['path'][b]).stem
            save_path = Path(save_dir) / 'rec_in' / f'{global_iter:06d}_{phase}_{vid_id}_{offset_sec:.2f}.mp4'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(str(save_path), vid_rec[b], vfps.item(),
                                       audio_array=aud_rec[b], audio_fps=audio_fps.item(), audio_codec='aac')
            fig = plt.Figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            spec_rec_b = self.wav2spec(aud_rec[b]).permute(1, 2, 0).log().numpy()
            ax.imshow(spec_rec_b.squeeze(), cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            fig.savefig(str(save_path.with_suffix('.jpg')))

    def vizualize_segment_sim(self, cfg, model, vid, aud, iter_step, phase):
        assert not model.training, 'model should be in eval mode'
        # Stage I: afeat_extractor.agg_time_module should be AveragePooling
        # Stage I: `a/vproj.target` should be torch.nn.Identity
        logging.warning('Time dim aggregated with AveragePooling')
        logging.warning('Projection is not supported for now. Assuming it was not used.')

        def _get_modality_segment_logits(extract_fn, x):
            # get segment features
            for_loop = False
            assert for_loop is False, 'for_loop is not supported for now. The checkerboard pattern on viz'
            feat = extract_fn(x, for_loop=for_loop)  # (B, S, tv, D) or (B, S, ta, D)
            # aggregate time dim
            # TODO: only AveragePooling is supported for now for time aggregation
            feat = feat.mean(dim=-2)  # (B, S, D)
            # project
            # TODO: only Identity is supported for now for projection
            pass
            # normalize (assumes that the inputs were normalized during Stage I)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            # flatten batch and segment dim
            B, S, D = feat.shape
            feat = feat.view(B*S, D)
            return feat

        with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
            with torch.no_grad():
                segment_v_feat = _get_modality_segment_logits(model.extract_vfeats, vid)
                segment_a_feat = _get_modality_segment_logits(model.extract_afeats, aud)
                # compute cross and self similarites (B*S, B*S) <- (B*S, D) @ (D, B*S)
                segment_sim_v2a = segment_v_feat @ segment_a_feat.mT
                segment_sim_a2v = segment_a_feat @ segment_v_feat.mT
                segment_sim_v2v = segment_v_feat @ segment_v_feat.mT
                segment_sim_a2a = segment_a_feat @ segment_a_feat.mT
        # show the similarity matrix on the image
        log_sim_matrices(cfg, segment_sim_v2a, segment_sim_a2v, segment_sim_v2v, segment_sim_a2a, phase,
                         iter_step)

    def finish_wandb_logging(self):
        if self.use_wandb:
            wandb.finish()
