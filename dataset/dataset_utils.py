import csv
import os
import random
from pathlib import Path
from glob import glob
import shutil
import logging

import torchaudio
import torchvision

from utils.utils import get_fixed_off_fname


def get_fixed_offsets(transforms, split, splits_path, dataset_name):
    '''dataset_name: `vggsound` or `lrs3`'''
    logging.info(f'Using fixed offset for {split}')
    vid2offset_params = {}
    fixed_offset_fname = get_fixed_off_fname(transforms, split)
    if fixed_offset_fname is None:
        raise ValueError('Cant find fixed offsets for given params. Perhaps you need to make it first?')
    fixed_offset_path = os.path.join(splits_path, f'fixed_offsets_{dataset_name}', fixed_offset_fname)
    fixed_offset_paths = sorted(glob(fixed_offset_path.replace(split, '*')))
    assert len(fixed_offset_paths) > 0, f'Perhaps: {fixed_offset_path} does not exist. Make fixed offsets'

    for fix_off_path in fixed_offset_paths:
        reader = csv.reader(open(fix_off_path))
        # k700_2020 has no header, and also `vstart` comes before `offset_sec`
        if dataset_name == 'k700_2020':
            header = ['path', 'vstart_sec', 'offset_sec', 'oos_target']
        else:
            header = next(reader)
        for line in reader:
            data = dict()
            for f, value in zip(header, line):
                if f == 'path':
                    v = value
                elif f == 'offset_sec':
                    data[f] = float(value)
                elif f in ['vstart_sec', 'v_start_sec']:
                    f = 'v_start_i_sec'
                    data[f] = float(value)
                elif f == 'oos_target':
                    data[f] = int(value)
                else:
                    data[f] = value
            # assert v not in vid2offset_params, 'otherwise, offs from other splits will override each other'

            # even if we have multiple splits (val=test), we want to make sure that the offsets are the same
            if v in vid2offset_params:
                assert all([vid2offset_params[v][k] == data[k] for k in data]), f'{v} isnt unique and vary'

            vid2offset_params[v] = data
    return vid2offset_params


def maybe_cache_file(path: os.PathLike):
    '''Motivation: if every job reads from a shared disk it`ll get very slow, consider an image can
    be 2MB, then with batch size 32, 16 workers in dataloader you`re already requesting 1GB!! -
    imagine this for all users and all jobs simultaneously.'''
    # checking if we are on cluster, not on a local machine
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        # a bit ugly but we need not just fname to be appended to `cache_dir` but parent folders,
        # otherwise the same fnames in multiple folders will create a bug (the same input for multiple paths)
        cache_path = os.path.join(cache_dir, Path(path).relative_to('/'))
        if not os.path.exists(cache_path):
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path


def get_video_and_audio(path, get_meta=False, start_sec=0, end_sec=None):
    orig_path = path
    path = maybe_cache_file(path)
    # (Tv, 3, H, W) [0, 255, uint8]; (Ca, Ta)
    rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, 'sec', output_format='TCHW')
    assert meta['video_fps'], f'No video fps for {orig_path}'
    # (Ta) <- (Ca, Ta)
    audio = audio.mean(dim=0)
    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    meta = {'video': {'fps': [meta['video_fps']]}, 'audio': {'framerate': [meta['audio_fps']]}, }
    return rgb, audio, meta


def get_audio_stream(path, get_meta=False):
    '''Used only in feature extractor training'''
    path = str(Path(path).with_suffix('.wav'))
    path = maybe_cache_file(path)
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if get_meta:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        meta = {'audio': {'duration': [duration], 'framerate': [info.sample_rate]}}
        return waveform, meta
    else:
        return waveform

def subsample_dataset(dataset: list, size_ratio: float, shuffle: bool = False):
    if size_ratio is not None and 0.0 < size_ratio < 1.0:
        logging.info(f'Subsampling dataset to {size_ratio}')
        # shuffling is important only during subsampling (sometimes paths are sorted by class)
        if shuffle:
            random.shuffle(dataset)
        cut_off = int(len(dataset) * size_ratio)
        # making sure that we have at least one example
        dataset = dataset[:max(1, cut_off)]
        logging.info(f'Subsampled dataset to {size_ratio} (size: {len(dataset)})')
    return dataset
