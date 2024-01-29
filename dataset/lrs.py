import csv
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

import torch


sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_fixed_offsets, get_video_and_audio, subsample_dataset)


class LRS3(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='VideoReader',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None,
                 to_filter_bad_examples=True,):
        super().__init__()
        self.max_clip_len_sec = 11
        logging.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        split_clip_ids_path = os.path.join(splits_path, f'lrs3_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            vid_folder = Path(vids_dir) / 'pretrain'
            clip_paths = sorted(vid_folder.rglob('*/*.mp4'))
            if to_filter_bad_examples:
                clip_paths = self.filter_bad_examples(clip_paths)
            self.make_split_files(clip_paths)

        # read the ids from a split
        split_clip_ids = sorted(open(split_clip_ids_path).read().splitlines())

        # make paths from the ids
        clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in split_clip_ids]

        if split in self.load_fixed_offsets_on:
            logging.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'lrs3')

        self.dataset = clip_paths
        self.dataset = subsample_dataset(self.dataset, size_ratio, shuffle=split == 'train')

        logging.info(f'{split} has {len(self.dataset)} items')

    def __getitem__(self, index):
        path = self.dataset[index]
        rgb, audio, meta = get_video_and_audio(path, get_meta=True, end_sec=self.max_clip_len_sec)

        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        item = {'video': rgb, 'audio': audio, 'meta': meta, 'path': path, 'targets': {}, 'split': self.split}

        # loading fixed offsets so we could evaluate on the same data each time (valid and test)
        if self.split in self.load_fixed_offsets_on:
            unique_id = path.replace(f'{self.vids_dir}/', '').replace(self.vids_dir, '').replace('.mp4', '')
            offset_params = self.vid2offset_params[unique_id]
            item['targets']['offset_sec'] = offset_params['offset_sec']
            item['targets']['v_start_i_sec'] = offset_params['v_start_i_sec']
            if 'oos_target' in offset_params:
                item['targets']['offset_target'] = {
                    'oos': offset_params['oos_target'], 'offset': item['targets']['offset_sec'],
                }

        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def filter_bad_examples(self, paths):
        bad = set()
        base_path = Path('./data/filtered_examples_lrs')
        lists = [open(p).read().splitlines() for p in sorted(glob(str(base_path / '*.txt')))]
        for s in lists:
            bad = bad.union(s)
        logging.info(f'Number of clips before filtering: {len(paths)}')
        video_ids = [str(i).replace(self.vids_dir, '') for i in paths]
        video_ids = [str(i).replace(f'{self.vids_dir}/', '') for i in video_ids]
        paths = sorted([r for r in video_ids if r not in bad])
        logging.info(f'Number of clips after filtering: {len(paths)}')
        return paths

    def make_split_files(self, paths):
        logging.warning(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')

        # will be splitting using videos, not clips to prevent train-test intersection
        all_vids = sorted(list(set([Path(p).parent.name for p in paths])))
        random.Random(self.seed).shuffle(all_vids)

        # 0.1: splits are 8:1:1
        hold_out_ratio = 0.1
        hold_out_size = int(len(all_vids) * hold_out_ratio)
        test_vids, train_valid_vids = all_vids[:hold_out_size], all_vids[hold_out_size:]
        valid_vids, train_vids = train_valid_vids[:hold_out_size], train_valid_vids[hold_out_size:]

        # making files
        for phase, vids in zip(['train', 'valid', 'test'], [train_vids, valid_vids, test_vids]):
            with open(os.path.join(self.splits_path, f'lrs3_{phase}.txt'), 'w') as wfile:
                for path in paths:
                    vid_name = Path(path).parent.name
                    # just in the case I forgot the trailing '/' in the path
                    unique_id = path.replace(f'{self.vids_dir}/', '').replace(self.vids_dir, '') \
                                    .replace('.mp4', '')
                    if vid_name in vids:
                        wfile.write(unique_id + '\n')

    def __len__(self):
        return len(self.dataset)

class LongerLRS3(LRS3):
    '''This class is different to the parent in the extra filtering it does. If the parent was
    making the splits with filtering for shorter than 9 second, this class filters for shorter than 9.5 sec.
    by applying extra filtering.
    '''

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='VideoReader',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None,
                 to_filter_bad_examples=True,):
        # size_ratio is not used here as we are doing it this class (avoiding double subsampling)
        super().__init__(split, vids_dir, transforms, splits_path, seed, load_fixed_offsets_on,
                         vis_load_backend, None, attr_annot_path, max_attr_per_vid,
                         to_filter_bad_examples)
        # does extra filtering
        if to_filter_bad_examples:
            self.dataset = self.filter_bad_examples(self.dataset)
        self.dataset = subsample_dataset(self.dataset, size_ratio, shuffle=split == 'train')
        logging.info(f'{split} has {len(self.dataset)} items')

    def filter_bad_examples(self, paths):
        bad = set()
        base_path = Path('./data/filtered_examples_lrs_extra')
        lists = [open(p).read().splitlines() for p in sorted(glob(str(base_path / '*.txt')))]
        for s in lists:
            bad = bad.union(s)
        logging.info(f'Number of clips before filtering: {len(paths)}')
        video_ids = [str(i).replace(self.vids_dir, '').replace(f'{self.vids_dir}/', '') for i in paths]
        paths = sorted([os.path.join(self.vids_dir, r) for r in video_ids if r not in bad])
        logging.info(f'Number of clips after filtering: {len(paths)}')
        return paths


if __name__ == '__main__':
    from time import time
    from omegaconf import OmegaConf
    import sys
    sys.path.insert(0, '.')  # nopep8
    from scripts.train_utils import get_transforms
    from utils.utils import cfg_sanity_check_and_patch
    cfg = OmegaConf.load('./configs/sparse_sync.yaml')
    cfg.data.vids_path = '/scratch/local/hdd/vi/data/lrs3/h264_uncropped_25fps_256side_16000hz_aac/'
    cfg.data.dataset.params.load_fixed_offsets_on = ['valid', 'test']

    cfg_sanity_check_and_patch(cfg)
    transforms = get_transforms(cfg)

    datasets = {
        'train': LRS3('train', cfg.data.vids_path, transforms['train'], load_fixed_offsets_on=[]),
        'valid': LRS3('valid', cfg.data.vids_path, transforms['test'], load_fixed_offsets_on=[]),
        'test': LRS3('test', cfg.data.vids_path, transforms['test'], load_fixed_offsets_on=[]),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][1]['audio'].shape, datasets['train'][1]['video'].shape)
    print(datasets['valid'][1]['audio'].shape, datasets['valid'][1]['video'].shape)
