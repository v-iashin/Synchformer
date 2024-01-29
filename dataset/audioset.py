import csv
import logging
import random
import sys
from glob import glob
from pathlib import Path

import torch

sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import (get_fixed_offsets, get_video_and_audio)


class AudioSet(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 to_filter_bad_examples=True,
                 splits_path='./data',
                 meta_path='./data/audioset.csv',
                 seed=1337,
                 load_fixed_offsets_on=['valid' 'test'],
                 vis_load_backend='read_video',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None):
        super().__init__()
        self.max_clip_len_sec = None
        self.split = split
        self.vids_dir = Path(vids_dir)
        self.transforms = transforms
        self.to_filter_bad_examples = to_filter_bad_examples
        self.splits_path = Path(splits_path)
        self.meta_path = Path(meta_path)
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio

        self.split2short = {'train': 'unbalanced', 'valid': 'balanced', 'test': 'eval'}
        short2long = {'unbalanced': 'unbalanced_train_segments',
                      'balanced': 'balanced_train_segments',
                      'eval': 'eval_segments'}

        # read meta
        split_meta = []
        for shortdir_vid, start, end, targets, phase in csv.reader(open(meta_path), quotechar='"'):
            if shortdir_vid.startswith(self.split2short[split]):
                # shortdir_vid 'unbalanced/NFap9qgsI_s' -> 'unbalanced_train_segments/NFap9qgsI_s'
                shortdir, vid = shortdir_vid.split('/')
                longdir_vid = '/'.join([short2long[shortdir], vid])
                split_meta.append([longdir_vid, float(start), float(end), targets, phase])

        # filter "bad" examples
        if to_filter_bad_examples:
            split_meta = self.filter_bad_examples(split_meta)

        # label maps
        self.label2target = {l: int(t) for t, _, l in csv.reader(open(self.splits_path / 'audioset_labels.csv'))}
        self.target2label = {t: l for l, t in self.label2target.items()}
        self.video2target = {key: list(map(int, targets.split(','))) for key, _, _, targets, _ in split_meta}

        clip_paths = [self.vids_dir / f'{k}_{int(s*1000)}_{int(e*1000)}.mp4' for k, s, e, t, p in split_meta]
        clip_paths = sorted(clip_paths)

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if transforms is not None and split in load_fixed_offsets_on:
            logging.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'audioset')

        self.dataset = clip_paths
        if size_ratio is not None and 0.0 < size_ratio < 1.0:
            cut_off = int(len(self.dataset) * size_ratio)
            random.seed(seed)
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:cut_off]

        logging.info(f'{split} has {len(self.dataset)} items')

    def filter_bad_examples(self, audioset_meta):
        bad = set()
        base_path = Path('./data/filtered_examples_audioset')
        files = sorted(glob(str(base_path / '*.txt')))
        lists = [open(p).read().splitlines() for p in files]
        logging.info(f'Filtering for {files}')
        for s in lists:
            bad = bad.union(s)
        # the ugly string converts '---g-f_I2yQ', '1' into `---g-f_I2yQ_1000_11000`
        audioset_meta = [r for r in audioset_meta if f'{r[0]}_{int(r[1]*1000)}_{int(r[2]*1000)}' not in bad]
        return audioset_meta

    def __getitem__(self, index):
        path = self.dataset[index]
        rgb, audio, meta = self.load_media(path)
        item = self.make_datapoint(path, rgb, audio, meta)
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def make_datapoint(self, path, rgb, audio, meta):
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        # TODO: since audioset is annotated by tagging, targets have multiple elemenets: default collate fails
        # targets = self.video2target[f'{Path(path).parent.stem}/{Path(path).stem[:11]}']
        item = {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            # 'targets': {'audioset_target': [targets], 'audioset_label': [self.target2label[t] for t in targets]},
            'targets': {},
            'split': self.split,
        }

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.transforms is not None and self.split in self.load_fixed_offsets_on:
            key = f'{self.split2short[self.split]}/{Path(path).stem}'
            item['targets']['offset_sec'] = self.vid2offset_params[key]['offset_sec']
            item['targets']['v_start_i_sec'] = self.vid2offset_params[key]['v_start_i_sec']

        return item

    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(path, get_meta=True, end_sec=self.max_clip_len_sec)
        return rgb, audio, meta

    def __len__(self):
        return len(self.dataset)

class AudioSetBalanced737k(AudioSet):

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True, splits_path='./data',
                 # here
                 meta_path='./data/audioset_balanced_737k.csv',
                 seed=1337, load_fixed_offsets_on=['valid', 'test'], vis_load_backend='read_video', size_ratio=None,
                 attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path, meta_path,
                         seed, load_fixed_offsets_on, vis_load_backend, size_ratio)

class AudioSetBalanced540k(AudioSet):
    ''' MBT's balanced 500k (from unbalanced part) + 20k from balaced part + 20k from eval part '''

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True, splits_path='./data',
                 # here
                 meta_path='./data/audioset_balanced_540k.csv',
                 seed=1337, load_fixed_offsets_on=['valid', 'test'], vis_load_backend='read_video', size_ratio=None,
                 attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path, meta_path,
                         seed, load_fixed_offsets_on, vis_load_backend, size_ratio)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from scripts.train_utils import get_transforms
    from utils.utils import cfg_sanity_check_and_patch
    cfg = OmegaConf.load('./configs/sparse_sync.yaml')
    vis_load_backend = 'read_video'

    transforms = get_transforms(cfg)

    # vids_path = 'PLACEHOLDER'
    vids_path = '/scratch/project_2000936/vladimir/data/audioset/h264_video_25fps_256side_16000hz_aac'
    load_fixed_offsets_on = []

    cfg.data.dataset.params.size_ratio = 0.1

    cfg_sanity_check_and_patch(cfg)

    datasets = {
        'train': AudioSet('train', vids_path, transforms['train'], vis_load_backend=vis_load_backend,
                          to_filter_bad_examples=True, size_ratio=cfg.data.dataset.params.size_ratio,
                          load_fixed_offsets_on=load_fixed_offsets_on),
        'valid': AudioSet('valid', vids_path, transforms['test'], vis_load_backend=vis_load_backend,
                          to_filter_bad_examples=True, load_fixed_offsets_on=load_fixed_offsets_on),
        'test': AudioSet('test', vids_path, transforms['test'], vis_load_backend=vis_load_backend,
                         to_filter_bad_examples=True, load_fixed_offsets_on=load_fixed_offsets_on),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])

    for i in range(300, 1000):
        datasets['train'][i]['path']
        print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
        print(datasets['train'][0]['meta'])

    datasets = {
        'train': AudioSetBalanced737k('train', vids_path, transforms['train'], vis_load_backend=vis_load_backend,
                          to_filter_bad_examples=True, size_ratio=cfg.data.dataset.params.size_ratio,
                          load_fixed_offsets_on=load_fixed_offsets_on),
        'valid': AudioSetBalanced737k('valid', vids_path, transforms['test'], vis_load_backend=vis_load_backend,
                          to_filter_bad_examples=True, load_fixed_offsets_on=load_fixed_offsets_on),
        'test': AudioSetBalanced737k('test', vids_path, transforms['test'], vis_load_backend=vis_load_backend,
                         to_filter_bad_examples=True, load_fixed_offsets_on=load_fixed_offsets_on),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])

    for i in range(300, 1000):
        datasets['train'][i]['path']
        print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
        print(datasets['train'][0]['meta'])
