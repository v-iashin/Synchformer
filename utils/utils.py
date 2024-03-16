import logging
from hashlib import md5
import difflib
import importlib
import subprocess
from multiprocessing import Pool
from pathlib import Path

import requests
from omegaconf import OmegaConf
from tqdm import tqdm

PARENT_LINK = 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
FNAME2LINK = {
    # S3: Synchability: AudioSet (run 2)
    '24-01-22T20-34-52.pt': f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/24-01-22T20-34-52.pt',
    'cfg-24-01-22T20-34-52.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/cfg-24-01-22T20-34-52.yaml',
    # S2: Synchformer: AudioSet (run 2)
    '24-01-04T16-39-21.pt': f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt',
    'cfg-24-01-04T16-39-21.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml',
    # S2: Synchformer: AudioSet (run 1)
    '23-08-28T11-23-23.pt': f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/23-08-28T11-23-23.pt',
    'cfg-23-08-28T11-23-23.yaml': f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/cfg-23-08-28T11-23-23.yaml',
    # S2: Synchformer: LRS3 (run 2)
    '23-12-23T18-33-57.pt': f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.pt',
    'cfg-23-12-23T18-33-57.yaml': f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/cfg-23-12-23T18-33-57.yaml',
    # S2: Synchformer: VGS (run 2)
    '24-01-02T10-00-53.pt': f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/24-01-02T10-00-53.pt',
    'cfg-24-01-02T10-00-53.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/cfg-24-01-02T10-00-53.yaml',
    # SparseSync: ft VGGSound-Full
    '22-09-21T21-00-52.pt': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/22-09-21T21-00-52.pt',
    'cfg-22-09-21T21-00-52.yaml': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/cfg-22-09-21T21-00-52.yaml',
    # SparseSync: ft VGGSound-Sparse
    '22-07-28T15-49-45.pt': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/22-07-28T15-49-45.pt',
    'cfg-22-07-28T15-49-45.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/cfg-22-07-28T15-49-45.yaml',
    # SparseSync: only pt on LRS3
    '22-07-13T22-25-49.pt': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/22-07-13T22-25-49.pt',
    'cfg-22-07-13T22-25-49.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/cfg-22-07-13T22-25-49.yaml',
    # SparseSync: feature extractors
    'ResNetAudio-22-08-04T09-51-04.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-04T09-51-04.pt',  # 2s
    'ResNetAudio-22-08-03T23-14-49.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-49.pt',  # 3s
    'ResNetAudio-22-08-03T23-14-28.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-28.pt',  # 4s
    'ResNetAudio-22-06-24T08-10-33.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T08-10-33.pt',  # 5s
    'ResNetAudio-22-06-24T17-31-07.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T17-31-07.pt',  # 6s
    'ResNetAudio-22-06-24T23-57-11.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T23-57-11.pt',  # 7s
    'ResNetAudio-22-06-25T04-35-42.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-25T04-35-42.pt',  # 8s
}


def check_if_file_exists_else_download(path, fname2link=FNAME2LINK, chunk_size=1024):
    '''Checks if file exists, if not downloads it from the link to the path'''
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        link = fname2link.get(path.name, None)
        if link is None:
            raise ValueError(f'Cant find the checkpoint file: {path}.',
                             f'Please download it manually and ensure the path exists.')
        with requests.get(fname2link[path.name], stream=True) as r:
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                with open(path, 'wb') as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if 'target' not in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def fix_prefix(prefix):
    if len(prefix) > 0:
        prefix += '_'
    return prefix

def cfg_sanity_check_and_patch(cfg):
    if cfg.action == 'train_avclip':
        assert cfg.model.params.afeat_extractor.params.get('add_global_repr') == \
            cfg.model.params.vfeat_extractor.params.get('add_global_repr'), 'add_global_repr is diff for A&V'
        assert cfg.model.params.afeat_extractor.params.get('max_segments') == \
            cfg.model.params.vfeat_extractor.params.get('max_segments')
        return
    if 'params' in cfg.data.dataset:
        if 'load_fixed_offsets_on_test' in cfg.data.dataset.params:
            if 'load_fixed_offsets_on' not in cfg.data.dataset.params:
                if cfg.data.dataset.params.load_fixed_offsets_on_test:
                    cfg.data.dataset.params.load_fixed_offsets_on = ['val', 'valid', 'test']
                else:
                    cfg.data.dataset.params.load_fixed_offsets_on = []
            # remove the argument from the omegaconf config
            del cfg.data.dataset.params.load_fixed_offsets_on_test

    assert not (cfg.training.resume and cfg.training.finetune), 'it is either funetuning or resuming'
    assert not (cfg.training.resume and cfg.training.run_test_only), 'it is either resuming or testing-only'
    assert not (cfg.training.finetune and cfg.training.run_test_only), 'it is either finetune or testing-only'

    offset_type = cfg.data.get('offset_type', None)
    if offset_type is not None:
        if 'grid' in offset_type and 'loss_fn' in cfg.training:
            assert 'mse' not in cfg.training.loss_fn[1], f'to class but loss: {cfg.training.loss_fn[1]}'
        elif 'uniform' in offset_type:
            assert 'cross_entropy' not in cfg.training.loss_fn[1], f'reg but loss: {cfg.training.loss_fn[1]}'

    # if cfg.data.get('iter_times', 1) > 1:
    #     assert cfg.data.dataset.params.load_fixed_offsets_on == [], 'iterating on the same data'

    if 'patience' in cfg.training:
        assert cfg.training.patience is not None, f'patience is {cfg.training.patience}'

    assert cfg.logging.get('log_max_items', 1) > 0, 'log_max_items should be > 0'

    if 'probe' in cfg:
        # assert all(n in ['off_head', 'global_transformer'] for n in cfg.probe.setting), \
        #     f'Not implemented for: {cfg.probe}'
        assert cfg.probe.setting in ['off_head', 'global_transformer', 'full', 'audio_fe', 'visual_fe'], \
            f'Not implemented for: {cfg.probe}'

    if cfg.training.resume or cfg.training.run_test_only or cfg.training.finetune:
        assert Path(cfg.ckpt_path).exists(), cfg.ckpt_path
        if cfg.training.resume or cfg.training.run_test_only:
            # the Feat extractor ckpts are already in the model ckpt
            cfg.model.params.afeat_extractor.params.ckpt_path = None
            cfg.model.params.vfeat_extractor.params.ckpt_path = None
    if cfg.training.resume:
        assert Path(cfg.logging.logdir, cfg.start_time).exists(), Path(cfg.logging.logdir, cfg.start_time)
    afeat_extractor = cfg.model.params.afeat_extractor.target
    vfeat_extractor = cfg.model.params.vfeat_extractor.target
    if afeat_extractor.endswith('ResNet18AudioFeatures') and vfeat_extractor.endswith('S3DVisualFeatures'):
        assert cfg.logging.vis_segment_sim is False, 'logger.vizualize_segment_sim mults pre-proj features'

def get_fixed_off_fname(data_transforms, split):
    '''data_transforms: should be transforms.Compose'''
    for t in data_transforms.transforms:
        if hasattr(t, 'class_grid'):
            min_off = t.class_grid.min().item()
            max_off = t.class_grid.max().item()
            grid_size = len(t.class_grid)
            crop_len_sec = int(t.crop_len_sec) if t.crop_len_sec == int(t.crop_len_sec) else t.crop_len_sec
            return f'{split}_size{grid_size}_crop{crop_len_sec}_min{min_off:.2f}_max{max_off:.2f}.csv'
        elif hasattr(t, 'offset_type'):
            min_off = t.off_dist.low.item()
            max_off = t.off_dist.high.item()
            crop_len_sec = int(t.crop_len_sec) if t.crop_len_sec == int(t.crop_len_sec) else t.crop_len_sec
            return f'{split}_unifbin_crop{crop_len_sec}_min{min_off:.2f}_max{max_off:.2f}.csv'


def disable_print_if_not_master(is_master):
    """
    from: https://github.com/pytorch/vision/blob/main/references/video_classification/utils.py
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def apply_fn_for_loop(fn, lst, *args):
    for path in tqdm(lst):
        fn(path, *args)


def apply_fn_in_parallel(fn, lst, num_workers):
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(fn, lst), total=len(lst)))


def show_cfg_diffs(a, b, save_diff_path=None):
    a = OmegaConf.to_yaml(a).split('\n')
    b = OmegaConf.to_yaml(b).split('\n')

    if save_diff_path is None:
        for line in difflib.unified_diff(a, b, fromfile='old', tofile='new', lineterm=''):
            print(line)
    else:
        with open(save_diff_path, 'w') as wfile:
            for line in difflib.unified_diff(a, b, fromfile='old', tofile='new', lineterm=''):
                wfile.write(f'{line}\n')
    logging.info(f'Config diff (current vs fine-tuning ckpt) saved to {save_diff_path}')

def get_param_by_name_from_transform_cfg(cfg, name, param):
    for t in cfg:
        if t.target == name or t.target.endswith(name):
            return t.params.get(param)
    raise ValueError(f'No transform with name {name} found in {cfg}.')

def get_transform_instance_from_compose(transforms_compose, name):
    for t in transforms_compose.transforms:
        if t.__class__.__name__ == name:
            return t
    raise ValueError(f'No transform with name {name} found in {transforms_compose}.')

def get_md5sum(path):
    hash_md5 = md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096*8), b''):
            hash_md5.update(chunk)
    md5sum = hash_md5.hexdigest()
    return md5sum
