import logging
import math
import random
from typing import Tuple
import torch
import torchvision
import torchaudio
import numpy as np
import einops


def sec2frames(sec, fps):
    return int(sec * fps)

def frames2sec(frames, fps):
    return frames / fps


class EqualifyFromRight(torch.nn.Module):

    def __init__(self, clip_max_len_sec=10):
        '''
        Takes the dataset item and makes sure more streams are of an equal size in terms of fps.
        It, however, assumes that the signal is synched and trims the ending parts ('from the right').
        '''
        super().__init__()
        self.clip_max_len_sec = clip_max_len_sec

    def forward(self, item):
        '''
        `item`: {'video': (Tv, C, H, W), 'audio': (Ta,),
                 'meta': {
                     'audio': {'framerate': [float], 'duration': [float]}
                     'video': {'fps': [float], 'duration': [float]}}
        '''
        a_fps = item['meta']['audio']['framerate'][0]
        v_fps = item['meta']['video']['fps'][0]

        Ta = item['audio'].shape[0]
        Tv, C, H, W = item['video'].shape

        a_len_secs = Ta / a_fps
        v_len_secs = Tv / v_fps
        min_len = min(self.clip_max_len_sec, a_len_secs, v_len_secs)

        a_frames_per_v_frame = a_fps // v_fps
        v_len_frames = int(v_fps * min_len)
        a_len_frames = int(a_frames_per_v_frame * v_len_frames)
        # print(a_len_frames, v_len_frames)

        assert a_len_frames <= Ta and v_len_frames <= Tv

        item['audio'] = item['audio'][:a_len_frames]
        item['video'] = item['video'][:v_len_frames, :, :, :]

        return item


class RGBSpatialCrop(torch.nn.Module):

    def __init__(self, input_size, is_random):
        super().__init__()
        assert input_size is not None, f'smaller_input_size is `{input_size}`'
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.is_random = is_random

    @staticmethod
    def get_random_crop_sides(vid, output_size):
        '''Slice parameters for random crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def get_center_crop_sides(vid, output_size):
        '''Slice parameters for center crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def forward(self, item):
        # (Tv, C, H, W)
        vid = item['video']
        if self.is_random:
            i, j, h, w = self.get_random_crop_sides(vid, self.input_size)
        else:
            i, j, h, w = self.get_center_crop_sides(vid, self.input_size)
        item['video'] = vid[..., i:(i + h), j:(j + w)]
        return item

class Resize(torchvision.transforms.Resize):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        return item


class RGBSpatialCropSometimesUpscale(torch.nn.Module):
    '''This (randomly) crops the input video and with prob `sometimes_p` this crop is smaller but upscaled
    to `target_input_size`'''

    def __init__(self, sometimes_p, target_input_size, is_random, smaller_input_size=None):
        super().__init__()
        self.sometimes_p = sometimes_p
        self.do_sometimes_upscale = sometimes_p is not None and sometimes_p > 0

        self.crop_only = RGBSpatialCrop(target_input_size, is_random)

        if self.do_sometimes_upscale:
            self.crop_further_and_upscale = torchvision.transforms.Compose([
                RGBSpatialCrop(smaller_input_size, is_random),
                Resize(target_input_size, antialias=None),
            ])

    def forward(self, item):
        assert len(item['video'].shape) == 4, \
            f"{item['video'].shape}: if it is applied after GenerateMultipleClips," \
            "augs should be applied to each clip separately, not to the whole video array. " \
            "Otherwise, ignore this warning (comment it)."
        if self.do_sometimes_upscale and self.sometimes_p > torch.rand(1):
            return self.crop_further_and_upscale(item)
        else:
            return self.crop_only(item)


class RandomApplyColorDistortion(torch.nn.Module):

    def __init__(self, p_gray_scale=0., p_color_jitter=0., s=1.) -> None:
        super().__init__()
        self.p_gray_scale = p_gray_scale
        self.p_color_jitter = p_color_jitter
        self.s = s
        assert 0 <= self.p_color_jitter <= 1 and 0 <= self.p_gray_scale <= 1, (p_color_jitter, p_gray_scale)
        # SimCLR params
        color_jitter = torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rand_color_jitter = torchvision.transforms.RandomApply([color_jitter], p_color_jitter)
        rand_gray = torchvision.transforms.RandomGrayscale(p_gray_scale)
        self.transforms = torchvision.transforms.Compose([rand_color_jitter, rand_gray])

    def apply_to_single_clip(self, clip):
        return self.transforms(clip)

    def apply_to_each_clip(self, clips):
        for i, clip in enumerate(clips):
            clips[i] = self.apply_to_single_clip(clip)
        return clips

    def forward(self, item):
        has_batch_dim = len(item['video'].shape) == 5
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['video'] = fn(item['video'])
        return item


class ApplyColorJitterFrameWise(torch.nn.Module):

    def __init__(self, s=1.) -> None:
        super().__init__()
        self.s = s
        # SimCLR params
        self.transform = torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)

    def apply_to_single_clip(self, clip):
        for i, frame in enumerate(clip):
            clip[i] = self.transform(frame)
        return clip

    def apply_to_each_clip(self, clips):
        for i, clip in enumerate(clips):
            clips[i] = self.apply_to_single_clip(clip)
        return clips

    def forward(self, item):
        has_batch_dim = len(item['video'].shape) == 5
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['video'] = fn(item['video'])
        return item


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_to_single_clip(self, clip):
        return super().forward(clip)

    def apply_to_each_clip(self, clips):
        for i, clip in enumerate(clips):
            clips[i] = self.apply_to_single_clip(clip)
        return clips

    def forward(self, item):
        has_batch_dim = len(item['video'].shape) == 5
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['video'] = fn(item['video'])
        return item


def make_class_grid(leftmost_val, rightmost_val, grid_size, add_extreme_offset: bool = False,
                    seg_size_vframes: int = None, nseg: int = None, step_size_seg: float = None,
                    vfps: float = None):
    assert grid_size >= 3, f'grid_size: {grid_size} doesnot make sense. If =2 -> (-1,1); =1 -> (-1); =0 -> ()'
    grid = torch.from_numpy(np.linspace(leftmost_val, rightmost_val, grid_size)).float()
    if add_extreme_offset:
        assert all([seg_size_vframes, nseg, step_size_seg]), f'{seg_size_vframes} {nseg} {step_size_seg}'
        seg_size_sec = seg_size_vframes / vfps
        trim_size_in_seg = nseg - (1 - step_size_seg) * (nseg - 1)
        extreme_value = trim_size_in_seg * seg_size_sec
        grid = torch.cat([grid, torch.tensor([extreme_value])])  # adding extreme offset to the class grid
    return grid


def quantize_offset(grid: torch.Tensor, off_sec: float) -> Tuple[float, int]:
    '''Takes in the offset in seconds and snaps it onto the closest grid element.
    Returns the grid value and its index.'''
    closest_grid_el = (grid - off_sec).abs().argmin()
    return grid[closest_grid_el], closest_grid_el

def apply_a_jitter(a_start_i, a_len_frames, a_crop_len_frames, a_fps, max_a_jitter_sec):
    max_a_start_i = a_len_frames - a_crop_len_frames
    max_a_jitter_i = sec2frames(max_a_jitter_sec, a_fps)
    max_a_jitter_i_left = min(a_start_i, max_a_jitter_i)
    max_a_jitter_i_right = min(max_a_start_i - a_start_i, max_a_jitter_i)
    # jitter is U[left, right]
    a_jitter_i = random.randint(-max_a_jitter_i_left, max_a_jitter_i_right)
    # apply jitter
    a_start_i = a_start_i + a_jitter_i
    # making sure that any value from `a_start_i + U[left, right]` will be inside of [0, len-crop] region
    assert 0 <= a_start_i <= max_a_start_i, f'{a_jitter_i} {max_a_jitter_i_left} {max_a_jitter_i_right} {max_a_start_i}'
    return a_start_i, a_jitter_i


class TemporalCropAndOffset(torch.nn.Module):

    def __init__(self, crop_len_sec: float, max_off_sec: float, offset_type='grid', do_offset: bool = True,
                 grid_size: int = None, max_wiggle_sec: float = None, add_doubt_cls: bool = False,
                 segment_size_vframes: int = None, n_segments: int = None, step_size_seg: float = None,
                 vfps: float = None, prob_oos: float = None):
        super().__init__()
        self.crop_len_sec = crop_len_sec
        self.do_offset = do_offset
        self.grid_size = grid_size
        self.offset_type = offset_type
        self.max_off_sec = max_off_sec
        self.max_a_jitter_sec = max_wiggle_sec
        if do_offset:
            if offset_type == 'grid':
                self.class_grid = make_class_grid(-max_off_sec, max_off_sec, grid_size, add_doubt_cls,
                                                  segment_size_vframes, n_segments, step_size_seg, vfps)
                logging.info(f'Offsets class grid: {self.class_grid}')
                if self.max_a_jitter_sec is not None:
                    assert (max_wiggle_sec-1e-6) <= ((self.class_grid[1] - self.class_grid[0]) / 2), f'{self.class_grid}'
            elif offset_type == 'uniform':
                self.off_dist = torch.distributions.uniform.Uniform(-max_off_sec, max_off_sec)
                logging.info(f'Offset uniform distribution: {self.off_dist}')
            elif offset_type == 'uniform_binary':
                self.itu_t_range = (-0.125, 0.045)
                self.prob_oos = prob_oos
                self.ins_dist = torch.distributions.uniform.Uniform(self.itu_t_range[0], self.itu_t_range[1])
                self.off_dist = torch.distributions.uniform.Uniform(-max_off_sec, max_off_sec)
            else:
                raise NotImplementedError(f'Unknown offset type: {offset_type}')

    def forward(self, item):
        vid = item['video']
        aud = item['audio']
        v_len_frames, C, H, W = vid.shape
        a_len_frames = aud.shape[0]

        v_fps = int(item['meta']['video']['fps'][0])
        a_fps = int(item['meta']['audio']['framerate'][0])

        v_crop_len_frames = sec2frames(self.crop_len_sec, v_fps)
        a_crop_len_frames = sec2frames(self.crop_len_sec, a_fps)

        if self.do_offset:
            # trying to get the offset parameters (for instance during valid and test we have fixed offsets)
            offset_sec = item['targets'].get('offset_sec', None)
            v_start_i_sec = item['targets'].get('v_start_i_sec', None)
            if 'offset_target' in item['targets']:
                is_oos = item['targets']['offset_target'].get('oos', None)
            # train-time
            if offset_sec is None and v_start_i_sec is None:
                # aud starts `offset_sec` earlier than it should; aud has what will be shown after offset_sec
                if self.offset_type == 'grid':
                    offset_sec = random.choice(self.class_grid.tolist())
                elif self.offset_type == 'uniform':
                    offset_sec = self.off_dist.sample().item()
                elif self.offset_type == 'uniform_binary':
                    # in-sync: Uniform(-0.125, 0.045)
                    # out-of-sync: Uniform(-5.5, 5.5) and resampled until not in Uniform(-0.125, 0.045)
                    # first, we sample if the offset is out-of-sync with prob_oss
                    is_oos = (torch.rand(1) < self.prob_oos).item()
                    if is_oos:
                        # second, we sample the offset itself (if in in-sync range, trying again)
                        offset_sec = self.off_dist.sample().item()
                        while self.itu_t_range[0] <= offset_sec <= self.itu_t_range[1]:
                            offset_sec = self.off_dist.sample().item()
                    else:
                        offset_sec = self.ins_dist.sample().item()
                offset_sec = round(offset_sec, 2)
                v_start_max_sec = frames2sec(v_len_frames - v_crop_len_frames, v_fps)
                assert v_start_max_sec > 0, f'{v_len_frames} {v_crop_len_frames} {v_fps} @ {item["path"]}'
                # `v_start_sec` IS NOT rounded to the fps grid
                v_start_sec = random.uniform(max(0, -offset_sec), min(v_start_max_sec, v_start_max_sec-offset_sec))
                assert 0 <= v_start_sec <= v_start_max_sec, f'{v_start_sec} {v_start_max_sec} {item["path"]}'
                v_start_i = sec2frames(v_start_sec, v_fps)
                # `v_start_i_sec` IS rounded to the fps grid
                v_start_i_sec = frames2sec(v_start_i, v_fps)
            else:
                offset_sec = round(offset_sec, 2)
                v_start_i = sec2frames(v_start_i_sec, v_fps)
            v_end_i = v_start_i + v_crop_len_frames
            # `a_start_i` depends on the rounded value `v_start_i_sec`, otherwise
            # (v_start_sec) we have ±0.1 jittering
            a_start_i = sec2frames(v_start_i_sec + offset_sec, a_fps)
        else:
            offset_sec = 0.0
            is_random_crop = item['split'] == 'train'
            v_start_i, v_end_i = self.get_crop_idx(v_len_frames, v_crop_len_frames, is_random=is_random_crop)
            v_start_i_sec = frames2sec(v_start_i, v_fps)
            a_start_i = sec2frames(v_start_i_sec, a_fps)

        # sometimes due to the rounding error e.g. v_start_sec = 1.505 but sec2frames(1.505, 25) = 1.48
        # given offset is -1.5, the a_start_i will be a small negative value. (likely a_fps * 1/v_fps * 0.5)
        if a_start_i < 0:
            how_much_out = a_start_i
            logging.info(f'a_start_i is negative ({how_much_out}) at {item["path"]}')
            if abs(how_much_out) <= a_fps / v_fps:
                logging.info('fixing it')
                a_start_i += abs(how_much_out)
            else:
                raise Exception(f'{how_much_out} {item["path"]}')

        if self.max_a_jitter_sec is not None and self.max_a_jitter_sec > 0:
            a_start_i, a_jitter_i = apply_a_jitter(a_start_i, a_len_frames, a_crop_len_frames, a_fps,
                                                   self.max_a_jitter_sec)
            item['meta']['a_jitter_i'] = a_jitter_i

        a_end_i = a_start_i + a_crop_len_frames

        assert v_start_i < v_end_i and a_start_i < a_end_i
        assert aud.shape[0] >= a_end_i, f'{aud.shape} {a_end_i} {item["path"]}'
        assert vid.shape[0] >= v_end_i, f'{vid.shape} {v_end_i} {item["path"]}'

        vid, aud = vid[v_start_i:v_end_i, :, :, :], aud[a_start_i:a_end_i]

        item['video'] = vid
        item['audio'] = aud

        assert item['video'].shape[0] == v_fps * self.crop_len_sec, f'{item["video"].shape} {item["path"]}'
        assert item['audio'].shape[0] == a_fps * self.crop_len_sec, f'{item["audio"].shape} {item["path"]}'

        # caching parameters
        if self.do_offset:
            if self.offset_type == 'grid':
                offset_label, offset_target = quantize_offset(self.class_grid, offset_sec)
            elif self.offset_type == 'uniform':
                offset_label, offset_target = offset_sec, offset_sec
            elif self.offset_type == 'uniform_binary':
                offset_label, offset_target = offset_sec, {'oos': is_oos, 'offset': offset_sec}
            item['targets']['offset_sec'] = offset_sec
            item['targets']['v_start_i_sec'] = v_start_i_sec
            item['targets']['offset_label'] = offset_label
            # assert 'offset_target' not in item['targets'], f'{item["targets"]}. What passed it there?'
            item['targets']['offset_target'] = offset_target

        return item

    def get_crop_idx(self, len_frames: int, crop_len_frames: int, is_random=True):
        if len_frames == crop_len_frames:
            return 0, len_frames
        if is_random:
            left_i = random.randint(0, len_frames - crop_len_frames)
        else:
            left_i = int(round((len_frames - crop_len_frames) / 2.))
        return left_i, left_i+crop_len_frames


class GenerateMultipleSegments(torch.nn.Module):
    '''
    Given an item with video and audio, generates a batch of `n_segments` segments
    of length `segment_size_vframes` (if None, the max number of segments will be made).
    If `is_start_random` is True, the starting position of the 1st segment will be random but respecting
    n_segments.
    `audio_jitter_sec` is the amount of audio offset in seconds.
    '''

    def __init__(self, segment_size_vframes: int, n_segments: int = None, is_start_random: bool = False,
                 audio_jitter_sec: float = 0., step_size_seg: float = 1):
        super().__init__()
        self.segment_size_vframes = segment_size_vframes
        self.n_segments = n_segments
        self.is_start_random = is_start_random
        self.audio_jitter_sec = audio_jitter_sec
        self.step_size_seg = step_size_seg
        logging.info(f'Segment step size: {self.step_size_seg}')

    def forward(self, item):
        v_len_frames, C, H, W = item['video'].shape
        a_len_frames = item['audio'].shape[0]

        v_fps = int(item['meta']['video']['fps'][0])
        a_fps = int(item['meta']['audio']['framerate'][0])

        ## Determining the number of segments
        # segment size
        segment_size_vframes = self.segment_size_vframes
        segment_size_aframes = sec2frames(frames2sec(self.segment_size_vframes, v_fps), a_fps)
        # step size (stride)
        stride_vframes = int(self.step_size_seg * segment_size_vframes)
        stride_aframes = int(self.step_size_seg * segment_size_aframes)
        # calculating the number of segments. (W - F + 2P) / S + 1
        n_segments_max_v = math.floor((v_len_frames - segment_size_vframes) / stride_vframes) + 1
        n_segments_max_a = math.floor((a_len_frames - segment_size_aframes) / stride_aframes) + 1
        # making sure audio and video can accommodate the same number of segments
        n_segments_max = min(n_segments_max_v, n_segments_max_a)
        n_segments = n_segments_max if self.n_segments is None else self.n_segments

        assert n_segments <= n_segments_max, \
            f'cant make {n_segments} segs of len {self.segment_size_vframes} in a vid ' \
            f'of len {v_len_frames} for {item["path"]}'

        # (n_segments, 2) each
        v_ranges, a_ranges = self.get_sequential_seg_ranges(v_len_frames, a_len_frames, v_fps, a_fps,
                                                            n_segments, segment_size_aframes)

        # segmenting original streams (n_segments, segment_size_frames, C, H, W)
        item['video'] = torch.stack([item['video'][s:e] for s, e in v_ranges], dim=0)
        item['audio'] = torch.stack([item['audio'][s:e] for s, e in a_ranges], dim=0)
        return item

    def get_sequential_seg_ranges(self, v_len_frames, a_len_frames, v_fps, a_fps, n_seg, seg_size_aframes):
        # if is_start_random is True, the starting position of the 1st segment will
        # be random but respecting n_segments like so: "-CCCCCCCC---" (maybe with fixed overlap),
        # else the segments are taken from the middle of the video respecting n_segments: "--CCCCCCCC--"

        seg_size_vframes = self.segment_size_vframes  # for brevity

        # calculating the step size in frames
        step_size_vframes = int(self.step_size_seg * seg_size_vframes)
        step_size_aframes = int(self.step_size_seg * seg_size_aframes)

        # calculating the length of the sequence of segments (and in frames)
        seg_seq_len = n_seg * self.step_size_seg + (1 - self.step_size_seg)
        vframes_seg_seq_len = int(seg_seq_len * seg_size_vframes)
        aframes_seg_seq_len = int(seg_seq_len * seg_size_aframes)

        # doing temporal crop
        max_v_start_i = v_len_frames - vframes_seg_seq_len
        if self.is_start_random:
            v_start_i = random.randint(0, max_v_start_i)
        else:
            v_start_i = max_v_start_i // 2
        a_start_i = sec2frames(frames2sec(v_start_i, v_fps), a_fps)  # vid frames -> seconds -> aud frames

        # make segments starts
        v_start_seg_i = torch.tensor([v_start_i + i * step_size_vframes for i in range(n_seg)]).int()
        a_start_seg_i = torch.tensor([a_start_i + i * step_size_aframes for i in range(n_seg)]).int()

        # apply jitter to audio
        if self.audio_jitter_sec > 0:
            jitter_aframes = sec2frames(self.audio_jitter_sec, a_fps)
            # making sure after applying jitter, the audio is still within the audio boundaries
            jitter_aframes = min(jitter_aframes, a_start_i, a_len_frames-a_start_i-aframes_seg_seq_len)
            a_start_seg_i += random.randint(-jitter_aframes, jitter_aframes)  # applying jitter to segments

        # make segment ends
        v_ends_seg_i = v_start_seg_i + seg_size_vframes
        a_ends_seg_i = a_start_seg_i + seg_size_aframes  # using the adjusted a_start_seg_i (with jitter)

        # make ranges
        v_ranges = torch.stack([v_start_seg_i, v_ends_seg_i], dim=1)
        a_ranges = torch.stack([a_start_seg_i, a_ends_seg_i], dim=1)
        assert (a_ranges >= 0).all() and (a_ranges <= a_len_frames).all(), f'{a_ranges} out of {a_len_frames}'
        assert (v_ranges <= v_len_frames).all(), f'{v_ranges} out of {v_len_frames}'
        return v_ranges, a_ranges


class TemporalCropAndOffsetForSyncabilityTraining(torch.nn.Module):

    def __init__(self, max_off_sec: float, do_offset: bool = True,
                 grid_size: int = None, max_wiggle_sec: float = None,
                 segment_size_vframes: int = None, n_segments: int = None, step_size_seg: float = None,
                 vfps: float = None):
        super().__init__()
        seg_size_sec = segment_size_vframes / vfps
        trim_size_in_seg = n_segments - (1 - step_size_seg) * (n_segments - 1)
        self.crop_len_sec = round(trim_size_in_seg * seg_size_sec, 2)
        logging.info(f'Crop len: {self.crop_len_sec}')
        self.do_offset = do_offset
        self.grid_size = grid_size
        self.max_off_sec = max_off_sec
        self.max_a_jitter_sec = max_wiggle_sec
        self.segment_size_vframes = segment_size_vframes
        self.n_segments = n_segments
        self.step_size_seg = step_size_seg
        self.prob_syncable = 0.5
        if do_offset:
            self.class_grid = make_class_grid(-max_off_sec, max_off_sec, grid_size)
            logging.info(f'Offset class grid: {self.class_grid}')
            if self.max_a_jitter_sec is not None:
                assert (max_wiggle_sec-1e-6) <= ((self.class_grid[1] - self.class_grid[0]) / 2), f'{self.class_grid}'

    def forward(self, item):
        vid = item['video']
        aud = item['audio']
        v_len_frames, C, H, W = vid.shape
        a_len_frames = aud.shape[0]

        v_fps = int(item['meta']['video']['fps'][0])
        a_fps = int(item['meta']['audio']['framerate'][0])

        v_crop_len_frames = sec2frames(self.crop_len_sec, v_fps)
        a_crop_len_frames = sec2frames(self.crop_len_sec, a_fps)

        if self.do_offset:
            # trying to get the offset parameters (for instance during valid and test we have fixed offsets)
            offset_sec = item['targets'].get('offset_sec', None)
            v_start_i_sec = item['targets'].get('v_start_i_sec', None)
            # train-time
            if offset_sec is None and v_start_i_sec is None:

                # for the syncability training, we want to have a syncable or non-syncable offset with 50% prob
                offset_is_syncable = random.random() < self.prob_syncable  # 1=syncable, 0=non-syncable
                if offset_is_syncable:
                    offset_sec = random.choice(self.class_grid.tolist())
                else:
                    offset_sec = random.choice([-self.crop_len_sec, self.crop_len_sec])  # either - or + offset
                # aud starts `offset_sec` earlier than it should; aud has what will be shown after offset_sec

                offset_sec = round(offset_sec, 2)
                v_start_max_sec = frames2sec(v_len_frames - v_crop_len_frames, v_fps)
                assert v_start_max_sec > 0, f'{v_len_frames} {v_crop_len_frames} {v_fps} @ {item["path"]}'
                # `v_start_sec` IS NOT rounded to the fps grid
                v_start_sec = random.uniform(max(0, -offset_sec), min(v_start_max_sec, v_start_max_sec-offset_sec))
                assert 0 <= v_start_sec <= v_start_max_sec, f'{v_start_sec} {v_start_max_sec} {item["path"]}'
                v_start_i = sec2frames(v_start_sec, v_fps)
                v_end_i = v_start_i + v_crop_len_frames
                # `v_start_i_sec` IS rounded to the fps grid
                v_start_i_sec = frames2sec(v_start_i, v_fps)
                # `a_start_i` depends on the rounded value `v_start_i_sec`, otherwise
                # (v_start_sec) we have ±0.1 jittering
                a_start_i = sec2frames(v_start_i_sec + offset_sec, a_fps)
                if self.max_a_jitter_sec is not None and self.max_a_jitter_sec > 0:
                    a_start_i, a_jitter_i = apply_a_jitter(a_start_i, a_len_frames, a_crop_len_frames, a_fps,
                                                           self.max_a_jitter_sec)
                    item['meta']['a_jitter_i'] = a_jitter_i
                a_end_i = a_start_i + a_crop_len_frames
            else:
                offset_sec = round(offset_sec, 2)
                v_start_i = sec2frames(v_start_i_sec, v_fps)
                a_start_i = sec2frames(v_start_i_sec + offset_sec, a_fps)
                v_end_i = v_start_i + v_crop_len_frames
                a_end_i = a_start_i + a_crop_len_frames
        else:
            offset_sec = 0.0
            is_random_crop = item['split'] == 'train'
            v_start_i, v_end_i = self.get_crop_idx(v_len_frames, v_crop_len_frames, is_random=is_random_crop)
            v_start_i_sec = frames2sec(v_start_i, v_fps)
            a_start_i = sec2frames(v_start_i_sec, a_fps)
            if self.max_a_jitter_sec is not None and self.max_a_jitter_sec > 0:
                a_start_i, a_jitter_i = apply_a_jitter(a_start_i, a_len_frames, a_crop_len_frames, a_fps,
                                                       self.max_a_jitter_sec)
                item['meta']['a_jitter_i'] = a_jitter_i
            a_end_i = a_start_i + a_crop_len_frames

        # sometimes due to the rounding error e.g. v_start_sec = 1.505 but sec2frames(1.505, 25) = 1.48
        # given offset is -1.5, the a_start_i will be a small negative value. (likely a_fps * 1/v_fps * 0.5)
        if a_start_i < 0:
            how_much_out = a_start_i
            logging.info(f'a_start_i is negative ({how_much_out}) at {item["path"]}')
            if abs(how_much_out) <= a_fps / v_fps:
                logging.info('fixing it')
                a_start_i += abs(how_much_out)
                a_end_i += abs(how_much_out)
            else:
                raise Exception(f'{how_much_out} {item["path"]}')

        assert v_start_i < v_end_i and a_start_i < a_end_i
        assert aud.shape[0] >= a_end_i, f'{aud.shape} {a_end_i} {item["path"]}'
        assert vid.shape[0] >= v_end_i, f'{vid.shape} {v_end_i} {item["path"]}'

        vid, aud = vid[v_start_i:v_end_i, :, :, :], aud[a_start_i:a_end_i]

        item['video'] = vid
        item['audio'] = aud

        assert item['video'].shape[0] == int(v_fps*self.crop_len_sec), f'{item["video"].shape} {item["path"]}'
        assert item['audio'].shape[0] == int(a_fps*self.crop_len_sec), f'{item["audio"].shape} {item["path"]}'

        # caching parameters
        if self.do_offset:
            # NOTE: this is useless for the extreme offsetting
            offset_label, offset_target = quantize_offset(self.class_grid, offset_sec)
            item['targets']['offset_sec'] = offset_sec
            item['targets']['offset_label'] = offset_label
            # assert 'offset_target' not in item['targets'], f'{item["targets"]}. What passed it there?'
            item['targets']['offset_target'] = offset_target
            item['targets']['v_start_i_sec'] = v_start_i_sec
            item['targets']['sync_target'] = int(offset_is_syncable)

        return item

    def get_crop_idx(self, len_frames: int, crop_len_frames: int, is_random=True):
        if len_frames == crop_len_frames:
            return 0, len_frames
        if is_random:
            left_i = random.randint(0, len_frames - crop_len_frames)
        else:
            left_i = int(round((len_frames - crop_len_frames) / 2.))
        return left_i, left_i+crop_len_frames


class RGBToFloatToZeroOne(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item):
        item['video'] = item['video'].to(torch.float32).div(255.)
        return item


class RGBToHalfToZeroOne(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item):
        item['video'] = item['video'].half().div(255.)
        return item


class RGBNormalize(torchvision.transforms.Normalize):
    '''The same as the torchvision`s but with different interface for the dict.
    This should work for any shape (..., C, H, W)'''

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
        logging.info(f'RGBNormalize: mean={mean}, std={std}')

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        item['meta']['video']['norm_stats'] = {'mean': torch.as_tensor(self.mean),
                                               'std': torch.as_tensor(self.std)}
        return item


class AudioRandomVolume(torch.nn.Module):

    def __init__(self, p: float, **kwargs):
        super().__init__()
        transform = torchaudio.transforms.Vol(**kwargs)
        self.transform = torchvision.transforms.RandomApply([transform], p)

    def apply_to_single_clip(self, clip):
        return self.transform(clip)

    def apply_to_each_clip(self, clips):
        for i, clip in enumerate(clips):
            clips[i] = self.apply_to_single_clip(clip)
        return clips

    def forward(self, item):
        has_batch_dim = len(item['audio'].shape) == 2
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['audio'] = fn(item['audio'])
        return item


class AudioRandomLowpassFilter(torch.nn.Module):

    def __init__(self, p: float, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.p = p
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def apply_to_single_clip(self, clip, sr):
        if self.p > torch.rand(1):
            return torchaudio.functional.lowpass_biquad(clip, sr, self.cutoff_freq, self.Q)
        else:
            return clip

    def apply_to_each_clip(self, clips, sr):
        for i, clip in enumerate(clips):
            clips[i] = self.apply_to_single_clip(clip, sr)
        return clips

    def forward(self, item):
        has_batch_dim = len(item['audio'].shape) == 2
        sr = int(item['meta']['audio']['framerate'][0])
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['audio'] = fn(item['audio'], sr)
        return item


class AudioRandomPitchShift(torch.nn.Module):

    def __init__(self, p: float, shift: int) -> None:
        super().__init__()
        self.p = p
        self.shift = shift

    def apply_to_single_clip(self, wave, sr):
        if self.p > torch.rand(1):
            effects = [['pitch', f'{self.shift}'], ['rate', f'{sr}']]
            wave = wave.unsqueeze(0)
            wave, _ = torchaudio.sox_effects.apply_effects_tensor(wave, sr, effects)
            wave = wave.squeeze(0)
        return wave

    def apply_to_each_clip(self, waves, sr):
        for i, wave in enumerate(waves):
            waves[i] = self.apply_to_single_clip(wave, sr)
        return waves

    def forward(self, item):
        has_batch_dim = len(item['audio'].shape) == 2
        sr = int(item['meta']['audio']['framerate'][0])
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['audio'] = fn(item['audio'], sr)
        return item


class AudioRandomReverb(torch.nn.Module):

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        self.effects = [['reverb', '-w']]

    def apply_to_single_clip(self, wave, fps):
        if self.p > torch.rand(1):
            wave = wave.unsqueeze(0)
            wave, _ = torchaudio.sox_effects.apply_effects_tensor(wave, fps, self.effects)
            wave = wave.mean(dim=0)
        return wave

    def apply_to_each_clip(self, waves, fps):
        for i, wave in enumerate(waves):
            waves[i] = self.apply_to_single_clip(wave, fps)
        return waves

    def forward(self, item):
        has_batch_dim = len(item['audio'].shape) == 2
        sr = int(item['meta']['audio']['framerate'][0])
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['audio'] = fn(item['audio'], sr)
        return item

class AudioRandomGaussNoise(torch.nn.Module):

    def __init__(self, p: float, amplitude=0.01) -> None:
        super().__init__()
        self.p = p
        self.amplitude = amplitude

    def apply_to_single_clip(self, wave):
        if self.p > torch.rand(1):
            noise = torch.randn_like(wave, dtype=wave.dtype)
            wave = wave + self.amplitude * noise
        return wave

    def apply_to_each_clip(self, waves):
        for i, wave in enumerate(waves):
            waves[i] = self.apply_to_single_clip(wave)
        return waves

    def forward(self, item):
        has_batch_dim = len(item['audio'].shape) == 2
        if has_batch_dim:
            fn = self.apply_to_each_clip
        else:
            fn = self.apply_to_single_clip
        item['audio'] = fn(item['audio'])
        return item


class AudioMelSpectrogram(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    def forward(self, item):
        item['audio'] = self.spec(item['audio'])  # safe for batched input
        return item


class AudioLog(torch.nn.Module):

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, item):
        item['audio'] = torch.log(item['audio'] + self.eps)
        return item

class PadOrTruncate(torch.nn.Module):

    def __init__(self, max_spec_t: int, pad_mode: str = 'constant', pad_value: float = 0.0):
        super().__init__()
        self.max_spec_t = max_spec_t
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, item):
        item['audio'] = self.pad_or_truncate(item['audio'])
        return item

    def pad_or_truncate(self, audio):
        difference = self.max_spec_t - audio.shape[-1]  # safe for batched input
        # pad or truncate, depending on difference
        if difference > 0:
            # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
            pad_dims = (0, difference)
            audio = torch.nn.functional.pad(audio, pad_dims, self.pad_mode, self.pad_value)
        elif difference < 0:
            logging.warning(f'Truncating spec ({audio.shape}) to max_spec_t ({self.max_spec_t}).')
            audio = audio[..., :self.max_spec_t]  # safe for batched input
        return audio


class AudioNormalizeAST(torch.nn.Module):
    '''Normalization is done with two specified mean and std (half)'''
    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, item):
        item['audio'] = (item['audio'] - self.mean) / (2 * self.std)
        item['meta']['audio']['norm_stats'] = {'mean': self.mean, 'std': self.std}
        return item


class PermuteStreams(torch.nn.Module):

    def __init__(self, einops_order_audio: str, einops_order_rgb: str) -> None:
        ''' For example:
                einops_order_audio: "S F T -> S T F"
                einops_order_rgb: "S T C H W -> S C T H W"'''
        super().__init__()
        self.einops_order_audio = einops_order_audio
        self.einops_order_rgb = einops_order_rgb

    def forward(self, item):
        if self.einops_order_audio is not None:
            item['audio'] = einops.rearrange(item['audio'], self.einops_order_audio).contiguous()
        if self.einops_order_rgb is not None:
            item['video'] = einops.rearrange(item['video'], self.einops_order_rgb).contiguous()
        return item


class ResampleAudio(torch.nn.Module):

    def __init__(self, new_fps: int):
        super().__init__()
        self.new_fps = new_fps

    def forward(self, item):
        orig_fps = int(item['meta']['audio']['framerate'][0])
        item['meta']['audio']['orig_shape'] = item['audio'].shape
        if orig_fps != self.new_fps:
            item['audio'] = torchaudio.functional.resample(item['audio'], orig_fps, self.new_fps)
            item['meta']['audio']['framerate'][0] = self.new_fps
        return item

class ResampleRGB(torch.nn.Module):

    def __init__(self, new_fps: int) -> None:
        super().__init__()
        self.new_fps = new_fps

    def forward(self, item):
        orig_fps = float(item['meta']['video']['fps'][0])
        item['meta']['video']['orig_shape'] = item['video'].shape
        if orig_fps != self.new_fps:
            duration_sec = item['video'].shape[0] / orig_fps
            indices = torch.arange(0, orig_fps * duration_sec - 1e-9, orig_fps / self.new_fps)
            # basically, rounding
            indices = indices.to(dtype=torch.long)
            item['video'] = item['video'][indices]
            item['meta']['video']['fps'][0] = self.new_fps
        return item

class ResizeAndLetterboxPad(torch.nn.Module):
    '''Adapted from WACV24 Amazon`s challenge'''

    def __init__(self, new_h, new_w):
        super().__init__()
        self.new_h = new_h
        self.new_w = new_w
        self.aspect_ratio = new_w / new_h

    def forward(self, item):
        item['video'] = self.resize_and_pad(item['video'])
        return item

    def resize_and_pad(self, rgb: torch.Tensor):
        _, _, height, width = rgb.shape
        current_aspect_ratio = width / height
        if current_aspect_ratio > self.aspect_ratio:
            scaled_height = round(self.new_w / current_aspect_ratio)
            rgb = torchvision.transforms.functional.resize(rgb, (scaled_height, self.new_w), antialias=None)
            top = (self.new_h - scaled_height) // 2
            bottom = self.new_h - (scaled_height + top)
            rgb = torch.nn.ConstantPad2d((0, 0, top, bottom), 0)(rgb)
        elif current_aspect_ratio < self.aspect_ratio:
            scaled_width = round(self.new_h*current_aspect_ratio)
            rgb = torchvision.transforms.functional.resize(rgb, (self.new_h, scaled_width), antialias=None)
            left = (self.new_w - scaled_width) // 2
            right = self.new_w - (scaled_width + left)
            rgb = torch.nn.ConstantPad2d((left, right, 0, 0), 0)(rgb)
        return rgb


class ResampleResizeLetterboxPad(torch.nn.Module):

    def __init__(self, afps, vfps, new_h, new_w) -> None:
        super().__init__()
        self.transforms = torchvision.transforms.Compose([
            ResampleAudio(new_fps=afps),
            ResampleRGB(new_fps=vfps),
            ResizeAndLetterboxPad(new_h=new_h, new_w=new_w)
        ])

    def forward(self, x: dict) -> dict:
        return self.transforms(x)

class DoNothing(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: dict) -> dict:
        return x


if __name__ == '__main__':
    grid = make_class_grid(-1, 1, 21)
    grid = make_class_grid(-2, 2, 41)
    print('grid:', grid)
    print('value quantization:', quantize_offset(grid, 0.06))
    v_fps = 25.0
    duration = 10.0

    input = {
        'video': torch.randint(0, 256, (int(duration * v_fps), 3, 720//2, 1280//2), dtype=torch.uint8),
        'audio': torch.arange(221184-1).float(),
        'targets': {},
        'meta': {
            'video': {'duration': [duration], 'fps': [v_fps]},
            'audio': {'duration': [duration], 'framerate': [22050.0]},
            'subtitles': {'duration': []},
            'cc': {'duration': []},
        },
        'path': '/home/nvme/data/vggsound/video/-5cWCaoEDlE_261000_271000.mp4',
        'split': 'train',
    }

    print(input['audio'].shape, input['video'].shape)

    fn = EqualifyFromRight(clip_max_len_sec=10)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape)

    fn = RGBSpatialCrop((224, 224), is_random=True)
    # fn = RGBSpatialCrop((112, 112), is_random=True)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = Resize((224, 224))
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = GenerateMultipleSegments(segment_size_vframes=16, n_segments=14,
                                  is_start_random=False, audio_jitter_sec=0.05, step_size_seg=0.5)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = RandomApplyColorDistortion(p_gray_scale=0.5, p_color_jitter=0.5, s=1.0)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = RGBToFloatToZeroOne()
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])
    print(input['meta'])

    fn = RGBNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])
    print(input['video'].mean(dim=(0, 2, 3)))
    print(input['meta'])

    fn = AudioRandomReverb(p=1.0)
    input = fn(input)

    fn = AudioRandomVolume(p=1.0, gain=2.0, gain_type='amplitude')
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioRandomPitchShift(p=1.0, shift=1000)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioRandomLowpassFilter(p=1.0, cutoff_freq=100)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioRandomGaussNoise(p=1.0, amplitude=0.01)
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    fn = AudioLog()
    input = fn(input)
    print(input['audio'].shape, input['video'].shape, input['meta']['audio'])

    # audio only
    input = {
        'audio': torch.arange(221184).float(),
        'meta': {
            'video': {'duration': [10.0], 'fps': [10.0]},
            'audio': {'duration': [11.0], 'framerate': [22050.0]},
            'subtitles': {'duration': []},
            'cc': {'duration': []}
        },
        'path': '/home/nvme/data/vggsound/video/-5cWCaoEDlE_261000_271000.mp4'
    }

    print(input['audio'].shape)

    fn = AudioLog()
    input = fn(input)
    print(input['audio'].shape, input['meta']['audio'])
    print(input['meta'])
    print(input['audio'].min(), input['audio'].max())
