'''adapted from https://github.com/kylemin/S3D'''
from pathlib import Path
import logging
import sys

import einops
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('.')  # nopep8
from model.modules.feat_extractors.visual.motionformer import AveragePooling, SpatialTransformerEncoderLayer, TemporalTransformerEncoderLayer


class S3D(nn.Module):
    def __init__(self, num_class, extract_features):
        super(S3D, self).__init__()
        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            Mixed_5b(),
            Mixed_5c(),
        )
        self.extract_features = extract_features
        self.embed_dim = 1024
        self.fc = nn.Sequential(nn.Conv3d(self.embed_dim, num_class, kernel_size=1, stride=1, bias=True),)

    def forward(self, x):
        y = self.base(x)

        if self.extract_features:
            return y

        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        y = torch.mean(y, 2)

        return y


class S3DVisualFeatures(S3D):

    # ckpt_path should default to None, otherwise when no pre-training is desired it will throw an error
    def __init__(self,
                 extract_features: bool = False,
                 ckpt_path: str = None,
                 factorize_space_time: bool = None,
                 agg_space_module: str = None,
                 agg_time_module: str = None,
                 add_global_repr: bool = True,
                 agg_segments_module: str = None,
                 max_segments: int = None,):
        super().__init__(num_class=400, extract_features=extract_features)
        assert extract_features, 'Not implemented otherwise'
        self.extract_features = extract_features
        self.ckpt_path = ckpt_path
        self.factorize_space_time = factorize_space_time
        # similar to those in motionformer
        self.num_heads = 8  # 12 can't devide 1024
        self.mlp_ratio = 4
        self.drop_rate = 0.0

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            was_pt_on_k400 = ckpt_path.endswith('S3D_kinetics400_torchified.pt')
            if was_pt_on_k400:
                self.load_state_dict(ckpt, strict=True)

        # do not keep fc as they have ~300k params
        if extract_features:
            self.fc = nn.Identity()

        # avoiding code duplication (used only if agg_*_module is TransformerEncoderLayer)
        transf_enc_layer_kwargs = dict(
            d_model=self.embed_dim, nhead=self.num_heads, activation=nn.GELU(), batch_first=True,
            dim_feedforward=self.mlp_ratio*self.embed_dim, dropout=self.drop_rate, layer_norm_eps=1e-6,
            norm_first=True,
        )
        # define adapters if needed
        if self.factorize_space_time:
            if agg_space_module == 'TransformerEncoderLayer':
                self.spatial_attn_agg = SpatialTransformerEncoderLayer(**transf_enc_layer_kwargs)
            elif agg_space_module == 'AveragePooling':
                self.spatial_attn_agg = AveragePooling(avg_pattern='BS D t h w -> BS D t',
                                                       then_permute_pattern='BS D t -> BS t D')
            if agg_time_module == 'TransformerEncoderLayer':
                self.temp_attn_agg = TemporalTransformerEncoderLayer(**transf_enc_layer_kwargs)
            elif agg_time_module == 'AveragePooling':
                self.temp_attn_agg = AveragePooling(avg_pattern='BS t D -> BS D')
            elif 'Identity' in agg_time_module:
                self.temp_attn_agg = nn.Identity()
        # define a global aggregation layer (aggregarate over segments)
        self.add_global_repr = add_global_repr
        if add_global_repr:
            if agg_segments_module == 'TransformerEncoderLayer':
                # we can reuse the same layer as for temporal factorization (B, dim_to_agg, D) -> (B, D)
                # we need to add pos emb (PE) because previously we added the same PE for each segment
                pos_max_len = max_segments if max_segments is not None else 16  # 16 = 10sec//0.64sec + 1
                self.global_attn_agg = TemporalTransformerEncoderLayer(
                    add_pos_emb=True, pos_emb_drop=self.drop_rate, pos_max_len=pos_max_len,
                    **transf_enc_layer_kwargs
                )
            elif agg_segments_module == 'AveragePooling':
                self.global_attn_agg = AveragePooling(avg_pattern='B S D -> B D')

        if ckpt_path is not None:
            if not was_pt_on_k400:
                # we need to filter out the state_dict of the AVCLIP model (has both A and V extractors)
                # and keep only the state_dict of the feat extractor
                ckpt_weights = dict()
                for k, v in ckpt['state_dict'].items():
                    if k.startswith(('module.v_encoder.', 'v_encoder.')):
                        k = k.replace('module.', '').replace('v_encoder.', '')
                        ckpt_weights[k] = v
                _load_status = self.load_state_dict(ckpt_weights, strict=False)
                if len(_load_status.missing_keys) > 0 or len(_load_status.unexpected_keys) > 0:
                    logging.warning(f'Loading exact vfeat_extractor ckpt from {self.ckpt_path} failed. \n' \
                                    f'Missing keys ({len(_load_status.missing_keys)}): ' \
                                    f'{_load_status.missing_keys}, \n' \
                                    f'Unexpected keys ({len(_load_status.unexpected_keys)}): ' \
                                    f'{_load_status.unexpected_keys} \n' \
                                    f'temp_attn_agg are expected to be missing if ckpt was pt contrastively.')
                else:
                    logging.info(f'Loading vfeat_extractor ckpt from {self.ckpt_path} succeeded.')

        # print the number of parameters
        logging.info(f'vfeat_extractor: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}')

    def forward(self, x, for_loop: bool = False, cont_mask: torch.Tensor = None):
        '''
        x is of shape (B, S, C, T, H, W) where S is the number of segments.
        '''
        assert for_loop is False and cont_mask is None, 'Not implemented'
        # Batch, Segments, Channels, T=frames, Height, Width
        B, S, C, T, H, W = x.shape

        # flatten the batch and segments dimensions
        x = einops.rearrange(x, 'B S C T H W -> (B S) C T H W')

        x = self.forward_segments(x)

        x = x.view(B, S, *x.shape[1:])

        global_x = None
        if self.extract_features and self.add_global_repr:  # lazy exec:, add_global_repr might not be there
            assert len(x.shape) == 3, f'Local representation should be (B, S, D) {x.shape}'
            global_x = self.global_attn_agg(x)  # (B, D)

        return x, global_x  # x is (B, S, ...), global_x is (B, D) or None

    def forward_segments(self, x) -> torch.Tensor:
        '''x is of shape (BS, C, T, H, W) where S is the number of segments.'''
        # (BS, D, t, h, w) <- (BS, C, Tv, H, W)
        x = self.base(x)

        if self.extract_features:
            if self.factorize_space_time:
                # (B*S, t, D) <- (B*S, D, t, h, w)
                x = self.spatial_attn_agg(x)
                x = self.temp_attn_agg(x)  # (B*S, D) or (BS, t, D) if `self.temp_attn_agg` is `Identity`
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=(
            1, stride, stride), padding=(0, padding, padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1),
                                stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


if __name__ == '__main__':
    B = 2
    ckpt_path = './model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt'
    if Path(ckpt_path).exists():
        vfeat_extractor = S3DVisualFeatures(ckpt_path=ckpt_path)
        # x = torch.rand(B, 200, 3, 224, 224)
        x = torch.rand(B, 125, 3, 224, 224)
        x = vfeat_extractor(x)
        print(x.shape)
    else:
        print(ckpt_path, 'does not exist')
