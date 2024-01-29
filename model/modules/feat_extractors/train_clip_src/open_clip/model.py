""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn

from utils.utils import instantiate_from_config

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256 # n_queries for attentional pooler
    attn_pooler_heads: int = 8 # n heads for attentional_pooling
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

class AVCLIP(nn.Module):

    def __init__(self, n_embd: int, afeat_extractor: OmegaConf, vfeat_extractor: OmegaConf,
                 aproj: OmegaConf, vproj: OmegaConf, init_scale: float = 0.07, clamp_scale_min: float = 0.001,
                 clamp_scale_max: float = 0.5, gather_for_loss: bool = False):
        super().__init__()
        self.output_dict = True
        self.n_embd = n_embd

        # loading audio and rgb towers
        self.v_encoder = instantiate_from_config(vfeat_extractor)
        self.a_encoder = instantiate_from_config(afeat_extractor)
        # loading audio and rgb towers and projection layers to account for different feature dimensions
        self.aproj = instantiate_from_config(aproj)
        self.vproj = instantiate_from_config(vproj)

        self.clamp_scale_min, self.clamp_scale_max = clamp_scale_min, clamp_scale_max
        self.init_scale = init_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * self.init_scale)  # NOTE: exp(1/OpenCLIP)

        self.gather_for_loss = gather_for_loss

        # self.ln_final = text.ln_final  # perhaps only useful for transformer towers
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, alpha: float = 0.0, for_loop: bool = False,
                world_size=1):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, S, C, Tv, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, Ta, F)
            alpha (float): linear interpolation coefficient for pseudo-targets (with 0.0 targets are 1-hot)
            for_loop (bool): whether to process each segment in a for loop or all at once
                             (speed-memory tradeoff)
        Returns:
            rgb_features (tuple(torch.Tensor)): local (B, S, D) and global (B, D) or None RGB features
            audio_features (tuple(torch.Tensor)): local (B, S, D) and global (B, D) or None audio features
            logit_scale (tuple(torch.Tensor)): local and global logit scales (1, )
        '''
        assert alpha == 0.0, f'alpha={alpha} not supported yet'
        logit_scales = self.clamp_logit_scales()
        vfeat, _, afeat, _ = self.encode_streams(vis, aud, for_loop, do_norm=True)

        if world_size > 1 and self.gather_for_loss:  # gather all features
            vfeat_all = torch.cat(torch.distributed.nn.all_gather(vfeat), dim=0)
            afeat_all = torch.cat(torch.distributed.nn.all_gather(afeat), dim=0)
        else:
            vfeat_all = vfeat
            afeat_all = afeat

        loss_avc, _ = self.compute_loss(vfeat, afeat, vfeat_all.mT, afeat_all.mT, self.logit_scale, alpha=0)
        out = {
            'rgb_features': (vfeat, None), 'audio_features': (afeat, None),
            'logit_scales': logit_scales,
            'losses': {'segment_contrastive_loss': loss_avc},
        }
        return out

    def compute_loss(self, vfeat, afeat, vfeat_all, afeat_all, scale, alpha=0.0, vfeat_m=None, afeat_m=None):
        '''For Multi-level contrastive learning, the losses are made the same way for all levels'''
        sim_v2a = vfeat @ afeat_all / scale
        sim_a2v = afeat @ vfeat_all / scale
        sim_v2a_t, sim_a2v_t = self._make_targets(sim_v2a, vfeat_all, afeat_all, scale, alpha, vfeat_m, afeat_m)
        loss = self._loss(sim_v2a, sim_a2v, sim_v2a_t, sim_a2v_t)
        return loss, (sim_v2a, sim_a2v)

    @torch.no_grad()
    def _make_targets(self, sim_v2a, vfeat_all, afeat_all, scale, alpha, vfeat_m, afeat_m):
        # NOTE: for simplicity, we assume that sim_v2a.shape[0] == sim_a2v.shape[0]
        # NOTE: sim_targets is not square (sim_v2a.shape is (bsize, bsize+Qsize) )
        sim_targets = torch.eye(*sim_v2a.shape, device=sim_v2a.device, dtype=sim_v2a.dtype)
        sim_v2a_targets = sim_targets
        sim_a2v_targets = sim_targets
        return sim_v2a_targets, sim_a2v_targets

    def _loss(self, sim_v2a, sim_a2v, sim_v2a_targets, sim_a2v_targets):
        loss_v2a = F.cross_entropy(sim_v2a, sim_v2a_targets)
        loss_a2v = F.cross_entropy(sim_a2v, sim_a2v_targets)
        return (loss_v2a + loss_a2v) / 2

    def encode_streams(self, vis, aud, for_loop, do_norm=True):
        # (B*S, D), (B, D) or None; because `flatten_to_2D = True`
        flatten_to_2D = True
        vfeat, _ = self.encode_stream(vis, self.v_encoder, self.vproj, do_norm, flatten_to_2D, for_loop)
        afeat, _ = self.encode_stream(aud, self.a_encoder, self.aproj, do_norm, flatten_to_2D, for_loop)
        return vfeat, None, afeat, None

    def encode_stream(self, x, feat_extractor_fn, proj_fn, do_norm, flatten_to_2D, for_loop):
        # x is (B, S, ...)
        segment_x, _ = feat_extractor_fn(x, for_loop)  # segment_x: (B, S, D), global_x: (B, D)
        if flatten_to_2D:
            B, S, D = segment_x.shape
            segment_x = segment_x.view(B*S, D)  # flatten batch and segment dims
        segment_x = proj_fn(segment_x)
        segment_x = F.normalize(segment_x, dim=-1) if do_norm else segment_x
        # do_global is passed in to avoid computing global features when not needed (e.g. during eval)
        return segment_x, None  # (B*S, D), (B, D) or None

    def forward_for_logging(self, vis, aud, for_momentum=False, for_loop=False, do_norm=True):
        '''
        Runs the forward pass but keeps certain tensors in memory for logging purposes, ie code duplication.
        NOTE: to be used outside of this module, most likely during logging

        Args:
            vis (torch.Tensor): RGB frames (B, S, C, Tv, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, Ta, F)
        '''
        flatten_to_2D = True
        out = dict()

        # factorizing self.encode_streams into encode_visual/encode_audio to avoid unnecessary computations
        # (B*S, D), (B, D) or None;
        # vfeat, _ = self.encode_stream(vis, self.v_encoder, self.vproj, do_norm, flatten_to_2D, for_loop)
        # afeat, _ = self.encode_stream(aud, self.a_encoder, self.aproj, do_norm, flatten_to_2D, for_loop)
        vfeat, _, afeat, _ = self.encode_streams(vis, aud, for_loop, do_norm)
        # cache features (for 0-shot evaluation)
        out['segment_vfeat'] = vfeat.clone()
        out['segment_afeat'] = afeat.clone()
        # and similiarity matrices (for visualization) (B*S, B*S)
        out['segment_sim_v2a'] = out['segment_vfeat'] @ out['segment_afeat'].mT / self.logit_scale
        out['segment_sim_a2v'] = out['segment_afeat'] @ out['segment_vfeat'].mT / self.logit_scale
        # self
        out['segment_sim_v2v'] = out['segment_vfeat'] @ out['segment_vfeat'].mT / self.logit_scale
        out['segment_sim_a2a'] = out['segment_afeat'] @ out['segment_afeat'].mT / self.logit_scale

        # compute losses
        loss, _ = self.compute_loss(vfeat, afeat, vfeat.mT, afeat.mT, self.logit_scale)
        out['segment_contrastive_loss'] = loss
        return out

    @torch.no_grad()
    def clamp_logit_scales(self):
        self.logit_scale.clamp_(self.clamp_scale_min, self.clamp_scale_max)
        return (self.logit_scale, None)


class MultilevelMoCoCLIP(nn.Module):

    def __init__(self, n_embd: int, queue_size: int, momentum: float,
                 afeat_extractor: OmegaConf, vfeat_extractor: OmegaConf, aproj: OmegaConf, vproj: OmegaConf,
                 init_scale: float = 0.07, clamp_scale_min: float = 0.001, clamp_scale_max: float = 0.5):
        super().__init__()
        self.output_dict = True
        self.n_embd = n_embd
        self.momentum = momentum
        self.to_add_global_repr = afeat_extractor.params.add_global_repr

        # loading audio and rgb towers
        self.v_encoder = instantiate_from_config(vfeat_extractor)
        self.a_encoder = instantiate_from_config(afeat_extractor)
        # loading audio and rgb towers and projection layers to account for different feature dimensions
        self.segment_aproj = instantiate_from_config(aproj)
        self.segment_vproj = instantiate_from_config(vproj)
        self.global_aproj = instantiate_from_config(aproj) if self.to_add_global_repr else None
        self.global_vproj = instantiate_from_config(vproj) if self.to_add_global_repr else None

        self.clamp_scale_min, self.clamp_scale_max = clamp_scale_min, clamp_scale_max
        self.init_scale = init_scale
        self.segment_logit_scale = nn.Parameter(torch.ones([]) * self.init_scale)  # NOTE: exp(1/OpenCLIP)
        self.global_logit_scale = nn.Parameter(torch.ones([]) * self.init_scale) if self.to_add_global_repr else None

        # create momentum models
        self.v_encoder_m = instantiate_from_config(vfeat_extractor)
        self.a_encoder_m = instantiate_from_config(afeat_extractor)
        self.segment_aproj_m = instantiate_from_config(aproj)
        self.segment_vproj_m = instantiate_from_config(vproj)
        self.global_aproj_m = instantiate_from_config(aproj) if self.to_add_global_repr else None
        self.global_vproj_m = instantiate_from_config(vproj) if self.to_add_global_repr else None

        self.model_pairs = [
            [self.v_encoder, self.v_encoder_m], [self.segment_vproj, self.segment_vproj_m],
            [self.a_encoder, self.a_encoder_m], [self.segment_aproj, self.segment_aproj_m],
        ]
        if self.to_add_global_repr:
            self.model_pairs += [
                [self.global_aproj, self.global_aproj_m], [self.global_vproj, self.global_vproj_m],
            ]

        self.copy_params()

        self.segment_queue_size = queue_size * afeat_extractor.params.max_segments  # scaled by # of segments
        self.global_queue_size = queue_size if self.to_add_global_repr else None
        self.init_Qs(self.segment_queue_size, self.global_queue_size, self.n_embd)

        # self.ln_final = text.ln_final  # perhaps only useful for transformer towers
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, alpha: float = 0.0, for_loop: bool = False,
                world_size=None):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, S, C, Tv, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, Ta, F)
            alpha (float): linear interpolation coefficient for pseudo-targets (with 0.0 targets are 1-hot)
            for_loop (bool): whether to process each segment in a for loop or all at once
                             (speed-memory tradeoff)
        Returns:
            rgb_features (tuple(torch.Tensor)): local (B, S, D) and global (B, D) or None RGB features
            audio_features (tuple(torch.Tensor)): local (B, S, D) and global (B, D) or None audio features
            logit_scale (tuple(torch.Tensor)): local and global logit scales (1, )
        '''
        logit_scales = self.clamp_logit_scales()
        to_add_global_repr = self.to_add_global_repr  # for readability only

        feats = self.encode_streams(vis, aud, for_momentum=False, for_loop=for_loop, do_norm=True)
        segment_vfeat, global_vfeat, segment_afeat, global_afeat = feats

        # get momentum features
        with torch.no_grad():
            if self.training:
                self._momentum_update()
            feats_m = self.encode_streams(vis, aud, for_momentum=True, for_loop=for_loop, do_norm=True)
            segment_vfeat_m, global_vfeat_m, segment_afeat_m, global_afeat_m = feats_m

            # cat with queue to extend the list of negatives
            segment_vfeat_all = torch.cat([segment_vfeat_m.t(), self.segment_v_queue.clone().detach()], dim=1)
            segment_afeat_all = torch.cat([segment_afeat_m.t(), self.segment_a_queue.clone().detach()], dim=1)
            if to_add_global_repr:
                global_vfeat_all = torch.cat([global_vfeat_m.t(), self.global_v_queue.clone().detach()], dim=1)
                global_afeat_all = torch.cat([global_afeat_m.t(), self.global_a_queue.clone().detach()], dim=1)

        segment_loss_avc, _ = self.compute_loss(segment_vfeat, segment_afeat, segment_vfeat_all,
                                                segment_afeat_all, self.segment_logit_scale,
                                                alpha, segment_vfeat_m, segment_afeat_m)

        global_loss_avc = None
        if to_add_global_repr:
            global_loss_avc, _ = self.compute_loss(global_vfeat, global_afeat, global_vfeat_all,
                                                   global_afeat_all, self.global_logit_scale,
                                                   alpha, global_vfeat_m, global_afeat_m)

        if self.training:
            self._multilevel_dequeue_and_enqueue(segment_vfeat_m, segment_afeat_m, global_vfeat_m, global_afeat_m)
        else:
            raise Exception('This module is used only during training. Use model.something instead.')

        out = {
            'rgb_features': (segment_vfeat, global_vfeat),
            'audio_features': (segment_afeat, global_afeat), 'logit_scales': logit_scales,
            'losses': {'segment_contrastive_loss': segment_loss_avc},
        }
        if global_loss_avc is not None:
            out['losses']['global_contrastive_loss'] = global_loss_avc
        return out

    def compute_loss(self, vfeat, afeat, vfeat_all, afeat_all, scale, alpha=0.0, vfeat_m=None, afeat_m=None):
        '''For Multi-level contrastive learning, the losses are made the same way for all levels'''
        sim_v2a = vfeat @ afeat_all / scale
        sim_a2v = afeat @ vfeat_all / scale
        sim_v2a_t, sim_a2v_t = self._make_targets(sim_v2a, vfeat_all, afeat_all, scale, alpha, vfeat_m, afeat_m)
        loss = self._loss(sim_v2a, sim_a2v, sim_v2a_t, sim_a2v_t)
        return loss, (sim_v2a, sim_a2v)

    @torch.no_grad()
    def _make_targets(self, sim_v2a, vfeat_all, afeat_all, scale, alpha, vfeat_m, afeat_m):
        # NOTE: for simplicity, we assume that sim_v2a.shape[0] == sim_a2v.shape[0]
        # NOTE: sim_targets is not square (sim_v2a.shape is (bsize, bsize+Qsize) )
        sim_targets = torch.eye(*sim_v2a.shape, device=sim_v2a.device, dtype=sim_v2a.dtype)
        # the ALBEF alpha trick
        if alpha > 0.0:
            sim_v2a_m = vfeat_m @ afeat_all / scale
            sim_a2v_m = afeat_m @ vfeat_all / scale
            sim_v2a_targets = alpha * F.softmax(sim_v2a_m, dim=1) + (1 - alpha) * sim_targets
            sim_a2v_targets = alpha * F.softmax(sim_a2v_m, dim=1) + (1 - alpha) * sim_targets
        else:
            sim_v2a_targets = sim_targets
            sim_a2v_targets = sim_targets
        return sim_v2a_targets, sim_a2v_targets

    def _loss(self, sim_v2a, sim_a2v, sim_v2a_targets, sim_a2v_targets):
        loss_v2a = F.cross_entropy(sim_v2a, sim_v2a_targets)
        loss_a2v = F.cross_entropy(sim_a2v, sim_a2v_targets)
        return (loss_v2a + loss_a2v) / 2

    def encode_streams(self, vis, aud, for_momentum, for_loop, do_norm=True):
        # (B*S, D), (B, D) or None; because `flatten_to_2D = True`
        flatten_to_2D = True
        segment_vfeat, global_vfeat = self.encode_visual(vis, for_momentum, self.to_add_global_repr, do_norm,
                                                         for_loop=for_loop, flatten_to_2D=flatten_to_2D)
        segment_afeat, global_afeat = self.encode_audio(aud, for_momentum, self.to_add_global_repr, do_norm,
                                                        for_loop=for_loop, flatten_to_2D=flatten_to_2D)
        return segment_vfeat, global_vfeat, segment_afeat, global_afeat

    def encode_audio(self, x, for_momentum: bool = False, do_global: bool = True, do_norm: bool = True,
                     flatten_to_2D=True, for_loop=False):
        # define callables
        encode_fn = self.a_encoder_m if for_momentum else self.a_encoder
        segment_proj_fn = self.segment_aproj_m if for_momentum else self.segment_aproj
        global_proj_fn = self.global_aproj_m if for_momentum else self.global_aproj
        # do the encoding
        return self.encode_stream(x, encode_fn, segment_proj_fn, global_proj_fn, do_global, do_norm,
                                  flatten_to_2D, for_loop)

    def encode_visual(self, x, for_momentum: bool = False, do_global: bool = True, do_norm: bool = True,
                      flatten_to_2D=True, for_loop=False):
        # define callables
        encode_fn = self.v_encoder_m if for_momentum else self.v_encoder
        segment_proj_fn = self.segment_vproj_m if for_momentum else self.segment_vproj
        global_proj_fn = self.global_vproj_m if for_momentum else self.global_vproj
        # do the encoding
        return self.encode_stream(x, encode_fn, segment_proj_fn, global_proj_fn, do_global, do_norm,
                                  flatten_to_2D, for_loop)

    def encode_stream(self, x, feat_extractor_fn, segment_proj_fn, global_proj_fn, do_global, do_norm,
                      flatten_to_2D, for_loop):
        # x is (B, S, ...)
        segment_x, global_x = feat_extractor_fn(x, for_loop)  # segment_x: (B, S, D), global_x: (B, D)
        if flatten_to_2D:
            B, S, D = segment_x.shape
            segment_x = segment_x.view(B*S, D)  # flatten batch and segment dims
        segment_x = segment_proj_fn(segment_x)
        segment_x = F.normalize(segment_x, dim=-1) if do_norm else segment_x
        # do_global is passed in to avoid computing global features when not needed (e.g. during eval)
        if do_global and self.to_add_global_repr:
            global_x = global_proj_fn(global_x)
            global_x = F.normalize(global_x, dim=-1) if do_norm else global_x
        return segment_x, global_x  # (B*S, D), (B, D) or None

    def forward_for_logging(self, vis, aud, for_momentum=False, for_loop=False, do_norm=True):
        '''
        Runs the forward pass but keeps certain tensors in memory for logging purposes, ie code duplication.
        NOTE: to be used outside of this module, most likely during logging

        Args:
            vis (torch.Tensor): RGB frames (B, S, C, Tv, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, Ta, F)
        '''
        flatten_to_2D = True

        out = dict()

        # factorizing self.encode_streams into encode_visual/encode_audio to avoid unnecessary computations
        # (B*S, D), (B, D) or None;
        segment_vfeat, global_vfeat = self.encode_visual(vis, for_momentum, self.to_add_global_repr, do_norm,
                                                         flatten_to_2D, for_loop)
        segment_afeat, global_afeat = self.encode_audio(aud, for_momentum, self.to_add_global_repr, do_norm,
                                                        flatten_to_2D, for_loop)
        # cache features (for 0-shot evaluation)
        out['segment_vfeat'] = segment_vfeat.clone()
        out['segment_afeat'] = segment_afeat.clone()
        # and similiarity matrices (for visualization) (B*S, B*S)
        out['segment_sim_v2a'] = out['segment_vfeat'] @ out['segment_afeat'].mT / self.segment_logit_scale
        out['segment_sim_a2v'] = out['segment_afeat'] @ out['segment_vfeat'].mT / self.segment_logit_scale
        # self
        out['segment_sim_v2v'] = out['segment_vfeat'] @ out['segment_vfeat'].mT / self.segment_logit_scale
        out['segment_sim_a2a'] = out['segment_afeat'] @ out['segment_afeat'].mT / self.segment_logit_scale

        # compute losses
        segment_loss, _ = self.compute_loss(segment_vfeat, segment_afeat, segment_vfeat.mT, segment_afeat.mT,
                                            self.segment_logit_scale)
        out['segment_contrastive_loss'] = segment_loss
        if self.to_add_global_repr:
            global_loss, _ = self.compute_loss(global_vfeat, global_afeat, global_vfeat.mT, global_afeat.mT,
                                               self.global_logit_scale)
            out['global_contrastive_loss'] = global_loss

        return out


    @torch.no_grad()
    def clamp_logit_scales(self):
        self.segment_logit_scale.clamp_(self.clamp_scale_min, self.clamp_scale_max)
        if self.to_add_global_repr:
            self.global_logit_scale.clamp_(self.clamp_scale_min, self.clamp_scale_max)
        return (self.segment_logit_scale, self.global_logit_scale)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _multilevel_dequeue_and_enqueue(self, segment_vfeat_m, segment_afeat_m, global_vfeat_m, global_afeat_m):
        if self.segment_queue_size > 0:
            self._dequeue_and_enqueue(segment_vfeat_m, segment_afeat_m, 'segment_')
        if self.to_add_global_repr and self.global_queue_size > 0:
            self._dequeue_and_enqueue(global_vfeat_m, global_afeat_m, 'global_')

    @torch.no_grad()
    def _dequeue_and_enqueue(self, vfeat, afeat, level_prefix_: str):
        # gather keys before updating queue
        if torch.distributed.is_initialized():
            vfeats = concat_all_gather(vfeat)
            afeats = concat_all_gather(afeat)
        else:
            vfeats = vfeat
            afeats = afeat

        batch_size = vfeats.shape[0]
        queue_size = getattr(self, level_prefix_ + 'queue_size')

        # same as `ptr = int(self.segment_queue_ptr)` but allows accessing the attribute by string
        ptr = int(getattr(self, level_prefix_ + 'queue_ptr'))
        assert queue_size % batch_size == 0, f'For simplicity: {queue_size} % {batch_size} == 0'

        # replace the keys at ptr (dequeue and enqueue)
        getattr(self, level_prefix_ + 'v_queue')[:, ptr:ptr + batch_size] = vfeats.T
        getattr(self, level_prefix_ + 'a_queue')[:, ptr:ptr + batch_size] = afeats.T
        ptr = (ptr + batch_size) % queue_size  # move pointer

        getattr(self, level_prefix_ + 'queue_ptr')[0] = ptr

    def init_Qs(self, segment_queue_size: int, global_queue_size: int, n_embd: int):
        # create the queues; TODO: flip the dimensions, yikes!
        self.register_buffer('segment_v_queue', torch.randn(n_embd, segment_queue_size))
        self.register_buffer('segment_a_queue', torch.randn(n_embd, segment_queue_size))
        self.register_buffer('segment_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.segment_v_queue = nn.functional.normalize(self.segment_v_queue, dim=0)
        self.segment_a_queue = nn.functional.normalize(self.segment_a_queue, dim=0)
        if self.to_add_global_repr:
            self.register_buffer('global_v_queue', torch.randn(n_embd, global_queue_size))
            self.register_buffer('global_a_queue', torch.randn(n_embd, global_queue_size))
            self.register_buffer('global_queue_ptr', torch.zeros(1, dtype=torch.long))
            self.global_v_queue = nn.functional.normalize(self.global_v_queue, dim=0)
            self.global_a_queue = nn.functional.normalize(self.global_a_queue, dim=0)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
