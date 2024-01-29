'''
mostly taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
Mods:
- we use it as the encoder so it was slightly modified to avoid confusion with language modelling (GPT)
'''

import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # mask = torch.tril(torch.ones(config.block_size,
        #                              config.block_size))
        # if hasattr(config, "n_unmasked"):
        #     mask[:config.n_unmasked, :config.n_unmasked] = 1
        # self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NoPosEncoding(nn.Module):
    '''Does not apply any positional encoding'''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x

class ZeroInitPositionalEncoding(nn.Module):
    ''' Zero inited trainable pos embedding. It is just applied on the sequence, thus respects no priors. '''

    def __init__(self, block_shape, n_embd):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.zeros(1, *block_shape, n_embd))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb

class RandInitPositionalEncoding(nn.Module):
    ''' Random inited trainable pos embedding. It is just applied on the sequence, thus respects no priors.'''

    def __init__(self, block_shape: list, n_embd: int):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.randn(1, *block_shape, n_embd))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb


class PositionEmbeddingLearnedVisual(nn.Module):

    def __init__(self, block_shape, n_embd) -> None:
        super().__init__()
        self.block_shape = block_shape
        self.max_t, self.max_h, self.max_w = block_shape
        self.n_embd = n_embd
        # dividing n_embd almost evenly among each dimension; the remainer will be given to the time
        # dimension if `n_embd` is not divisible
        self.n_dims = len(block_shape)
        self.n_embd_t = self.n_embd_h = self.n_embd_w = self.n_embd // self.n_dims
        self.n_embd_t += self.n_embd % self.n_dims
        self.time_embed = nn.Embedding(self.max_t, self.n_embd_t)
        self.height_embed = nn.Embedding(self.max_h, self.n_embd_h)
        self.width_embed = nn.Embedding(self.max_w, self.n_embd_w)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)
        nn.init.uniform_(self.height_embed.weight)
        nn.init.uniform_(self.width_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): a batch of visual feature maps (B, t, h, w, d)
        Returns:
            torch.Tensor: x + pos
        '''
        return x + self.make_pos_emb(x)

    def make_pos_emb(self, x):
        B, t, h, w, d = x.shape
        t_i = torch.arange(t, device=x.device)
        w_i = torch.arange(w, device=x.device)
        h_i = torch.arange(h, device=x.device)
        # (t/w/h, D)
        t_emb = self.time_embed(t_i)
        w_emb = self.width_embed(w_i)
        h_emb = self.height_embed(h_i)
        # (t, w, h, d//3)
        t_emb = t_emb.view(t, 1, 1, self.n_embd_t).repeat(1, h, w, 1)
        w_emb = w_emb.view(1, 1, w, self.n_embd_w).repeat(t, h, 1, 1)
        h_emb = h_emb.view(1, h, 1, self.n_embd_h).repeat(t, 1, w, 1)
        # (t, w, h, d)
        pos = torch.cat([t_emb, w_emb, h_emb], dim=-1)
        # (B, t, w, h, d) -- same as x
        pos = pos.view(1, t, h, w, d).repeat(B, 1, 1, 1, 1)
        return pos


class PositionEmbeddingLearnedAudio(nn.Module):

    def __init__(self, block_shape, n_embd) -> None:
        super().__init__()
        self.block_shape = block_shape
        self.max_f, self.max_t = block_shape
        # dividing n_embd almost evenly among each dimension; the remainer will be given to the time
        # dimension if `n_embd` is not divisible: e.g. 512 / 3 -> [170, 170, 170+2]
        self.n_dims = len(block_shape)
        self.n_embd_t = self.n_embd_f = n_embd // self.n_dims
        self.n_embd_t += n_embd % self.n_dims
        self.freq_embed = nn.Embedding(self.max_f, self.n_embd_f)
        self.time_embed = nn.Embedding(self.max_t, self.n_embd_t)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.freq_embed.weight)
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): a batch of spectrogram features maps (B, f, t, d)
        Returns:
            torch.Tensor: x + pos
        '''
        return x + self.make_pos_emb(x)

    def make_pos_emb(self, x):
        B, f, t, d = x.shape
        f_i = torch.arange(f, device=x.device)
        t_i = torch.arange(t, device=x.device)
        # (f/t, D)
        f_emb = self.freq_embed(f_i)
        t_emb = self.time_embed(t_i)
        # (f, t, d//2)
        f_emb = f_emb.view(f, 1, self.n_embd_f).repeat(1, t, 1)
        t_emb = t_emb.view(1, t, self.n_embd_t).repeat(f, 1, 1)
        # (f, t, d)
        pos = torch.cat([f_emb, t_emb], dim=-1)
        # (B, f, t, d)
        pos = pos.view(1, f, t, d).repeat(B, 1, 1, 1)
        return pos


class L2Normalize(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        eps = 1e-6 if isinstance(x, (torch.HalfTensor, torch.cuda.HalfTensor)) else 1e-12
        x = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=eps)
        return x


class Transformer(nn.Module):
    ''' the full transformer, with a context size of block_size '''

    def __init__(self, vis_pos_emb_module, aud_pos_emb_module, num_offset_cls,
                 visual_block_shape, audio_block_shape, pre_norm_cfg,
                 n_layer=12, n_head=8, n_embd=256, tok_pdrop=0., embd_pdrop=0., resid_pdrop=0., attn_pdrop=0.,
                 n_unmasked=0):
        super().__init__()
        config = Config(num_offset_cls=num_offset_cls,
                        audio_block_shape=audio_block_shape, visual_block_shape=visual_block_shape,
                        embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                        tok_pdrop=tok_pdrop, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                        n_unmasked=n_unmasked)
        self.config = config
        # input embedding stem
        self.OFF_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = nn.Parameter(torch.randn(1, 1, n_embd))
        self.pre_lnorm_vis = instantiate_from_config(pre_norm_cfg)
        self.pre_lnorm_aud = instantiate_from_config(pre_norm_cfg)
        self.vis_pos_emb = instantiate_from_config(vis_pos_emb_module)
        self.aud_pos_emb = instantiate_from_config(aud_pos_emb_module)
        self.tok_drop_vis = nn.Dropout3d(tok_pdrop)
        self.tok_drop_aud = nn.Dropout2d(tok_pdrop)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # heads
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.off_head = nn.Linear(config.n_embd, config.num_offset_cls, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, vis_embd: torch.Tensor, aud_embd: torch.Tensor, targets=None):
        B, Tv, Dv, H, W = vis_embd.shape
        B, Da, F, Ta = aud_embd.shape
        assert Da == Dv, f'Please define a bridge or fix {Da} vs {Dv}'

        # (B, Tv, H, W, Dv), (B, F, Ta, Da) <-
        vis_embd = vis_embd.permute(0, 1, 3, 4, 2).contiguous()
        aud_embd = aud_embd.permute(0, 2, 3, 1).contiguous()

        # making sure that both embeddings are normalized to the same base (can be configured as identity tho)
        vis_embd = self.pre_lnorm_vis(vis_embd)
        aud_embd = self.pre_lnorm_aud(aud_embd)

        # apply individual pos embeddings
        vis_embd = self.vis_pos_emb(vis_embd)
        aud_embd = self.aud_pos_emb(aud_embd)

        # apply token dropout
        vis_embd = self.tok_drop_vis(vis_embd)
        aud_embd = self.tok_drop_aud(aud_embd.transpose(2, 1)).transpose(2, 1)  # applying dropout on time-dim

        # (B, Nv, Dv), (B, Na, Da)
        vis_embd = vis_embd.view(B, Tv*H*W, Dv)
        aud_embd = aud_embd.view(B, F*Ta, Da)

        # broadcasting special tokens to the batch size
        off_tok = self.OFF_tok.expand(B, -1, -1)
        mod_tok = self.MOD_tok.expand(B, -1, -1)

        # (B, 2+Nv+1+Na, D)
        token_embeddings = torch.cat((off_tok, vis_embd, mod_tok, aud_embd), dim=1)

        x = self.drop(token_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        # this depends on the order in `token_embeddings`
        off_logits = self.off_head(x[:, 0, :])

        return off_logits


if __name__ == '__main__':
    # pos_enc = FourierPositionEncoding(block_shape=[9, 27], n_embd=512)
    # x = torch.rand(3, 9, 27, 512)
    # print('input', x.shape)
    # x = pos_enc(x)
    # print(x.shape)

    # pos_enc = FourierPositionEncoding(block_shape=[50, 7, 7], n_embd=512)
    # x = torch.rand(3, 50, 7, 7, 512)
    # print('input', x.shape)
    # x = pos_enc(x)
    # print(x.shape)

    pos_enc = PositionEmbeddingLearnedVisual([50, 7, 7], 512)
    x = torch.rand(3, 50, 7, 7, 512)
    print('x', x.shape)
    print('x.shape', pos_enc(x).shape)

    pos_enc = PositionEmbeddingLearnedAudio([9, 27], 512)
    x = torch.rand(3, 9, 27, 512)
    print('x', x.shape)
    print('x.shape', pos_enc(x).shape)

    ############################################################################
    # from omegaconf import OmegaConf
    # from time import time

    # cfg = OmegaConf.load('./configs/av_sync.yaml')
    # cfg.training.use_half_precision = True

    # device = torch.device('cuda:2')
    # torch.cuda.set_device(device)

    # model = instantiate_from_config(cfg.model.params.transformer)
    # model = model.to(device)

    # start_time = time()
    # for i in range(500):
    #     # vis = torch.rand(1, 50, 512, 7, 7, device=device)
    #     # aud = torch.rand(1, 512, 9, 27, device=device)
    #     vis = torch.rand(1, 50, 512, 1, 1, device=device)
    #     aud = torch.rand(1, 512, 1, 27, device=device)
    #     # cls_logits, off_logits, sync_logits, att = model(vis, aud)
    #     # inference in half precision
    #     with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
    #         out = model(vis, aud)
    # print('Time:', round(time() - start_time, 3))
    ############################################################################
