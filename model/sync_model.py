import logging
from typing import Any, Mapping
import sys

import einops
import torch
from torch.nn import functional as F

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config
from model.modules.transformer import Block, Config

def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Synchformer(torch.nn.Module):
    ''' The module has similar structure to SparseSync (SparseSync) but has a diffrerent
    forward pass. It expects the output of the feature extractors to have global and
    segment-level representations.'''

    def __init__(self, afeat_extractor, vfeat_extractor, aproj, vproj, transformer):
        super().__init__()
        self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        self.afeat_extractor = instantiate_from_config(afeat_extractor)
        # bridging the s3d latent dim (1024) into what is specified in the config
        # to match e.g. the transformer dim
        self.vproj = instantiate_from_config(vproj)
        self.aproj = instantiate_from_config(aproj)
        self.transformer = instantiate_from_config(transformer)

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, targets: torch.Tensor = None, for_loop=False,
                vis_mask: torch.Tensor = None, aud_mask: torch.Tensor = None, loss_fn=None):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, S, Tv, C, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, 1, F, Ta)
            for_loop (bool): if True, will use a for loop inside of feat_extractors to iterate over the
                             segments or process them in parallel (False), treating segment dim as batch dim
                             (speed-memory tradeoff).
            vis_mask (torch.Tensor): mask for the visual tokens (as input)
            aud_mask (torch.Tensor): mask for the audio tokens (as input)
        Returns:
            tuple(Tensor, Tensor), Tensor: loss values, logits
        '''
        vis = self.extract_vfeats(vis, for_loop, vis_mask=vis_mask)
        aud = self.extract_afeats(aud, for_loop, aud_mask=aud_mask)

        vis = self.vproj(vis)
        aud = self.aproj(aud)

        # flatten the segment dim (treating the sequence of segments as a single sequence)
        B, S, tv, D = vis.shape
        B, S, ta, D = aud.shape
        vis = vis.view(B, S*tv, D)  # (B, S*tv, D)
        aud = aud.view(B, S*ta, D)  # (B, S*ta, D)

        # self.transformer will concatenate the vis and aud in one sequence with aux tokens,
        # ie `CvvvvMaaaaaa`, and will return the logits for the CLS tokens
        logits = self.transformer(vis, aud)  # (B, cls); or (B, cls) and (B, 2) if DoubtingTransformer

        loss = self.compute_loss(logits, targets, loss_fn)  # (B,); or a tuple of (B,) and (B,)

        return loss, logits

    def extract_vfeats(self, vis, for_loop, vis_mask=None):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        if vis_mask is not None:
            vis_mask = vis_mask.permute(0, 1, 3, 2, 4, 5)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis, _ = self.vfeat_extractor(vis, for_loop=for_loop, cont_mask=vis_mask)
        return vis

    def extract_afeats(self, aud, for_loop, aud_mask=None):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        if aud_mask is not None:
            aud_mask = aud_mask.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud, for_loop=for_loop, cont_mask=aud_mask)
        return aud

    def compute_loss(self, logits, targets, loss_fn: str = None):
        loss = None
        if targets is not None:
            if loss_fn is None or loss_fn == 'cross_entropy':
                # logits: (B, cls) and targets: (B,)
                loss = F.cross_entropy(logits, targets)
            else:
                raise NotImplementedError(f'Loss {loss_fn} not implemented')
        return loss

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        ''' Overriding the default load_state_dict to allow loading a state dict with longer sequence.'''
        if 'transformer.pos_emb_cfg.pos_emb' in sd:
            # get the weight length from the state dict
            weight_len = sd['transformer.pos_emb_cfg.pos_emb'].shape[1]
            # get the weight length from the current model
            self_len = self.transformer.pos_emb_cfg.pos_emb.shape[1]
            # trim the weights if the state dict is longer than the current model
            if weight_len > self_len:
                sd['transformer.pos_emb_cfg.pos_emb'] = sd['transformer.pos_emb_cfg.pos_emb'][:, :self_len, :]
                logging.warning(f'Trimming the state dict for pos emb from {weight_len} to {self_len}')
            elif weight_len < self_len:
                raise ValueError(f'Cant load state dict with shorter seq len ({weight_len} vs {self_len})')
        return super().load_state_dict(sd, strict)


class GlobalTransformer(torch.nn.Module):
    '''Same as in SparseSync but without the selector transformers and the head'''

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd,
                 pos_emb_cfg=None, off_head_cfg=None) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        # input norm
        self.vis_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        # aux tokens
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        # whole token dropout
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        # maybe add pos emb
        if pos_emb_cfg is not None:
            # FIXME: `_cfg` suffix is confusing; kept for state_dict compatibility
            self.pos_emb_cfg = instantiate_from_config(pos_emb_cfg)
        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # pre-output norm
        self.ln_f = torch.nn.LayerNorm(self.config.n_embd)
        # maybe add a head
        if off_head_cfg is not None:
            self.off_head = instantiate_from_config(off_head_cfg)

        self.apply(init_weights)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape
        B, Sa, D = a.shape
        # broadcasting special tokens to the batch size
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        mod_tok = einops.repeat(self.MOD_tok, '1 1 d -> b 1 d', b=B)
        # norm
        v, a = self.vis_in_lnorm(v), self.aud_in_lnorm(a)
        # maybe whole token dropout
        if self.tok_pdrop > 0:
            v, a = self.tok_drop_vis(v), self.tok_drop_aud(a)
        # (B, 1+Sv+1+Sa, D)
        x = torch.cat((off_tok, v, mod_tok, a), dim=1)
        # maybe add pos emb
        if hasattr(self, 'pos_emb_cfg'):
            x = self.pos_emb_cfg(x)
        # dropout -> stem -> norm
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        # maybe add heads
        if attempt_to_apply_heads and hasattr(self, 'off_head'):
            x = self.off_head(x[:, 0, :])
        return x


class GlobalTransformerWithSyncabilityHead(GlobalTransformer):

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd,
                 pos_emb_cfg=None, off_head_cfg=None) -> None:
        super().__init__(tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd, pos_emb_cfg,
                         off_head_cfg)
        # remove the off_head from the parent class
        self.off_head = torch.nn.Identity()  # this class is used only during ftuning so this is not needed
        self.sync_head = torch.nn.Linear(self.config.n_embd, 2)
        self.apply(init_weights)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        x = super().forward(v, a, targets, attempt_to_apply_heads=False)
        logits_sync = self.sync_head(x[:, 0, :])
        return logits_sync


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from omegaconf import OmegaConf
    from time import time

    cfg = OmegaConf.load('./configs/sync.yaml')
    cfg.training.use_half_precision = use_half_precision = False

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = instantiate_from_config(cfg.model)
    model = model.to(device)

    start_time = time()
    for i in range(3):
        vis = torch.rand(1, 125, 3, 224, 224, device=device)
        aud = torch.rand(1, 1, 257, 626, device=device)
        # cls_logits, off_logits, sync_logits = model(vis, aud)
        # inference in half precision
        with torch.cuda.amp.autocast(cfg.training.use_half_precision):
            out = model(vis, aud)
    print('Time:', round(time() - start_time, 3))
