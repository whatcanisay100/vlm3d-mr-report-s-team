"""Frozen BrainMVP encoder wrappers.

The released checkpoint stores ``patch_embed*.norm.weight/bias`` only, which
is the LayerNorm schema used by the original patch embedding.  A previous
local wrapper replaced those layers with BatchNorm3d and silently left its
running statistics uninitialised.  ``CheckpointFaithfulBrainMVPEncoder`` is
the corrected interface and requires a strict load.  ``BrainMVPEncoder``
retains the legacy BatchNorm behavior only so frozen historical pipelines do
not silently change semantics; new code must not use it.

Forward: ``(B, 1, D, H, W) -> (B, N, 512)`` from the deepest spatial scale.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uniformer_blocks import UniFormer


class BMVPPatchEmbed(nn.Module):
    """Legacy mismatched patch embed retained for historical reproduction."""

    def __init__(self, in_ch, out_ch, stride=(2, 2, 2)):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=(2, 2, 2), stride=stride, bias=True)
        self.norm = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))


class LayerNormPatchEmbed(nn.Module):
    """Checkpoint-faithful Conv3d plus channel-wise LayerNorm patch embed."""

    def __init__(self, in_ch, out_ch, stride=(2, 2, 2)):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=(2, 2, 2), stride=stride, bias=True)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.proj(x)
        batch, channels, depth, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


class BrainMVPEncoder(nn.Module):
    """Legacy BatchNorm wrapper; do not use for new extraction."""

    def __init__(self, ckpt):
        super().__init__()
        self.uniformer = UniFormer(depth=[3, 4, 8, 3], img_size=128, in_chans=1, num_classes=0,
                                   embed_dim=[64, 128, 320, 512], conv_stem=True)
        self.uniformer.patch_embed1 = BMVPPatchEmbed(1, 64, (2, 2, 2))
        self.uniformer.patch_embed2 = BMVPPatchEmbed(64, 128, (2, 2, 2))
        self.uniformer.patch_embed3 = BMVPPatchEmbed(128, 320, (1, 2, 2))
        self.uniformer.patch_embed4 = BMVPPatchEmbed(320, 512, (1, 2, 2))
        self._load(ckpt)
        self.eval()
        for p in self.parameters(): p.requires_grad = False

    def _load(self, ckpt):
        ck = torch.load(ckpt, map_location='cpu', weights_only=True)
        sd = ck.get('state_dict', ck)
        sd = {k.replace('module.encoder.uniformer.', ''): v for k, v in sd.items()
              if k.startswith('module.encoder.uniformer.')}
        for index in range(1, 5):
            prefix = f'patch_embed{index}.'
            patch_state = {
                key.replace(prefix, ''): value
                for key, value in sd.items()
                if key.startswith(prefix)
            }
            result = getattr(self.uniformer, f'patch_embed{index}').load_state_dict(
                patch_state, strict=False
            )
            print(
                f"patch_embed{index}: missing={len(result.missing_keys)} "
                f"unexpected={len(result.unexpected_keys)}"
            )
        rest = {key: value for key, value in sd.items() if not key.startswith('patch_embed')}
        result = self.uniformer.load_state_dict(rest, strict=False)
        print(
            f"blocks/norm: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )

    def forward(self, x):
        _, _, _, _, x_enc4 = self.uniformer(x)
        return x_enc4.flatten(2).transpose(1, 2)  


class CheckpointFaithfulBrainMVPEncoder(nn.Module):
    """Strict LayerNorm wrapper for versioned corrected feature families."""

    def __init__(self, ckpt):
        super().__init__()
        self.uniformer = UniFormer(
            depth=[3, 4, 8, 3],
            img_size=128,
            in_chans=1,
            num_classes=0,
            embed_dim=[64, 128, 320, 512],
            conv_stem=True,
        )
        self.uniformer.patch_embed1 = LayerNormPatchEmbed(1, 64, (2, 2, 2))
        self.uniformer.patch_embed2 = LayerNormPatchEmbed(64, 128, (2, 2, 2))
        self.uniformer.patch_embed3 = LayerNormPatchEmbed(128, 320, (1, 2, 2))
        self.uniformer.patch_embed4 = LayerNormPatchEmbed(320, 512, (1, 2, 2))
        self._load(ckpt)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _load(self, ckpt):
        checkpoint = Path(ckpt)
        checkpoint_obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
        source = checkpoint_obj.get('state_dict', checkpoint_obj)
        prefix = 'module.encoder.uniformer.'
        state = {key[len(prefix):]: value for key, value in source.items()
                 if key.startswith(prefix)}
        if not state:
            raise RuntimeError(f'checkpoint has no keys under {prefix!r}: {checkpoint}')
        self.uniformer.load_state_dict(state, strict=True)

    def forward(self, x):
        _, _, _, _, deepest = self.uniformer(x)
        return deepest.flatten(2).transpose(1, 2)
