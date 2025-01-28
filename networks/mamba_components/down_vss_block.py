import torch
import torch.nn as nn

from typing import Callable
from functools import partial

from networks.mamba_components.vss_block import VSSBlock

class DownVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand=2,
        d_conv=3,
        bias=False,
        device=None,
        dtype=None,
        conv_bias=True,
        dropout=0,
        out_channels: int = 1,
        down_srides: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.vssblock = VSSBlock(hidden_dim=out_channels,
                                drop_path=drop_path,
                                norm_layer=norm_layer, 
                                attn_drop_rate=attn_drop_rate, 
                                d_state=d_state, expand=expand, 
                                d_conv=d_conv, bias=bias, 
                                device=device, dtype=dtype, 
                                conv_bias=conv_bias, 
                                dropout=dropout, 
                                **kwargs)
        self.downsample = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=down_srides, padding=1)
    def forward(self, inputs: torch.Tensor):
        inputs_down = self.downsample(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.vssblock(inputs_down)
        return out
