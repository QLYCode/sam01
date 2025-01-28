from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath

from networks.mamba_components.es2d import ES2D

class VSSBlock(nn.Module):
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
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln_1 = norm_layer(hidden_dim)
        self.d_model = hidden_dim
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.es2d = ES2D(
            d_model=self.d_inner, 
            d_state=16, 
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # ==========================
            d_conv=3,
            conv_bias=True,
            # ==========================
            dropout=0,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=False,
            # ==========================
            forward_type="v2",
            step_size=2,)
        self.drop_path = DropPath(drop_path)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.res_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    def forward(self, inputs: torch.Tensor):
        

        input_LN = self.ln_1(inputs)
        B, H, W, C = inputs.shape

        xz = self.in_proj(input_LN)                                #linear
        
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)         #卷积+激活


        # y1, y2, y3, y4 = self.forward_core(x)
        # assert y1.dtype == torch.float32
        # y = y1 + y2 + y3 + y4
        y = self.es2d(x.permute(0, 2, 3, 1).contiguous())


        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #(b, h, w, d)
        y = self.out_norm(y)                                #layer norm

        y = y + self.res_proj(input_LN)                     #残差连接

        y = y * F.silu(z)
        out = self.out_proj(y)                              #linear
        if self.dropout is not None:
            out = self.dropout(out)

        out = inputs + out                                  #残差连接
        return out
