import torch
import math

from networks.mamba_components.efficient_scan import EfficientScan
from networks.mamba_components.efficient_merge import EfficientMerge
from networks.mamba_components.selective_scan import SelectiveScan, SSMODE

def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    step_size = 2,
):
    
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # H * W
    ori_h, ori_w = H, W

    xs = EfficientScan.apply(x, step_size) # [B, C, H*W] -> [B, 4, C, H//w * W//w]
    
    # H//w * W//w
    H = math.ceil(H / step_size)
    W = math.ceil(W / step_size)
    
    L = H * W

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    ori_h, ori_w = int(ori_h), int(ori_w)
    y = EfficientMerge.apply(ys, ori_h, ori_w, step_size) # [B, 4, C, H//w * W//w] -> [B, C, H*W]

    H = ori_h
    W = ori_w
    L = H * W

    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

