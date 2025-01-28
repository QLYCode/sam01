import torch
import torch.nn.functional as F
import math

class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2): # [B, C, H, W] -> [B, 4, H/w * W/w]
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))  
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H*W))


        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs
    
    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor): # [B, 4, H/w * W/w] -> [B, C, H, W]

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))
        
        grad_xs = grad_xs.view(B, 4, C, newH, newW)
        
        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None 
