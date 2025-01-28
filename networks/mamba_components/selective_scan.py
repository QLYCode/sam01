import torch
    
SSMODE = None
try:
    import selective_scan_cuda_core
    SSMODE = "sscore"
except Exception as e:
    # print(e, flush=True)
    pass
try:
    if SSMODE == None:
        import selective_scan_cuda
        SSMODE = "mamba_ssm"
except Exception as e:
    # print(e, flush=True)
    pass
try:
    if SSMODE == None:
        from networks.mamba_components.selective_scan_torch import SelectiveScanPyTorch as selective_scan_torch
        SSMODE = "pytorch"
except Exception as e:
    # print(e, flush=True)
    pass
    exit()
# print(f"Using SSMODE: {SSMODE}")

class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cpu')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        elif SSMODE == "sscore":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        elif SSMODE == "pytorch":
            out, x, *rest = selective_scan_torch.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        else:
            raise NotImplementedError("Unknown SSMODE: " + SSMODE)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.amp.custom_bwd(device_type='cpu')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            )
        elif SSMODE == "sscore":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif SSMODE == "pytorch":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_torch.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        else:
            raise NotImplementedError("Unknown SSMODE: " + SSMODE)
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)
