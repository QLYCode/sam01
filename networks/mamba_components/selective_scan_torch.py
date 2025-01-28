import torch
import torch.nn.functional as F

class SelectiveScanPyTorch:
    @staticmethod
    def fwd(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        batch, dim, length = u.shape  # [6, 256, 16384]
        states = A.shape[1]  # 16
        groups = B.shape[1]  # 4
        channels_per_group = dim // groups  # 256 // 4 = 64
        
        if delta_bias is not None:
            delta = delta + delta_bias.view(1, -1, 1)
        if delta_softplus:
            delta = F.softplus(delta)
        
        out = torch.zeros_like(u)
        x = torch.zeros(batch, groups, states, device=u.device, dtype=u.dtype)
        
        delta = delta.view(batch, groups, channels_per_group, length)
        u = u.view(batch, groups, channels_per_group, length)
        A = A.view(groups, channels_per_group, states)
        
        for t in range(length):
            delta_t = delta[..., t].mean(dim=2, keepdim=True)
            x = x * torch.exp(-delta_t)
            
            u_t = u[..., t]
            B_t = B[..., t]
            gate = torch.einsum('bgc,bgs->bgs', u_t, B_t)
            x = x + torch.einsum('bgs,gcs->bgs', gate, A)
            
            C_t = C[..., t]
            y = torch.einsum('bgs,bgs->bg', x, C_t)
            
            # Reshape y to match original dimensions before skip connection
            y = y.repeat_interleave(channels_per_group, dim=1)
            if D is not None:
                y = y + D * u[..., t].view(batch, -1)
            
            out[..., t] = y
            
        return out, x, None
    
    @staticmethod
    def bwd(u, delta, A, B, C, D, delta_bias, dout, x_last, delta_softplus, nrows):
        batch, dim, length = u.shape
        groups = B.shape[1]
        channels_per_group = dim // groups
        states = A.shape[1]
        
        # Recompute forward pass and save intermediates
        if delta_bias is not None:
            delta_modified_before_softplus = delta + delta_bias.view(1, -1, 1)
        else:
            delta_modified_before_softplus = delta.clone()
        if delta_softplus:
            delta_modified = F.softplus(delta_modified_before_softplus)
        else:
            delta_modified = delta_modified_before_softplus
        
        delta_modified = delta_modified.view(batch, groups, channels_per_group, length)
        u = u.view(batch, groups, channels_per_group, length)
        A = A.view(groups, channels_per_group, states)
        
        x_list = []
        delta_t_list = []
        gate_list = []
        u_list = []
        B_list = []
        C_list = []
        
        x = torch.zeros(batch, groups, states, device=u.device, dtype=u.dtype)
        for t in range(length):
            delta_t = delta_modified[..., t].mean(dim=2, keepdim=True)
            delta_t_list.append(delta_t)
            
            x_prev = x.clone()
            x = x_prev * torch.exp(-delta_t)
            
            u_t = u[..., t]
            u_list.append(u_t)
            B_t = B[..., t]
            B_list.append(B_t)
            gate = torch.einsum('bgc,bgs->bgs', u_t, B_t)
            gate_list.append(gate)
            
            x = x + torch.einsum('bgs,gcs->bgs', gate, A)
            C_t = C[..., t]
            C_list.append(C_t)
            
            x_list.append(x.clone())
        
        # Initialize gradients
        du = torch.zeros_like(u)
        ddelta = torch.zeros_like(delta_modified)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D) if D is not None else None
        ddelta_bias = torch.zeros_like(delta_bias) if delta_bias is not None else None
        
        sum_A = A.sum(dim=1)  # (groups, states)
        dx_next = torch.zeros_like(x_last)
        
        for t in reversed(range(length)):
            x_t = x_list[t]
            delta_t = delta_t_list[t]
            u_t = u_list[t]
            B_t = B_list[t]
            gate_t = gate_list[t]
            C_t = C_list[t]
            
            # Compute dy_t from dout
            dout_t = dout[..., t]
            dout_t_reshaped = dout_t.view(batch, groups, channels_per_group)
            dy_t = dout_t_reshaped.sum(dim=2)  # (batch, groups)
            
            dx_from_y = dy_t.unsqueeze(-1) * C_t  # (batch, groups, states)
            dx_total = dx_from_y + dx_next
            
            # Gradient for C_t
            dC[:, :, :, t] += x_t * dy_t.unsqueeze(-1)
            
            # Gradient for the gate
            dgate = dx_total * sum_A[None, :, :]
            
            # Gradient for B_t
            sum_u_t = u_t.sum(dim=2)  # (batch, groups)
            dB[:, :, :, t] = dgate * sum_u_t.unsqueeze(-1)
            
            # Gradient for u_t
            sum_B_t = (B_t * dgate).sum(dim=2)  # (batch, groups)
            du[:, :, :, t] = sum_B_t.unsqueeze(-1).expand_as(u_t)
            
            # Gradient for A
            dA_contribution = torch.einsum('bgs->gs', gate_t * dx_total).unsqueeze(1).expand(-1, channels_per_group, -1)
            dA += dA_contribution
            
            # Compute x_prev before the current time step's update
            if t > 0:
                x_prev_before_update = x_list[t-1]
            else:
                x_prev_before_update = torch.zeros_like(x_list[0])
            
            # Gradient for delta_t
            exp_neg_delta_t = torch.exp(-delta_t)
            grad_delta_t = - (x_prev_before_update * exp_neg_delta_t * dx_total).sum(dim=2)  # (batch, groups)
            
            ddelta_t = grad_delta_t.unsqueeze(2) / channels_per_group
            ddelta[:, :, :, t] += ddelta_t.expand(batch, groups, channels_per_group)
            
            # Gradient for x_prev (to be passed to previous step)
            dx_next = dx_total * exp_neg_delta_t
        
        # Gradient for D
        if D is not None:
            dout_reshaped = dout.view(batch, groups, channels_per_group, length)
            dD = torch.einsum('bgct,bgct->gc', u, dout_reshaped).view(-1)
        
        # Handle delta_bias and delta_softplus gradients
        if delta_softplus:
            sig = torch.sigmoid(delta_modified_before_softplus).view_as(ddelta)
            ddelta_modified = ddelta * sig
        else:
            ddelta_modified = ddelta
        
        # Gradient for original delta (input delta)
        ddelta = ddelta_modified.view_as(delta)
        
        # Gradient for delta_bias
        if delta_bias is not None:
            ddelta_bias = ddelta_modified.sum(dim=(0, 3)).view(-1)
        else:
            ddelta_bias = None
        
        # Reshape dA to original shape
        dA = dA.view(-1, dA.shape[-1])
        
        return du.view(batch, dim, length), ddelta.view_as(delta), dA, dB, dC, dD, ddelta_bias, None