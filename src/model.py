
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask_type, activation_cls=nn.Softplus):
        super().__init__()
        self.mask_type = mask_type  # 0 or 1, determines which half is transformed
        
        # Scale and translation networks
        # Input to nets is half the dimension (1 for 2D inputs)
        # Output is also half the dimension * 2 (s and t)
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            activation_cls(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_cls(),
            nn.Linear(hidden_dim, 2) # Output s and t
        )
        
        # Initialize zero to start as identity (important for stability)
        # Last layer weights/bias 0 -> s=0 (scale=1), t=0
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        # x: (B, 2)
        
        # Split based on mask
        if self.mask_type == 0:
            # Identity on x[:, 0], transform x[:, 1]
            z_id = x[:, 0:1]
            z_tr = x[:, 1:2]
        else:
            # Identity on x[:, 1], transform x[:, 0]
            z_id = x[:, 1:2]
            z_tr = x[:, 0:1]
            
        # Predict parameters based on identity part
        # s (scale), t (translation)
        st = self.net(z_id)
        s, t = st.chunk(2, dim=1)
        
        # Constrain scaling for stability
        # Tanh allows range [-1, 1], so exp(s) in [1/e, e]
        s = torch.tanh(s)
        
        if not inverse:
            # Forward: y = x * exp(s) + t
            z_tr_out = z_tr * torch.exp(s) + t
        else:
            # Inverse: x = (y - t) * exp(-s)
            z_tr_out = (z_tr - t) * torch.exp(-s)
            
        # Reassemble
        if self.mask_type == 0:
            out = torch.cat([z_id, z_tr_out], dim=1)
        else:
            out = torch.cat([z_tr_out, z_id], dim=1)
            
        return out

    def get_jacobian_and_forward(self, x):
        # Returns (Jacobian (B, 2, 2), Output (B, 2))
        
        # Split
        if self.mask_type == 0:
            z_id = x[:, 0:1]
            z_tr = x[:, 1:2]
        else:
            z_id = x[:, 1:2]
            z_tr = x[:, 0:1]
            
        # We need derivatives of s and t w.r.t z_id
        # Since net is small (1 -> hidden -> 2), we can use vmap(jacrev) here
        # or just autograd.grad efficiently if we structure it right.
        # But vmap(jacrev) on the small net is very cheap compared to whole model.
        
        # Define wrapper for net
        def net_func(inp):
            return self.net(inp) # inp (1), out (2)
            
        # Jac: (B, 2, 1)
        # z_id is (B, 1)
        # net_out is (B, 2)
        # We want d(net_out)/d(z_id).
        
        # To avoid re-running forward pass redundantly, we can carry it?
        # But jacrev usually re-runs. That's fine for small net.
        
        # Optim: use torch.func.jacrev
        # Input to jacrev must be single sample (1). Output (2).
        
        # But wait, we need 'st' (s and t) for the forward pass output as well.
        # Running jacrev will give us grads, but we also need values.
        # Let's run forward pass first to get values, then jacobian?
        # Or use functional call that returns both?
        
        # Let's just run forward pass normally for 'out', 
        # and run jacrev for derivatives.
        
        
        # Optimize: Use autograd.grad instead of vmap(jacrev)
        # Since samples are independent, d(sum(s))/d(z) gives ds/dz for each sample.
        # This is compatible with checkpointing.
        
        # We need z_id to require grad for autograd.grad to work
        # If it doesn't (input leaf?), we might need to enable it.
        # But z comes from x which requires grad in get_jacobian inputs?
        # Actually in check_memory.py we set inputs.requires_grad=True.
        # But inside checkpoint, the input is a detached tensor with requires_grad=True (managed by checkpoint).
        
        # 1. Forward Pass
        st = self.net(z_id)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) # (B, 1)
        
        # 2. Compute Gradients
        # ds/dz
        ds_dz = torch.autograd.grad(
            s, z_id, 
            grad_outputs=torch.ones_like(s), 
            create_graph=torch.is_grad_enabled(), 
            retain_graph=torch.is_grad_enabled() 
        )[0]
        
        # dt/dz
        dt_dz = torch.autograd.grad(
            t, z_id, 
            grad_outputs=torch.ones_like(t), 
            create_graph=torch.is_grad_enabled()
        )[0]
        
        # 3. Compute Output and Jacobian
        exp_s = torch.exp(s)
        
        if self.mask_type == 0:
            # z_id = z0, z_tr = z1
            
            # y1 = z1 * exp_s + t
            z_tr_out = z_tr * exp_s + t
            out = torch.cat([z_id, z_tr_out], dim=1)
            
            # Gradients
            # J_00 = 1, J_01 = 0
            # J_10 = z_tr * exp_s * ds_dz + dt_dz
            # J_11 = exp_s
            
            batch_size = x.shape[0]
            J = torch.zeros(batch_size, 2, 2, device=x.device)
            # Row 0
            J[:, 0, 0] = 1.0
            # Row 1
            J[:, 1, 0] = (z_tr * exp_s * ds_dz + dt_dz).squeeze()
            J[:, 1, 1] = exp_s.squeeze()
            
        else:
            # z_id = z1, z_tr = z0
            # y0 = z0 * exp(s(z1)) + t(z1)
            # y1 = z1
            
            z_tr_out = z_tr * exp_s + t
            out = torch.cat([z_tr_out, z_id], dim=1)
            
            # Gradients
            # J_00 = dy0/dz0 = exp(s)
            # J_01 = dy0/dz1 = z0 * exp_s * ds_dz + dt_dz
            # J_10 = 0
            # J_11 = 1
            
            batch_size = x.shape[0]
            J = torch.zeros(batch_size, 2, 2, device=x.device)
            # Row 0
            J[:, 0, 0] = exp_s.squeeze()
            J[:, 0, 1] = (z_tr * exp_s * ds_dz + dt_dz).squeeze()
            # Row 1
            J[:, 1, 1] = 1.0
            
        return J, out

import torch.utils.checkpoint

class RealNVP(nn.Module):
    def __init__(self, num_layers=24, hidden_dim=1024, num_bins=None, activation_cls=nn.Softplus):
        # num_bins is unused for RealNVP but kept for compatibility with existing calls
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate mask type
            self.layers.append(AffineCouplingLayer(2, hidden_dim, i % 2, activation_cls=activation_cls))
            
        # Learnable log aspect ratio parameter
        # Initialized to 0.0 (Aspect Ratio 1.0) - THE CANARY
        self.log_aspect_ratio = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x is in [0, 1]
        
        # 1. Logit Transform: [0, 1] -> Real
        # Add epsilon to avoid inf
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        z = torch.log(x) - torch.log(1.0 - x)
        
        # 2. Coupling Layers
        for layer in self.layers:
            z = layer(z, inverse=False)
            
        # 3. Sigmoid Transform: Real -> [0, 1]
        out = torch.sigmoid(z)
        
        # 4. Aspect Ratio Scaling (preserving Area)
        # AR = exp(lambda)
        # x_scale = sqrt(AR) = exp(0.5 * lambda)
        # y_scale = 1/sqrt(AR) = exp(-0.5 * lambda)
        
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        # Apply scaling
        # out[:, 0] is x/u, out[:, 1] is y/v
        out_x = out[:, 0:1] * scale_x
        out_y = out[:, 1:2] * scale_y
        
        return torch.cat([out_x, out_y], dim=1)

    def inverse(self, y):
        # y is in output rectangle [0, W] x [0, H]
        
        # 1. Inverse Aspect Ratio Scaling
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        z_x = y[:, 0:1] / scale_x
        z_y = y[:, 1:2] / scale_y
        z = torch.cat([z_x, z_y], dim=1) # in [0, 1]
        
        # 2. Inverse Sigmoid (Logit)
        z = torch.clamp(z, 1e-6, 1.0 - 1e-6)
        z = torch.log(z) - torch.log(1.0 - z) # Map to Real
        
        # 3. Inverse Coupling Layers (Reverse order)
        for layer in reversed(self.layers):
            z = layer(z, inverse=True)
            
        # 4. Inverse Logit (Sigmoid)
        out = torch.sigmoid(z)
        
        return out

    def get_jacobian(self, x):
        # x: (B, 2)
        # Compute Jacobian layer by layer to save memory
        # J_total = J_last @ ... @ J_first
        
        batch_size = x.shape[0]
        
        # Start with Identity (B, 2, 2)
        J_total = torch.eye(2, device=x.device).unsqueeze(0).expand(batch_size, 2, 2)
        
        z = x
        
        # 1. Logit Transform Jacobian
        # z = log(x) - log(1-x)
        # dz/dx = 1/x + 1/(1-x) = 1/(x(1-x))
        # Diagonal matrix
        # Clamp x for stability (same as forward)
        x_safe = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        z = torch.log(x_safe) - torch.log(1.0 - x_safe)
        
        deriv_logit = 1.0 / (x_safe * (1.0 - x_safe)) # (B, 2)
        J_logit = torch.diag_embed(deriv_logit) # (B, 2, 2)
        
        J_total = torch.bmm(J_logit, J_total) # J_logit @ Identity
        
        # 2. Coupling Layers
        for layer in self.layers:
            # Get layer Jacobian and output z
            J_layer, z_new = layer.get_jacobian_and_forward(z)
            # Chain Rule: J_new = J_layer @ J_current
            J_total = torch.bmm(J_layer, J_total)
            z = z_new
            
        # 3. Sigmoid Transform Jacobian
        # out = sigmoid(z)
        # dout/dz = sigmoid(z) * (1 - sigmoid(z))
        sig_z = torch.sigmoid(z)
        deriv_sigmoid = sig_z * (1.0 - sig_z)
        J_sigmoid = torch.diag_embed(deriv_sigmoid)
        
        J_total = torch.bmm(J_sigmoid, J_total)
        
        # 4. Aspect Ratio Scaling
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        # Diagonal scaling matrix
        J_scale = torch.zeros(batch_size, 2, 2, device=x.device)
        J_scale[:, 0, 0] = scale_x
        J_scale[:, 1, 1] = scale_y
        
        J_total = torch.bmm(J_scale, J_total)
        
        return J_total

# Alias for compatibility with training script
BijectiveSquareFlow = RealNVP

class IdentityAR(nn.Module):
    def __init__(self, **kwargs):
        # Ignore args
        super().__init__()
        # Initialize to 0.0 (Ratio 1.0)
        self.log_aspect_ratio = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x is in [0, 1]
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        out_x = x[:, 0:1] * scale_x
        out_y = x[:, 1:2] * scale_y
        
        return torch.cat([out_x, out_y], dim=1)
        
    def inverse(self, y):
        # y is in output rectangle
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        z_x = y[:, 0:1] / scale_x
        z_y = y[:, 1:2] / scale_y
        
        return torch.cat([z_x, z_y], dim=1)
    
    def get_jacobian(self, x):
        # Jacobian is constant diagonal: [scale_x, 0; 0, scale_y]
        batch_size = x.shape[0]
        scale_x = torch.exp(0.5 * self.log_aspect_ratio)
        scale_y = torch.exp(-0.5 * self.log_aspect_ratio)
        
        j00 = scale_x.expand(batch_size)
        j11 = scale_y.expand(batch_size)
        
        zeros = torch.zeros_like(j00)
        
        # [[s_x, 0], [0, s_y]]
        row1 = torch.stack([j00, zeros], dim=-1)
        row2 = torch.stack([zeros, j11], dim=-1)
        
        # Shape (B, 2, 2)
        jac = torch.stack([row1, row2], dim=-2)
        return jac
