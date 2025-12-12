
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask_type):
        super().__init__()
        self.mask_type = mask_type  # 0 or 1, determines which half is transformed
        
        # Scale and translation networks
        # Input to nets is half the dimension (1 for 2D inputs)
        # Output is also half the dimension * 2 (s and t)
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

class RealNVP(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=128, num_bins=None):
        # num_bins is unused for RealNVP but kept for compatibility with existing calls
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate mask type
            self.layers.append(AffineCouplingLayer(2, hidden_dim, i % 2))
            
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
        # Same logic as before, using vmap and jacrev
        
        def flow_func(inp):
            out = self.forward(inp.unsqueeze(0))
            return out.squeeze(0)

        batch_jac = torch.vmap(torch.func.jacrev(flow_func))(x)
        return batch_jac

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
