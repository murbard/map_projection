
import torch
import torch.optim as optim
import os
from src.train import load_land_mask
from src.model import BijectiveSquareFlow
from src.utils import compute_distortion_loss
import numpy as np

def train_fixed():
    mask = load_land_mask("world_mask_highres.png")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    H, W = mask.shape
    
    # 1.356 is the Identity Optimum
    target_ar = 1.356
    target_log_ar = np.log(target_ar)
    
    print(f"Training with FIXED Aspect Ratio: {target_ar} (Log: {target_log_ar:.4f})")
    
    # Use Ultra configuration
    model = BijectiveSquareFlow(num_layers=16, hidden_dim=512, num_bins=32).to(device)
    
    # Fix the AR parameter
    with torch.no_grad():
        model.log_aspect_ratio.fill_(target_log_ar)
    model.log_aspect_ratio.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    batch_size = 16384
    
    print("Starting training...")
    for epoch in range(1001):
        optimizer.zero_grad()
        
        u = torch.rand(batch_size, device=device)
        z = torch.rand(batch_size, device=device) * 2.0 - 1.0
        v = torch.arccos(z) / torch.tensor(3.1415926535, device=device)
        inputs = torch.stack([u, v], dim=-1)
        
        u_idx = (u * (W - 1)).long().clamp(0, W - 1)
        v_idx = (v * (H - 1)).long().clamp(0, H - 1)
        is_land = mask[v_idx, u_idx]
        
        jac = model.get_jacobian(inputs)
        loss = compute_distortion_loss(jac, u, v, is_land, land_weight=50.0)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    print("Done.")

if __name__ == "__main__":
    train_fixed()
