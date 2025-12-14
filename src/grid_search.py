import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from src.model import RealNVP
from src.train import train_model, load_land_mask

def run_grid_search():
    # Grid Configuration
    layers_list = [32, 64, 96]
    hidden_dims = [8, 16, 32]
    
    # Common settings
    epochs = 300
    land_weight = 50.0
    batch_size = 4096
    
    # Load mask once
    mask_path = "world_mask_highres.png"
    mask = load_land_mask(mask_path, size=2048)
    
    results = []
    
    print(f"{'Layers':<10} {'Hidden':<10} {'Params':<15} {'Final Loss':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    for L in layers_list:
        for H in hidden_dims:
            save_dir = f"grid_results/L{L}_H{H}"
            
            start_time = time.time()
            
            # Train
            # We need to modify train_model slightly to return the final loss or we just parse stats?
            # actually train_model returns the model. We can capture the loss from the training loop if we hacked it,
            # but for now let's just run it and maybe rely on the logs or modify train_model to return loss.
            # To keep it simple without modifying train.py again, we will trust the loss printed or 
            # we can look at the history if we modified train_model to return it.
            # Wait, train_model currently returns 'model'. It doesn't return loss.
            # I should modify train_model to return (model, final_loss) or just use a wrapped version here.
            
            # Actually, I'll essentially verify by running validatoin/check on the returned model?
            # Or better: I will instantiate the model here and train it using a local loop?
            # No, re-using train_model is better.
            # Let's Modify train.py quickly to return history or final loss. 
            pass

if __name__ == "__main__":
    # Because I cannot easily modify train_model return signature without breaking other things potentially
    # (though I wrote it so I can), I'll just write a custom loop here that is a stripped down version of train_model
    # This avoids changing the production code for a temp script.
    
    layers_list = [32, 64, 96]
    hidden_dims = [8, 16, 32]
    batch_sizes = [4096, 16384]
    
    mask_path = "world_mask_highres.png"
    mask = load_land_mask(mask_path, size=2048)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    
    H_img, W_img = mask.shape
    
    print(f"{'Layers':<8} | {'Hidden':<8} | {'Batch':<8} | {'Params':<10} | {'Loss':<10} | {'Time':<8}")
    print("-" * 65)

    for L in layers_list:
        for H in hidden_dims:
            for B in batch_sizes:
                torch.cuda.empty_cache()
                
                # Setup Model
                model = RealNVP(num_layers=L, hidden_dim=H, num_bins=32).to(device)
                
                # Count params
                num_params = sum(p.numel() for p in model.parameters())
                
                # Optimizer
                optimizer = optim.Adam([
                    {'params': model.layers.parameters(), 'lr': 2e-5}, 
                    {'params': [model.log_aspect_ratio], 'lr': 0.01}
                ])
                
                start_t = time.time()
                final_loss = 0.0
                
                model.train()
                
                # Short training loop
                # Adjust steps based on batch size to see fair comparison?
                # Or just keep epochs constant? Epochs constant = same number of "passes" over data distribution.
                # But since we sample randomly, an "epoch" is just an iteration here.
                # Let's keep iterations constant? No, let's keep "samples seen" constant?
                # Usually "Epoch" means N steps.
                # Let's run for 500 steps regardless of batch size? That means larger batch sees more data.
                # Fair comparison for wall-clock time vs quality.
                steps = 500
                
                for step in range(steps):
                    # Sample
                    u = torch.rand(B, device=device)
                    z = torch.rand(B, device=device) * 2.0 - 1.0
                    theta = torch.arccos(z)
                    v = theta / np.pi
                    inputs = torch.stack([u, v], dim=-1)
                    
                    # Center 1/3 Scaling (Hardcoded here to match current experiment)
                    scale_factor = 1.0/3.0
                    inputs_net = inputs * scale_factor + (1.0/3.0)
                    inputs_net.requires_grad_(True)
                    
                    # Land mask
                    u_idx = (u * (W_img - 1)).long().clamp(0, W_img - 1)
                    v_idx = (v * (H_img - 1)).long().clamp(0, H_img - 1)
                    is_land = mask[v_idx, u_idx]
                    
                    optimizer.zero_grad()
                    
                    # Forward + Jacobian
                    # We need Jacobian of flow wrt inputs_net
                    # But wait, Jacobian used to be vmap. Now it's analytic.
                    # model.get_jacobian(inputs_net) returns d(out)/d(in_net)
                    jacobian_flow = model.get_jacobian(inputs_net)
                    
                    # Adjust for input scaling
                    jacobian_flow = jacobian_flow * scale_factor
                    
                    # Sphere Jacobian
                    from src.utils import compute_sphere_jacobian_to_equirectangular
                    jacobian_sphere = compute_sphere_jacobian_to_equirectangular(u, v)
                    
                    # Total
                    j_total = torch.bmm(jacobian_flow, jacobian_sphere)
                    
                    # Loss
                    _, s, _ = torch.linalg.svd(j_total)
                    log_s = torch.log(s + 1e-8)
                    loss_per_point = torch.sum(log_s**2, dim=-1)
                    
                    weights = torch.ones_like(loss_per_point)
                    weights[is_land > 0.5] = 50.0 # Land Weight
                    
                    loss = (loss_per_point * weights).mean()
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    final_loss = loss.item()
                
                
            elapsed = time.time() - start_t
            print(f"{L:<8} | {H:<8} | {B:<8} | {num_params:<10} | {final_loss:.4f}     | {elapsed:.1f}s")
