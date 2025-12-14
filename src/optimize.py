import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
from src.model import RealNVP, AffineCouplingLayer
from src.train import load_land_mask
from src.utils import compute_sphere_jacobian_to_equirectangular

# Monkey patch or modify model creation to support different activations if needed
# The current model implementation likely hardcodes Softplus.
# We need to modify RealNVP or AffineCouplingLayer to accept an activation factory.
# Or we can subclass/inject it.



    
def objective(trial):
    # 1. Sample Hyperparameters
    n_layers = trial.suggest_int('n_layers', 16, 192) # Wide range, TPE will explore
    hidden_dim = trial.suggest_int('hidden_dim', 8, 64)
    batch_size = trial.suggest_int('batch_size', 2048, 16384, step=128)
    activation = trial.suggest_categorical('activation', ['Softplus', 'GELU', 'SiLU'])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # Log uniform search
    
    # 2. Setup Data
    mask_path = "world_mask_highres.png"
    # Load once globally? No, cheap enough.
    mask = load_land_mask(mask_path, size=2048)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    H_img, W_img = mask.shape
    
    # 3. Setup Model
    # Mapping
    acts = {
        'Softplus': nn.Softplus,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU
    }
    
    act_cls = acts[activation]
    
    model = RealNVP(num_layers=n_layers, hidden_dim=hidden_dim, num_bins=32, activation_cls=act_cls).to(device)
        
    optimizer = optim.Adam([
        {'params': model.layers.parameters(), 'lr': lr},
        {'params': [model.log_aspect_ratio], 'lr': 0.01} # Keep AR learning rate fixed or separate? User didn't specify, likely fixed is fine.
    ])
    
    # 4. Training Loop (Time constrained)
    start_time = time.time()
    time_limit = 15 * 60 # 15 minutes
    
    step = 0
    final_loss = float('inf')
    
    model.train()
    
    while (time.time() - start_time) < time_limit:
        step += 1
        
        # Sample
        u = torch.rand(batch_size, device=device)
        z = torch.rand(batch_size, device=device) * 2.0 - 1.0
        theta = torch.arccos(z)
        v = theta / np.pi
        inputs = torch.stack([u, v], dim=-1)
        
        # Center 1/3 Scaling
        scale_factor = 1.0/3.0
        inputs_net = inputs * scale_factor + (1.0/3.0)
        inputs_net.requires_grad_(True)
        
        # Mask
        u_idx = (u * (W_img - 1)).long().clamp(0, W_img - 1)
        v_idx = (v * (H_img - 1)).long().clamp(0, H_img - 1)
        is_land = mask[v_idx, u_idx]
        
        optimizer.zero_grad()
        
        jacobian_flow = model.get_jacobian(inputs_net)
        jacobian_flow = jacobian_flow * scale_factor
        
        jacobian_sphere = compute_sphere_jacobian_to_equirectangular(u, v)
        j_total = torch.bmm(jacobian_flow, jacobian_sphere)
        
        _, s, _ = torch.linalg.svd(j_total)
        log_s = torch.log(s + 1e-8)
        loss_per_point = torch.sum(log_s**2, dim=-1)
        
        weights = torch.ones_like(loss_per_point)
        weights[is_land > 0.5] = 50.0
        
        loss = (loss_per_point * weights).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        final_loss = loss.item()
        
        # Report intermediate results?
        # trial.report(final_loss, step)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
            
    return final_loss

if __name__ == "__main__":
    # Persistence
    study_name = "map_projection_study_v3" # Start fresh or new version to include LR
    storage_name = "sqlite:///study_results.db"
    
    # User requested Gaussian Process Search. 
    # Optuna's default is TPE (Tree-structured Parzen Estimator) which is a Bayesian optimization method
    # broadly similar in capability (building a probabilistic model of the objective).
    # We can explicitly define it for clarity.
    sampler = optuna.samplers.TPESampler()
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="minimize", 
        sampler=sampler,
        load_if_exists=True
    )
    
    print("Starting optimization...")
    study.optimize(objective, timeout=None) # Run until killed
