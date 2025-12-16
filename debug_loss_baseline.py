
import torch
import numpy as np
from src.utils import compute_distortion_loss

def check_identity_loss():
    # Simulate batch of points uniformly sampled on sphere (like train.py)
    batch_size = 10000
    u = torch.rand(batch_size)
    
    # Area preserving v sampling
    z = torch.rand(batch_size) * 2 - 1
    theta = torch.arccos(z)
    v = theta / np.pi
    
    # Identity Flow Jacobian: Identity Matrix
    # f(u,v) = (u,v) => J = I
    jacobian_flow = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Dummy land mask (all ocean, weight 1.0)
    is_land = torch.zeros(batch_size)
    
    print("Computing Loss for Uniformly Sampled Sphere (Area-Weighted)...")
    loss = compute_distortion_loss(jacobian_flow, u, v, is_land)
    print(f"Identity Loss (Uniform Sphere): {loss.item():.6f}")

    # Simulate Mesh Grid (Uniform UV)
    # This corresponds to implicit 'mean' without area weights
    print("\nComputing Loss for Uniform Grid (Over-weighted Poles, implicit mean)...")
    u_grid = torch.linspace(0, 1, 100)
    v_grid = torch.linspace(0, 1, 100)
    u_g, v_g = torch.meshgrid(u_grid, v_grid, indexing='xy')
    u_flat = u_g.flatten()
    v_flat = v_g.flatten()
    
    # Jacobian I
    jac_grid = torch.eye(2).unsqueeze(0).repeat(len(u_flat), 1, 1)
    is_land_grid = torch.zeros(len(u_flat))
    
    loss_grid = compute_distortion_loss(jac_grid, u_flat, v_flat, is_land_grid)
    print(f"Identity Loss (Uniform UV Grid): {loss_grid.item():.6f}")
    
    # Check what happens with my weights?
    # No, I can't check weights easily here without reimplementing the integral logic.
    # But the first case (Uniform Sphere) IS mathematically equivalent to "Grid + Area Weights".
    
if __name__ == "__main__":
    check_identity_loss()
