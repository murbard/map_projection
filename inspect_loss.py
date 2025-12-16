
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.train import load_land_mask
from src.utils import compute_distortion_loss

def main():
    mask_path = "world_mask_highres.png"
    # Load mask
    mask = load_land_mask(mask_path, size=2048) # Higher res
    H, W = mask.shape
    
    print(f"Mask Shape: {H}x{W}")
    print(f"Mask Land Percentage: {mask.mean()*100:.2f}%")
    
    # Check Poles
    # Top rows and Bottom rows
    north_pole_land = mask[0:10, :].mean()
    south_pole_land = mask[-10:, :].mean()
    print(f"North Pole Land Fraction: {north_pole_land:.4f}")
    print(f"South Pole Land Fraction: {south_pole_land:.4f}")
    
    # Compute Weighted Identity Loss (Grid Integration)
    # We want to approximate Integral[ Loss * Weight ] / Integral[ Weight ]
    
    # 1. Grid coords
    v_lin = torch.linspace(0, 1, H)
    u_lin = torch.linspace(0, 1, W)
    u_grid, v_grid = torch.meshgrid(u_lin, v_lin, indexing='xy')
    
    # 2. Area Weights (Geometry)
    # Area element dA = sin(theta) d_theta d_phi
    # theta = v * pi
    theta = v_grid * np.pi
    area_weights = torch.sin(theta)
    
    # 3. Import Weights (Land)
    # 50 for Land, 1 for Water
    land_weight_val = 50.0
    import_weights = mask * (land_weight_val - 1.0) + 1.0
    
    # 4. Total Weights
    total_weights = area_weights * import_weights
    
    # 5. Loss (Identity)
    # J_flow = I
    jac_flow = torch.eye(2).unsqueeze(0).unsqueeze(0).expand(H, W, 2, 2)
    # Reshape for function
    jac_flat = jac_flow.reshape(-1, 2, 2)
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    mask_flat = mask.flatten()
    
    # Actually locally compute loss to be explicit about weighting
    # compute_distortion_loss applies Jacobian Sphere.
    # But wait, compute_distortion_loss in utils.py ALSO computes weights internally if we don't pass them?
    # Or if we pass geometry_weights?
    
    # Let's call the function I patched in utils.py
    # But pass the 'area_weights_flat' as geometry_weights
    area_weights_flat = area_weights.flatten()
    
    loss = compute_distortion_loss(
        jac_flat, u_flat, v_flat, mask_flat, 
        land_weight=land_weight_val, 
        geometry_weights=area_weights_flat
    )
    
    print(f"Computed Weighted Identity Loss: {loss.item():.6f}")
    
    # Also compute WITHOUT Area Weights (Just Mean of Points)
    # This simulates "Uniform Grid Sampling" (Bad)
    loss_unweighted = compute_distortion_loss(
        jac_flat, u_flat, v_flat, mask_flat, 
        land_weight=land_weight_val,
        geometry_weights=None 
    )
    print(f"Computed Unweighted Grid Loss (Biased): {loss_unweighted.item():.6f}")

if __name__ == "__main__":
    main()
