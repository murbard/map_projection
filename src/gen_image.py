
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn as nn
from src.model import BijectiveSquareFlow

import argparse

def generate_image(model_path, source_map_path, output_path, resolution=2048, batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    # Optimized: L=38, H=35, SiLU
    model = BijectiveSquareFlow(num_layers=38, hidden_dim=35, num_bins=32, activation_cls=nn.SiLU).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model path {model_path} not found. Using random init.")

    model.eval()
    
    
    # --- Dynamic Output Sizing ---
    print("Calculating Map Bounding Box...")
    # 1. Sample coarse grid on Earth [0, 1] x [0, 1]
    # Use enough points to find the extrema
    grid_res = 100
    u_vals = torch.linspace(0, 1, grid_res, device=device)
    v_vals = torch.linspace(0, 1, grid_res, device=device)
    grid_u, grid_v = torch.meshgrid(u_vals, v_vals, indexing='xy')
    earth_grid = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1) # (N, 2)
    
    # 2. Map to Network Domain [1/3, 2/3]
    scale_factor = 1.0 / 3.0
    offset = 1.0 / 3.0
    net_inputs = earth_grid * scale_factor + offset
    
    # 3. Project Forward to get Map Coordinates
    with torch.no_grad():
        map_coords = model(net_inputs)
        
    # 4. Compute Bounds
    min_x = map_coords[:, 0].min().item()
    max_x = map_coords[:, 0].max().item()
    min_y = map_coords[:, 1].min().item()
    max_y = map_coords[:, 1].max().item()
    
    width = max_x - min_x
    height = max_y - min_y
    
    print(f"Map Bounds: X[{min_x:.4f}, {max_x:.4f}], Y[{min_y:.4f}, {max_y:.4f}]")
    print(f"Physical Size: {width:.4f} x {height:.4f}")
    
    # 5. Determine Image Resolution
    # Keep min dimension at least 2048
    # Keep min dimension to resolution arg
    min_dim = resolution
    if width < height:
        W_out = min_dim
        H_out = int(min_dim * (height / width))
    else:
        H_out = min_dim
        W_out = int(min_dim * (width / height))
        
    # Ensure divisible by 16 or something reasonable
    W_out = (W_out // 16) * 16
    H_out = (H_out // 16) * 16
        
    print(f"Output Image Resolution: {W_out} x {H_out}")
    
    # --- Generate Grid for Inverse Sampling ---
    # We want to sample the rectangle [min_x, max_x] x [min_y, max_y]
    xs = torch.linspace(min_x, max_x, W_out, device=device)
    ys = torch.linspace(min_y, max_y, H_out, device=device)
    
    # Meshgrid (H, W)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    flat_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1) # (N, 2)
    
    # Buffer for output image
    output_buffer = torch.zeros(H_out * W_out, 4, device=device)
    
    # Load Source Image (Texture)
    # Expected (C, H, W) for grid_sample
    source_img = load_source_map(source_map_path).to(device)
    
    N = flat_coords.shape[0]
    print(f"Processing {N} pixels in batches of {batch_size}...")
    
    for i in range(0, N, batch_size):
        # We need grad for Jacobian
        with torch.enable_grad():
            batch_inputs = flat_coords[i : i + batch_size].clone().requires_grad_(True)
            
            def inverse_func(inp):
                # model.inverse handles unscaling + RealNVP inverse
                return model.inverse(inp.unsqueeze(0)).squeeze(0)
            
            # Compute Jacobian: d(source)/d(target)
            # source = (u, v) in [0, 1]
            # target = (x, y) in Physical Space
            jac_inv = torch.vmap(torch.func.jacrev(inverse_func))(batch_inputs)
            
            jac_inv = jac_inv.detach()
            # Compute actual source coords
            source_coords = model.inverse(flat_coords[i : i + batch_size]).detach()
        
        # 1. Un-pad the coordinates (Network [0.1, 0.9] -> Texture [0, 1])
        # source_coords is output of model.inverse (Network Domain)
        
        # Mask valid pixels (must be within [1/3, 2/3] range approx)
        # We'll give it a tiny epsilon tolerance
        eps = 1e-4
        valid_u = (source_coords[:, 0] >= (1.0/3.0 - eps)) & (source_coords[:, 0] <= (2.0/3.0 + eps))
        valid_v = (source_coords[:, 1] >= (1.0/3.0 - eps)) & (source_coords[:, 1] <= (2.0/3.0 + eps))
        valid_mask = valid_u & valid_v
        
        # Unscale to [0, 1] for texture lookup
        # coords = input * scale + offset
        # input = (coords - offset) / scale
        # input = (coords - 1/3) * 3
        source_coords = (source_coords - (1.0/3.0)) * 3.0
        
        # --- Tissot Indicatrix (Red Caps) ---
        # Grid: 30 degrees spacing
        # u range [0, 1] -> 360 deg. 30 deg = 1/12
        # v range [0, 1] -> 180 deg. 30 deg = 1/6
        
        Nu = 24.0
        Nv = 12.0
        
        # Find nearest grid center in (u,v) space
        # This assumes the "middle of the square" logic corresponds to half-offsets
        u_src = source_coords[:, 0]
        v_src = source_coords[:, 1]
        
        u_center = (torch.floor(u_src * Nu) + 0.5) / Nu
        v_center = (torch.floor(v_src * Nv) + 0.5) / Nv
        
        # Convert to Spherical coordinates
        # Latitude phi: pi/2 - pi * v
        # Longitude lambda: 2pi * (u - 0.5)
        
        def to_sphere(u, v):
            phi = torch.tensor(np.pi / 2.0, device=device) - torch.tensor(np.pi, device=device) * v
            lam = torch.tensor(2.0 * np.pi, device=device) * (u - 0.5)
            
            x = torch.cos(phi) * torch.cos(lam)
            y = torch.cos(phi) * torch.sin(lam)
            z = torch.sin(phi)
            return torch.stack([x, y, z], dim=-1)

        P = to_sphere(u_src, v_src)
        C = to_sphere(u_center, v_center)
        
        # Dot product
        # (B, 3) dot (B, 3) -> (B,)
        dot_prod = (P * C).sum(dim=-1).clamp(-1.0, 1.0)
        dist_rad = torch.arccos(dot_prod)
        
        # Radius: Fixed 600km? Earth radius ~6371km. 
        # Radius: Reduced by half per user request
        radius = 0.04 
        tissot_mask = dist_rad < radius
        
        # Sample Texture
        # Map u, v to pixel coords
        # source_img is (C, H, W)
        _, ht, wt = source_img.shape
        tex_x = (u_src * wt).long().clamp(0, wt - 1)
        tex_y = (v_src * ht).long().clamp(0, ht - 1)
        
        # Need to gather colors from source_img (C, H, W) -> (H, W, C) for indexing
        # Or use grid_sample as before. Let's stick to grid_sample for consistency and interpolation.
        
        source_coords_clamped = torch.clamp(source_coords, 0.0, 1.0)
        # grid_sample needs [-1, 1]
        grid_coord = (source_coords_clamped * 2.0) - 1.0
        grid_coord = grid_coord.view(1, -1, 1, 2)
        
        # source_img is (C, H, W) -> unsqueeze -> (1, C, H, W)
        sampled = F.grid_sample(source_img.unsqueeze(0), grid_coord, align_corners=False)
        # (1, C, B, 1) -> (C, B) -> (B, C)
        colors = sampled.squeeze(0).squeeze(-1).permute(1, 0) # Renamed to colors for clarity
        
        # --- Adaptive Graticules (on valid coords) ---
        # Grid spacing (15 degrees)
        u_step = 1.0 / 24.0 
        v_step = 1.0 / 12.0
        
        # Gradient magnitudes
        grad_u_mag = torch.norm(jac_inv[:, 0, :], dim=1)
        grad_v_mag = torch.norm(jac_inv[:, 1, :], dim=1)
        
        # Target thickness in Pixels
        target_px = 1.5
        
        # Pixel size in physical space
        px_size = width / W_out
        
        thresh_u = target_px * px_size * grad_u_mag
        thresh_v = target_px * px_size * grad_v_mag
        
        # Distance to grid lines
        u_nearest = torch.round(u_src / u_step) * u_step
        v_nearest = torch.round(v_src / v_step) * v_step
        
        dist_u = torch.abs(u_src - u_nearest)
        dist_v = torch.abs(v_src - v_nearest)
        
        on_grid = (dist_u < thresh_u) | (dist_v < thresh_v)
        
        # Apply graticules (Black lines)
        # on_grid is (B,)
        on_grid_expanded = on_grid.unsqueeze(-1)
        colors = torch.where(on_grid_expanded, torch.tensor(0.0, device=device), colors)
        
        # Apply Tissot Indicatrix (Red Overlay 50%)
        # tissot_mask is (B,)
        # Red: (1, 0, 0)
        red_color = torch.tensor([1.0, 0.0, 0.0], device=device)
        tissot_expanded = tissot_mask.unsqueeze(-1)
        
        # Blend: 0.5 * Original + 0.5 * Red
        colors = torch.where(tissot_expanded, colors * 0.5 + red_color * 0.5, colors)
        
        # Add Alpha Channel
        alpha = valid_mask.float().unsqueeze(-1) # 1.0 for valid, 0.0 for invalid
        
        # Combine [R, G, B, A]
        sampled_rgba = torch.cat([colors, alpha], dim=1) # (B, 4)
        
        # Force invalid pixels to transparent black
        sampled_rgba[:, :3] = sampled_rgba[:, :3] * alpha
        
        output_buffer[i : i + batch_size] = sampled_rgba
        
        if i % (batch_size * 50) == 0:
            print(f"Processed {i}/{N}...")

    # Save
    img_final = output_buffer.view(H_out, W_out, 4).cpu().numpy()
    img_final = (np.clip(img_final, 0, 1) * 255).astype(np.uint8)
    
    print(f"Saved {output_path}")
    plt.imsave(output_path, img_final)

def load_source_map(path):
    img = Image.open(path).convert('RGB')
    # Return (C, H, W)
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Map Projection")
    parser.add_argument("--model", type=str, default="world_results_optimized/model_latest.pth", help="Path to model checkpoint")
    parser.add_argument("--source", type=str, default="light_map.png", help="Path to source map texture")
    parser.add_argument("--output", type=str, default="final_map_optimized_tissot.png", help="Output filename")
    parser.add_argument("--resolution", type=int, default=2048, help="Output resolution (min dimension)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    
    args = parser.parse_args()
    
    generate_image(
        args.model,
        args.source,
        args.output,
        resolution=args.resolution,
        batch_size=args.batch_size
    )
