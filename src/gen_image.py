
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from src.model import BijectiveSquareFlow

def generate_image(model_path, source_map_path, resolution=2048, batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    # RealNVP with hidden=512, layers=16 (Ultra Scale)
    model = BijectiveSquareFlow(num_bins=32).to(device)
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
    
    # 2. Map to Network Domain [0.1, 0.9]
    net_inputs = earth_grid * 0.8 + 0.1
    
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
    min_dim = 2048
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
        
        # Check for transparency (Out of bounds [0.1, 0.9])
        # We need a mask for pixels that fell outside the Earth domain
        valid_mask = (source_coords[:, 0] >= 0.1) & (source_coords[:, 0] <= 0.9) & \
                     (source_coords[:, 1] >= 0.1) & (source_coords[:, 1] <= 0.9)
                     
        # Unscale valid coords
        # u_tex = (u_net - 0.1) / 0.8
        source_coords = (source_coords - 0.1) / 0.8
        
        # --- Adaptive Graticules (on valid coords) ---
        u_src = source_coords[:, 0]
        v_src = source_coords[:, 1]
        
        # Grid spacing (15 degrees)
        u_step = 1.0 / 24.0 
        v_step = 1.0 / 12.0
        
        # Gradient magnitudes: length of gradient vector in target space?
        # No, J = d(u)/d(x). J is gradient of u in target space.
        # Its magnitude tells us how fast u changes per pixel unit.
        grad_u_mag = torch.norm(jac_inv[:, 0, :], dim=1)
        grad_v_mag = torch.norm(jac_inv[:, 1, :], dim=1)
        
        # Target thickness in Pixels
        target_px = 1.5
        
        # Pixel size in physical space
        # width is the map width (Physical Size)
        px_size = width / W_out
        
        # Threshold in u-units = target_px_width * (du/d_px)
        # du/d_px = du/d_dist * d_dist/d_px = |grad u| * px_size
        thresh_u = target_px * px_size * grad_u_mag
        thresh_v = target_px * px_size * grad_v_mag
        
        # Distance to grid lines
        # d = min(u%step, step - u%step)
        dist_u = u_src % u_step
        dist_u = torch.minimum(dist_u, u_step - dist_u)
        
        dist_v = v_src % v_step
        dist_v = torch.minimum(dist_v, v_step - dist_v)
        
        # Check proximity
        on_meridian = dist_u < (thresh_u * 0.5)
        on_parallel = dist_v < (thresh_v * 0.5)
        on_grid = torch.logical_or(on_meridian, on_parallel)
        
        # --- Sampling ---
        source_coords = torch.clamp(source_coords, 0.0, 1.0)
        # grid_sample needs [-1, 1]
        grid_coord = (source_coords * 2.0) - 1.0
        grid_coord = grid_coord.view(1, -1, 1, 2)
        
        # source_img is (C, H, W) -> unsqueeze -> (1, C, H, W)
        sampled = F.grid_sample(source_img.unsqueeze(0), grid_coord, align_corners=False)
        # (1, C, B, 1) -> (C, B) -> (B, C)
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        
        # Apply graticules (Invert Color)
        on_grid_expanded = on_grid.unsqueeze(-1).expand_as(sampled)
        sampled = torch.where(on_grid_expanded, 1.0 - sampled, sampled)
        
        # Add Alpha Channel
        # sampled is (B, 3). Create alpha (B, 1)
        alpha = valid_mask.float().unsqueeze(-1) # 1.0 for valid, 0.0 for invalid
        
        # Combine [R, G, B, A]
        sampled_rgba = torch.cat([sampled, alpha], dim=1) # (B, 4)
        
        # Force invalid pixels to transparent black (or just transparent)
        # sampled_rgba = sampled_rgba * alpha # Pre-multiply alpha? 
        # Usually PNG handles unmultiplied. But let's zero out RGB for clean look.
        sampled_rgba[:, :3] = sampled_rgba[:, :3] * alpha
        
        output_buffer[i : i + batch_size] = sampled_rgba
        
        if i % (batch_size * 50) == 0:
            print(f"Processed {i}/{N}...")

    # Save
    img_final = output_buffer.view(H_out, W_out, 4).cpu().numpy()
    img_final = (np.clip(img_final, 0, 1) * 255).astype(np.uint8)
    
    output_file = "final_map_latest.png"
    print(f"Saved {output_file}")
    plt.imsave(output_file, img_final)

def load_source_map(path):
    img = Image.open(path).convert('RGB')
    # Return (C, H, W)
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)



if __name__ == "__main__":
    generate_image(
        "world_results_massive/model_latest.pth",
        "light_map.png", 
        resolution=4096, 
        batch_size=2048 # Lower batch size since H=512 increases VRAM usage significantly
    )
