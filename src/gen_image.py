
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
    model = BijectiveSquareFlow(num_layers=16, hidden_dim=512, num_bins=32).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model path {model_path} not found. Using random init.")

    model.eval()
    
    # Check Aspect Ratio
    if hasattr(model, 'log_aspect_ratio'):
        ar_val = torch.exp(model.log_aspect_ratio).item()
        print(f"Model Learned Aspect Ratio: {ar_val:.4f}")
    else:
        ar_val = 1.0
        print("Model has no aspect ratio parameter, using 1.0")

    # Load Source Image (Texture)
    # Expected (C, H, W) for grid_sample
    source_img = load_source_map(source_map_path).to(device)
    
    # Determine Output Dimensions based on AR
    # W * H = R^2
    # W / H = AR
    # W = R * sqrt(AR)
    # H = R / sqrt(AR)
    W_out = int(resolution * np.sqrt(ar_val))
    H_out = int(resolution / np.sqrt(ar_val))
    print(f"Generating Output Image: {W_out} x {H_out}")
    
    # Create coordinate grid in target "physical" space
    # The domain model outputs to is [0, sqrt(AR)] x [0, 1/sqrt(AR)] (scaled)
    # Or rather, model outputs [0, 1] then we scale it.
    # So we should create grid in Scaled Space?
    # Our `inverse` function expects Scaled Space inputs if we implemented it right.
    # Checking model.py:
    # Scale_x = exp(0.5*lambda).
    # Forward: out = sigmoid(z) * scale.
    # Inverse: in = y / scale; z = logit(in).
    # So yes, inputs to inverse should be in [0, scale_x] x [0, scale_y].
    
    scale_x = np.sqrt(ar_val)
    scale_y = 1.0 / np.sqrt(ar_val)
    
    y_coords = torch.linspace(0, scale_y, H_out, device=device)
    x_coords = torch.linspace(0, scale_x, W_out, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    flat_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1) # (N, 2)
    
    # Buffer for output image
    output_buffer = torch.zeros(H_out * W_out, 3, device=device)
    
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
        
        # --- Adaptive Graticules ---
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
        # x range is scale_x, covered by W_out pixels.
        px_size = scale_x / W_out
        
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
        
        output_buffer[i : i + batch_size] = sampled
        
        if i % (batch_size * 50) == 0:
            print(f"Processed {i}/{N}...")

    # Save
    img_final = output_buffer.view(H_out, W_out, 3).cpu().numpy()
    img_final = (np.clip(img_final, 0, 1) * 255).astype(np.uint8)
    
    output_file = "final_map_ultra_deep.png"
    print(f"Saved {output_file}")
    plt.imsave(output_file, img_final)

def load_source_map(path):
    img = Image.open(path).convert('RGB')
    # Return (C, H, W)
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)



if __name__ == "__main__":
    generate_image(
        "world_results_ultra/model_final.pth",
        "equi_sat_8000x2000.png", 
        resolution=4096, 
        batch_size=2048 # Lower batch size since H=512 increases VRAM usage significantly
    )
