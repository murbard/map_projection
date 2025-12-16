
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.mesh import DifferentiableMesh
from src.utils import compute_distortion_loss, compute_sphere_jacobian_to_equirectangular
from src.train import load_land_mask
import time

def main():
    save_dir = "mesh_results_100k"
    os.makedirs(save_dir, exist_ok=True)
    mask_path = "world_mask_highres.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Hyperparameters
    # Mesh Resolution: 100k triangles
    # 2 * (H-1) * (W-1) approx 100,000
    # Let H=160, W=320. 159*319*2 = 101,442.
    H_mesh = 160
    W_mesh = 320
    # This gives Grid Resolution. Number of triangles = 2 * (H-1) * (W-1).
    # Approx 500 triangles.
    
    # Loading Mask
    # We need mask values at the CENTER of each triangle for weighting.
    # Load mask at higher res?
    mask_high = load_land_mask(mask_path, size=2048).to(device)
    
    # Initialize Mesh
    mesh = DifferentiableMesh(H_mesh, W_mesh).to(device)
    
    # Optimizer
    # User requested switching back to Adam
    optimizer = optim.Adam(mesh.parameters(), lr=1e-4)
    
    # Auto-Resume Logic
    start_step = 0
    latest_ckpt = os.path.join(save_dir, "mesh_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        mesh.load_state_dict(checkpoint['model_state_dict'])
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # except Exception as e:
        #     print(f"Could not load optimizer state (expected if switching Optimizers): {e}")
        start_step = checkpoint['step'] + 1
        print(f"Resumed from Step {start_step}.")
        
    print("Precomputing Triangle Centers and Sphere Jacobians...")
    # u_c, v_c (N_tri)
    u_c = mesh.u_centers
    v_c = mesh.v_centers
    
    # Sample Land Mask: Area Averaging
    # Grid Dimensions (Quads)
    H_grid = H_mesh - 1
    W_grid = W_mesh - 1
    
    # Resize mask to (H_grid, W_grid) using area interpolation (avg_pool like)
    # mask_high is (H, W). Add batch/channel dims -> (1, 1, H, W)
    mask_tensor = mask_high.float().unsqueeze(0).unsqueeze(0)
    
    # Interpolate to grid size
    # 'area' mode is adaptive average pooling
    mask_down = torch.nn.functional.interpolate(mask_tensor, size=(H_grid, W_grid), mode='area')
    
    # Flatten: (1, 1, H_grid, W_grid) -> (N_quads)
    mask_flat = mask_down.reshape(-1)
    
    # Assign to Triangles
    # T1 corresponds to quads in order. T2 same.
    # is_land shape: (N_tri) -> (2 * N_quads)
    is_land = torch.cat([mask_flat, mask_flat], dim=0)
    
    print(f"Land Fraction Range: {is_land.min():.4f} to {is_land.max():.4f}")
    print(f"Global Land Percentage: {is_land.mean()*100:.2f}%")
    
    # Note: compute_distortion_loss calculates sphere Jacobian internally.
    
    # Load Texture for Preview
    texture_path = "light_map.png"
    if not os.path.exists(texture_path):
        texture_path = "equi_sat_8000x2000.png" # Fallback
    
    try:
        from PIL import Image
        tex_img = Image.open(texture_path).convert('RGB')
        tex_img = tex_img.resize((W_mesh, H_mesh), Image.Resampling.BILINEAR)
        tex_data = np.array(tex_img) / 255.0 # (H, W, 3)
        # Flatten for scatter
        tex_colors = tex_data.reshape(-1, 3)
    except Exception as e:
        print(f"Could not load texture: {e}")
        tex_colors = None

    print(f"Global Land Percentage: {is_land.mean()*100:.2f}%")
    
    # Initialize step if not resumed
    if 'step' not in locals():
        step = start_step
        optimizer.zero_grad()
        
        # 1. Get Jacobians of Mesh
        # J_mesh: (N_tri, 2, 2)
        J_mesh, E_q = mesh.get_jacobians()
        
        # 3. Distortion Loss
        # We assume compute_distortion_loss handles (N, 2, 2) input
        # It expects Jacobian, u, v, points.
        # It returns scalar loss.
        # Note: compute_distortion_loss usually computes singular values etc.
        # But it also composes with sphere jacobian internally? 
        # Let's check src/utils.py.
        # Ah, compute_distortion_loss TAKES jacobian_flow. And IT calls compute_sphere_jacobian.
        # And IT does the multiplication.
        # So we should pass J_mesh as 'jacobian_flow' and u_c, v_c as points.
        
    # Compute Exact Spherical Areas for T1 and T2
    # Grid steps
    dv = 1.0 / (H_mesh - 1)
    
    # Precompute Exact Weights for T1 and T2
    # we need vector of v coordinates for each quad
    v_rows = torch.linspace(0, 1, H_mesh).to(device)
    # T1 and T2 are repeated for each column (W_mesh-1), but weights depend only on v (row).
    # We can compute weight for row i, then repeat.
    
    # Row i corresponds to v_i. Quad spans [v_i, v_{i+1}].
    # v_min = v_rows[:-1]
    # v_max = v_rows[1:]
    
    # Integral of sin(pi * v) * width(v)
    # Width is linear.
    # Scale factors: 
    #   lambda range: 2*pi
    #   phi range: pi
    #   Total Area = 4*pi.
    # We just need relative weights proportional to area.
    # Integral sin(pi*v) dv = -1/pi cos(pi*v)
    # Integral v*sin(pi*v) dv = (sin(pi*v) - pi*v*cos(pi*v)) / pi^2
    
    def integral_sin(v):
        return -torch.cos(torch.pi * v) / torch.pi
        
    def integral_v_sin(v):
        return (torch.sin(torch.pi * v) - torch.pi * v * torch.cos(torch.pi * v)) / (torch.pi**2)
    
    v_start = v_rows[:-1]
    v_end = v_rows[1:]
    
    # T1: Width w(v') goes from 1 to 0 as v' goes from v_start to v_end.
    # w(v') = (v_end - v') / dv
    # Integral = 1/dv * [ v_end * Int(sin) - Int(v*sin) ]
    term1_t1 = v_end * (integral_sin(v_end) - integral_sin(v_start))
    term2_t1 = (integral_v_sin(v_end) - integral_v_sin(v_start))
    area_t1_rows = (term1_t1 - term2_t1) / dv
    
    # T2: Width w(v') goes from 0 to 1 as v' goes from v_start to v_end.
    # w(v') = (v' - v_start) / dv
    # Integral = 1/dv * [ Int(v*sin) - v_start * Int(sin) ]
    term1_t2 = (integral_v_sin(v_end) - integral_v_sin(v_start))
    term2_t2 = v_start * (integral_sin(v_end) - integral_sin(v_start))
    area_t2_rows = (term1_t2 - term2_t2) / dv
    
    # Expand to all quads
    # shape (H-1) -> (H-1, W-1) -> Flatten
    area_t1 = area_t1_rows.unsqueeze(1).repeat(1, W_mesh - 1).flatten()
    area_t2 = area_t2_rows.unsqueeze(1).repeat(1, W_mesh - 1).flatten()
    
    # Concatenate T1 and T2 weights (matches is_land order)
    # shape (N_tri)
    area_weights = torch.cat([area_t1, area_t2])
    
    # Normalize to keep loss scale similar?
    # Max weight is around dv * sin(pi/2)?
    # Let's keep them as "relative areas".
    
    print("Computed Exact Spherical Area Weights.")
    
    successful_steps = 0
    
    # Initialize step if not resumed
    if 'step' not in locals():
        step = start_step

    while True:
        step += 1
        optimizer.zero_grad()
        
        # 1. Get Jacobians of Mesh
        J_mesh, E_q = mesh.get_jacobians()
        
        # 2. Distortion Loss
        loss_distortion = compute_distortion_loss(J_mesh, u_c, v_c, is_land, land_weight=50.0, geometry_weights=area_weights)
        # 4. Barrier Loss (Soft Constraint)
        # Det2D = J_00 * J_11 - J_01 * J_10
        dets = J_mesh[:, 0, 0] * J_mesh[:, 1, 1] - J_mesh[:, 0, 1] * J_mesh[:, 1, 0]
        
        # Log Barrier: -log(det).
        # We also weight this by area! 
        # Polar triangles SHOULD be allowed to shrink to match their 0 sphere area.
        # Unweighted barrier forces them to be large (uniform grid size).
        barrier_term = -torch.log(torch.clamp(dets, min=1e-9))
        
        # Weighted Barrier Mean
        # Normalize sum by sum of weights + epsilon
        barrier_loss = (barrier_term * area_weights).sum() / (area_weights.sum() + 1e-8)
            # Anneal Barrier Weight
        # Start at 0.01, decay to 0 over 1,000,000 steps
        progress = min(1.0, step / 1000000.0)
        barrier_weight = 0.01 * (1.0 - progress)
        
        loss = loss_distortion + barrier_weight * barrier_loss
        
        loss.backward()
        
        # 5. Safe Step
        # Proposed update: param - lr * grad
        with torch.no_grad():
            old_params = mesh.vertices.clone()
            
            # Simple Backtracking check handled after step
            
        optimizer.step()
        
        with torch.no_grad():
            # Check collisions
            new_J, _ = mesh.get_jacobians()
            det = new_J[:, 0, 0] * new_J[:, 1, 1] - new_J[:, 0, 1] * new_J[:, 1, 0]
            
            min_det = det.min()
            if min_det < 1e-6:
                # Collision detected!
                # Revert
                mesh.vertices.data.copy_(old_params)
                
                # CRITICAL Fix for SGD: Clear momentum state on collision
                # Otherwise momentum will force the same bad update next step.
                optimizer.state.clear()
                
                if step % 100 == 0:
                     print(f"Step {step}: Crossing detected (Min Det {min_det.item():.2e}). Reducing LR.")
                
                successful_steps = 0 # Reset counter
                
                # Decay LR
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                    
                if optimizer.param_groups[0]['lr'] < 1e-8:
                    print("LR too small. Stopping.")
                
                # Don't step this time.
                continue
            
            # If we survived: Increase successful steps counter
            successful_steps += 1
            
            if successful_steps > 50:
                # Encourage speedup if stable
                old_lr = optimizer.param_groups[0]['lr']
                if old_lr < 1e-2:
                    new_lr = old_lr * 1.05
                    for g in optimizer.param_groups:
                        g['lr'] = new_lr
                    successful_steps = 0 
                
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.6f} (Barrier W: {barrier_weight:.5f})")
            
        if step % 1000 == 0:
            if tex_colors is not None:
                save_map_preview(mesh, tex_colors, save_dir, step)
            
            save_mesh_vis(mesh, save_dir, step)
            # Save Checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': mesh.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, f"mesh_{step}.pth"))
            torch.save(checkpoint, os.path.join(save_dir, "mesh_latest.pth"))

            
        if step >= 1000000:
            print("Reached max steps.")
            break


def save_mesh_vis(mesh, save_dir, step):
    # Plot wireframe density or grid
    H, W = mesh.H, mesh.W
    verts = mesh.vertices.detach().cpu().numpy() # (H, W, 2)
    
    plt.figure(figsize=(10, 5))
    # Plot every k-th line
    k = 4 # Plot every 4th line to avoid clutter
    for i in range(0, H, k):
        plt.plot(verts[i, :, 0], verts[i, :, 1], 'b-', lw=0.5, alpha=0.5)
    for j in range(0, W, k):
        plt.plot(verts[:, j, 0], verts[:, j, 1], 'b-', lw=0.5, alpha=0.5)
        
    plt.title(f"Mesh Optimization Step {step}")
    plt.axis('equal')
    plt.gca().invert_yaxis() # Fix: North (v=0) should be at Top
    plt.savefig(os.path.join(save_dir, f"mesh_vis_{step}.png"))
    plt.close()

def save_map_preview(mesh, tex_colors, save_dir, step):
    # Scatter plot of vertices colored by texture
    verts = mesh.vertices.detach().cpu().numpy().reshape(-1, 2)
    
    plt.figure(figsize=(10, 5))
    # s=1 might be too small or big depending on resolution. with 256x512=131k points.
    plt.scatter(verts[:, 0], verts[:, 1], c=tex_colors, s=1, marker='.')
    plt.axis('equal')
    plt.title(f"Map Preview Step {step}")
    plt.gca().invert_yaxis() # Fix: North (v=0) should be at Top
    plt.savefig(os.path.join(save_dir, f"map_preview_{step}.png"))
    plt.close()

if __name__ == "__main__":
    main()
