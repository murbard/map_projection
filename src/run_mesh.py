
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
    save_dir = "mesh_results"
    os.makedirs(save_dir, exist_ok=True)
    mask_path = "world_mask_highres.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Hyperparameters
    # Mesh Resolution
    H_mesh = 256
    W_mesh = 512 
    # This gives Grid Resolution. Number of triangles = 2 * (H-1) * (W-1).
    # Approx 260k triangles.
    
    # Loading Mask
    # We need mask values at the CENTER of each triangle for weighting.
    # Load mask at higher res?
    mask_high = load_land_mask(mask_path, size=2048).to(device)
    
    # Initialize Mesh
    mesh = DifferentiableMesh(H_mesh, W_mesh).to(device)
    
    # Optimizer
    optimizer = optim.Adam(mesh.parameters(), lr=1e-3)
    
    # Auto-Resume Logic
    start_step = 0
    latest_ckpt = os.path.join(save_dir, "mesh_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        mesh.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resumed from Step {start_step}.")
        
    print("Precomputing Triangle Centers and Sphere Jacobians...")
    # u_c, v_c (N_tri)
    u_c = mesh.u_centers
    v_c = mesh.v_centers
    
    # Sample Land Mask at Centers
    # u_c, v_c in [0, 1]
    H_mask, W_mask = mask_high.shape
    u_idx = (u_c * (W_mask - 1)).long().clamp(0, W_mask - 1)
    v_idx = (v_c * (H_mask - 1)).long().clamp(0, H_mask - 1)
    is_land = mask_high[v_idx, u_idx] # (N_tri)
    
    # Note: compute_distortion_loss calculates sphere Jacobian internally.
    
    # Load Texture for Preview
    texture_path = "lighter_map.jpg"
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

    print("Starting Optimization...")
    
    successful_steps = 0
    
    for step in range(start_step, 10001):
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
        
        loss_distortion = compute_distortion_loss(J_mesh, u_c, v_c, is_land, land_weight=50.0)
        
        # 4. Barrier Loss (Soft Constraint)
        # Det2D = J_00 * J_11 - J_01 * J_10
        dets = J_mesh[:, 0, 0] * J_mesh[:, 1, 1] - J_mesh[:, 0, 1] * J_mesh[:, 1, 0]
        
        # Log Barrier: -log(det).
        barrier_loss = -torch.log(torch.clamp(dets, min=1e-9)).mean()
        
        # Total Loss
        # Barrier weight: Small enough not to dominate, large enough to push back.
        barrier_weight = 0.01
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
            # Det J = Det(Mesh) * Det(Sphere). Sphere is > 0 (except poles).
            # Mesh Det = Det(E_q) * Det(E_p_inv).
            # We strictly check if Det(Mesh matrix) > 0.
            
            # Determinant of 2x2:
            det = new_J[:, 0, 0] * new_J[:, 1, 1] - new_J[:, 0, 1] * new_J[:, 1, 0]
            
            min_det = det.min()
            if min_det < 1e-6:
                # Collision detected!
                # Revert
                mesh.vertices.data.copy_(old_params)
                print(f"Step {step}: Crossing detected (Min Det {min_det.item():.2e}). Reducing LR.")
                
                successful_steps = 0 # Reset counter
                
                # Decay LR
                
                # Decay LR
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                    
                if optimizer.param_groups[0]['lr'] < 1e-7:
                    print("LR too small. Stopping.")
                    break
                
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
                    # print(f"Step {step}: Increasing LR to {new_lr:.2e}")
                    successful_steps = 0 # Reset to require another stable period
                
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")
            if tex_colors is not None:
                save_map_preview(mesh, tex_colors, save_dir, step)
            
        if step % 1000 == 0:
            save_mesh_vis(mesh, save_dir, step)
            # Save Checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': mesh.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, f"mesh_{step}.pth"))
            torch.save(checkpoint, os.path.join(save_dir, "mesh_latest.pth"))

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
