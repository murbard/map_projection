
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image
import os
import argparse
from src.mesh import DifferentiableMesh
from scipy.interpolate import LinearNDInterpolator

def main():
    save_dir = "mesh_results"
    model_path = os.path.join(save_dir, "mesh_latest.pth")
    texture_path = "light_map.png"
    if not os.path.exists(texture_path):
        texture_path = "equi_sat_8000x2000.png"
        
    output_path = "mesh_highres.png"
    
    device = torch.device('cpu') # Interpolation is CPU bound mostly
    
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Init Mesh
    # We need dimensions from weights or hardcode?
    # Weights has 'vertices' shape (H, W, 2).
    verts_shape = checkpoint['model_state_dict']['vertices'].shape
    H_mesh, W_mesh = verts_shape[0], verts_shape[1]
    
    mesh = DifferentiableMesh(H_mesh, W_mesh).to(device)
    mesh.load_state_dict(checkpoint['model_state_dict'])
    
    # 1. Get Triangulation in Input Space (UV) and Output Space (XY)
    # We need to map Output XY -> Input UV.
    # So we build interpolator on XY Points with UV Values.
    
    # Vertices (Output XY)
    verts = mesh.vertices.detach().numpy().reshape(-1, 2)
    # verts[:, 1] is Y. Invert Y for plotting if needed? 
    # Current mesh viz needed invert_yaxis to look right.
    # That means the optimizer produced Y values where "Up" is negative (or standard screen coords).
    # We will treat XY as just coordinates and handle orientation at the end.
    
    X = verts[:, 0]
    Y = verts[:, 1]
    
    # UVs (Input)
    # Mesh initialized with grid_u, grid_v
    v = torch.linspace(0, 1, H_mesh)
    u = torch.linspace(0, 1, W_mesh)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    
    U_flat = grid_u.flatten().numpy()
    V_flat = grid_v.flatten().numpy()
    
    # 2. Build Triangulation
    # We assume mesh topology is preserved (no flips), but even if deformed, we use it.
    indices = mesh.indices.cpu().numpy()
    
    print("Building Triangulation and Interpolator...")
    triangulation = mtri.Triangulation(X, Y, triangles=indices)
    
    interp_u = mtri.LinearTriInterpolator(triangulation, U_flat)
    interp_v = mtri.LinearTriInterpolator(triangulation, V_flat)
    
    # 3. Define Output Grid
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()
    
    print(f"Map Bounds: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
    
    # Resolution
    target_width = 4000
    aspect = (max_x - min_x) / (max_y - min_y)
    target_height = int(target_width / aspect)
    
    print(f"Generating {target_width}x{target_height} image...")
    
    xs = np.linspace(min_x, max_x, target_width)
    ys = np.linspace(min_y, max_y, target_height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    
    # 4. Interpolate
    print("Interpolating U...")
    grid_u_out = interp_u(grid_x, grid_y)
    print("Interpolating V...")
    grid_v_out = interp_v(grid_x, grid_y)
    
    # Mask invalid (outside convex hull)
    mask = grid_u_out.mask if np.ma.is_masked(grid_u_out) else np.zeros_like(grid_u_out, dtype=bool)
    
    # 5. Sample Texture
    print("Loading Texture...")
    tex = Image.open(texture_path).convert('RGB')
    tex_w, tex_h = tex.size
    tex_arr = np.array(tex) / 255.0
    
    # Coordinates to pixel coords
    # U, V in [0, 1]
    # U -> X_tex, V -> Y_tex
    
    # Handle masked/NaNs
    # Fill with 0 for safe indexing (will be overwritten by mask color later)
    u_vals = np.nan_to_num(grid_u_out, nan=0.0)
    v_vals = np.nan_to_num(grid_v_out, nan=0.0)
    
    u_vals = np.clip(u_vals, 0, 1)
    v_vals = np.clip(v_vals, 0, 1)
    
    # Map to integer indices
    # (H_out, W_out)
    ix = (u_vals * (tex_w - 1)).astype(int)
    iy = (v_vals * (tex_h - 1)).astype(int)
    
    print("Sampling Texture...")
    # Vectorized lookup
    sampled = tex_arr[iy, ix] # (H_out, W_out, 3)
    
    # Apply Mask (Background White or Transparent)
    # Let's make background white
    if np.ma.is_masked(grid_u_out):
        sampled[mask] = [1.0, 1.0, 1.0]
        
    # 6. Save
    # Check orientation.
    # Current mesh Y implies Y increases downwards? Or Upwards?
    # Mesh preview needed invert_axis.
    # This means Y values are standard math, but plots default 0 at bottom.
    # Inverse yaxis makes 0 at top.
    # Check orientation.
    # Mesh Y=0 is North (Top).
    # Array Index 0 is Y_min (0).
    # Image Top is Index 0.
    # So North is Top. No flip needed.
    
    plt.imsave(output_path, sampled)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
