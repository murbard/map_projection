
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src.mesh import DifferentiableMesh

def main():
    save_dir = "mesh_results"
    model_path = "temp_mesh_wire.pth"
    output_path = "mesh_wireframe_highres.png"
    
    device = torch.device('cpu')
    
    print(f"Loading {model_path}...")
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    verts_shape = checkpoint['model_state_dict']['vertices'].shape
    H_mesh, W_mesh = verts_shape[0], verts_shape[1]
    
    mesh = DifferentiableMesh(H_mesh, W_mesh).to(device)
    mesh.load_state_dict(checkpoint['model_state_dict'])
    
    # Vertices (Output XY)
    verts = mesh.vertices.detach().numpy() # (H, W, 2)
    X = verts[:, :, 0]
    Y = verts[:, :, 1]
    
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()
    
    width = max_x - min_x
    height = max_y - min_y
    aspect = width / height
    
    print(f"Map Bounds: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
    print(f"Aspect Ratio: {aspect:.4f}")
    
    # Target Resolution
    # User said "High Res". Let's go for 4000px width.
    target_w_px = 4000
    dpi = 100
    w_inches = target_w_px / dpi
    h_inches = w_inches / aspect
    
    print(f"Generating Wireframe {target_w_px}x{int(target_w_px/aspect)}...")
    
    fig = plt.figure(figsize=(w_inches, h_inches), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]) # Fill entire figure
    ax.axis('off')
    
    # Plot Triangulation
    # We use matplotlib.tri.Triangulation to plot the actual mesh edges
    import matplotlib.tri as mtri
    
    # Flatten vertices
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Get indices from mesh
    indices = mesh.indices.cpu().numpy()
    
    triang = mtri.Triangulation(x_flat, y_flat, triangles=indices)
    
    # Plot edges using triplot
    # 'k-' for edges
    ax.triplot(triang, 'k-', lw=0.3, alpha=0.3)
        
    # Aspect Equal
    ax.set_aspect('equal')
    
    # Invert Y per user requirement (North Up)
    ax.invert_yaxis()
    
    # Set Limits tight
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y) # Matches invert_yaxis? 
    # invert_yaxis swaps the display direction.
    # set_ylim(bottom, top). If inverted, pass (max, min).
    # Wait, invert_yaxis just toggles the direction.
    # We should set limits normally and let invert work, or set explicit.
    # Let's set limits (min, max) and call invert_yaxis.
    # Actually, let's look at previous success.
    
    # Previous script:
    # axis('equal')
    # invert_yaxis()
    
    ax.set_xlim(min_x, max_x)
    # Default Y increases up. invert makes it increase down.
    # We want top to be North (min_y).
    # Mesh v=0 is North.
    # Y values: Are they consistent with screen coords?
    # run_mesh loop needs 'invert_yaxis' to look correct.
    # This implies Y values are such that North has LOWER Y value than South (Standard Screen Coords),
    # BUT matplotlib default origin is Bottom-Left.
    # So plots look upside down. Inverting Y fixes it.
    
    # So we just set limits min to max, and invert.
    ax.set_ylim(min_y, max_y)
    ax.invert_yaxis()
    
    plt.savefig(output_path, dpi=dpi, transparent=True)
    plt.close()
    
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
