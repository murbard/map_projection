
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from .model import BijectiveSquareFlow
from .utils import compute_distortion_loss

def load_land_mask(image_path, size=256):
    """
    Loads a land mask image, resizes it to square (size x size), 
    and returns a boolean tensor.
    Expects input image to be 2:1 (Equirectangular), 
    mapped to 1:1 by simple resizing (squeezing).
    """
    try:
        img = Image.open(image_path).convert('L') # Grayscale
    except Exception as e:
        print(f"Could not load image at {image_path}: {e}")
        # Return dummy mask
        return torch.zeros((size, size))

    img = img.resize((size, size), Image.Resampling.NEAREST)
    data = np.array(img)
    # Norm to [0, 1]
    data = data / 255.0
    
    # FIX: Previously (data > 0.5) yielded 1 for Sea (White).
    # We want 1 for Land. Land is Dark/Black in the source or mask found.
    # So we want (data < 0.5).
    mask = torch.tensor(data < 0.5, dtype=torch.float32)
    return mask

def train_model(land_mask, num_epochs=1000, batch_size=1024, lr=1e-3, 
                save_dir='results', land_weight=10.0, hidden_dim=64, num_bins=16, layers=6, resume_from=None):
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    land_mask = land_mask.to(device)
    H, W = land_mask.shape
    print(f"Land Mask Mean: {land_mask.mean().item():.4f} (Should be ~0.3 for Earth)")
    
    
    model = BijectiveSquareFlow(num_layers=layers, hidden_dim=hidden_dim, num_bins=num_bins).to(device)
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}...")
        try:
             # Weights only since we don't save optimizer state
            model.load_state_dict(torch.load(resume_from, map_location=device))
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            
    # Differential Learning Rate Parameter Groups
    # Group 1: Log Aspect Ratio (Needs High LR)
    # Group 2: Network Weights (Needs Low LR)
    
    ar_params = []
    net_params = []
    
    for name, param in model.named_parameters():
        if "log_aspect_ratio" in name:
            ar_params.append(param)
        else:
            net_params.append(param)
            
    print(f"Optimizer: {len(ar_params)} AR params (LR={1e-2}), {len(net_params)} Net params (LR={lr})")
            
    optimizer = optim.Adam([
        {'params': net_params, 'lr': lr},
        {'params': ar_params, 'lr': 1e-2} # Boost AR learning rate by 500x
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    history = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Sample random points
        # Strategy: Uniform on Sphere
        # u (phi) is uniform [0, 1]
        u = torch.rand(batch_size, device=device)
        
        # v (theta) needs to be sampled such that area is uniform.
        # z = cos(theta) ~ U[-1, 1]
        z = torch.rand(batch_size, device=device) * 2.0 - 1.0
        # theta = arccos(z) (ranges 0 to pi)
        theta = torch.arccos(z)
        # v = theta / pi (ranges 0 to 1)
        v = theta / np.pi
        
        inputs = torch.stack([u, v], dim=-1)
        
        # Determine is_land for these points
        # Map u, v to indices
        # u is x-axis (cols), v is y-axis (rows)
        # u index: floor(u * W)
        # v index: floor(v * H)
        u_idx = (u * (W - 1)).long()
        v_idx = (v * (H - 1)).long()
        
        # Clamp just in case
        u_idx = torch.clamp(u_idx, 0, W - 1)
        v_idx = torch.clamp(v_idx, 0, H - 1)
        
        is_land = land_mask[v_idx, u_idx]
        
        # Forward pass (Get Jacobian)
        jacobian_flow = model.get_jacobian(inputs)
        
        loss = compute_distortion_loss(jacobian_flow, u, v, is_land, land_weight=land_weight)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        scheduler.step(loss)
        
        history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            save_visualization(model, land_mask, save_dir, epoch)
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))
            
        # Also save latest constantly in case of crash
        if epoch % 100 == 0:
             torch.save(model.state_dict(), os.path.join(save_dir, 'model_latest.pth'))

    # Final save
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    save_visualization(model, land_mask, save_dir, 'final')
    return model

def save_visualization(model, land_mask, save_dir, tag):
    model.eval()
    device = next(model.parameters()).device
    
    # Grid for visualization
    grid_size = 20
    u = np.linspace(0, 1, grid_size)
    v = np.linspace(0, 1, grid_size)
    
    # Plot transformed grid lines
    plt.figure(figsize=(10, 10))
    
    # Prepare background land mask for reference
    # Show the "Target" domain (Unit Square) land?
    # No, show the "Source" domain (Equirect) grid mapped to Target?
    # Wait, the map is Source(Earth) -> Target(Square).
    # We want to see how the Earth grid looks on the Square.
    # So we take a regular grid on Earth (u,v), map it through f, and plot lines in x,y.
    
    # Background: Since the output is unit square, we can just plot lines.
    # But where is the land? 
    # Land is at specific (u,v). If we map (u,v) -> (x,y), the land moves to (x,y).
    # We should plot the mapped land mask?
    # Mapping points is easy.
    # Let's map a dense cloud of land points to see where they end up.
    
    with torch.no_grad():
        # Plot distorted grid lines
        for i in u:
            # Vertical line in UV (constant u, varying v)
            line_v = torch.linspace(0, 1, 100, device=device)
            line_u = torch.full_like(line_v, i)
            pts = torch.stack([line_u, line_v], dim=-1)
            mapped = model(pts).cpu().numpy()
            plt.plot(mapped[:, 0], mapped[:, 1], 'b-', alpha=0.3)
            
        for i in v:
            # Horizontal line in UV (varying u, constant v)
            line_u = torch.linspace(0, 1, 100, device=device)
            line_v = torch.full_like(line_u, i)
            pts = torch.stack([line_u, line_v], dim=-1)
            mapped = model(pts).cpu().numpy()
            plt.plot(mapped[:, 0], mapped[:, 1], 'r-', alpha=0.3)

        # Plot land points
        # Sample points where mask is true
        y_idxs, x_idxs = torch.where(land_mask > 0.5)
        # Limit number of points
        num_pts = 5000
        if len(y_idxs) > num_pts:
            perm = torch.randperm(len(y_idxs))[:num_pts]
            y_idxs = y_idxs[perm]
            x_idxs = x_idxs[perm]
            
        # Convert to UV
        H, W = land_mask.shape
        u_land = x_idxs.float() / (W - 1)
        v_land = y_idxs.float() / (H - 1)
        land_pts = torch.stack([u_land, v_land], dim=-1).to(device)
        
        if len(land_pts) > 0:
            mapped_land = model(land_pts).cpu().numpy()
            plt.scatter(mapped_land[:, 0], mapped_land[:, 1], s=1, c='g', alpha=0.5, label='Land')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.title(f"Map Distortion at epoch {tag}")
    plt.savefig(os.path.join(save_dir, f"vis_{tag}.png"))
    plt.close()
