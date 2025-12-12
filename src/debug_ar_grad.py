
import torch
import torch.optim as optim
from src.train import load_land_mask
from src.model import BijectiveSquareFlow
from src.utils import compute_distortion_loss

def debug_gradients():
    mask = load_land_mask("world_mask_highres.png")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    H, W = mask.shape
    
    # Load Ultra model
    model = BijectiveSquareFlow(num_layers=16, hidden_dim=512, num_bins=32).to(device)
    path = "world_results_ultra/model_deep.pth"
    if os.path.exists(path):
        print(f"Loading {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print("Model not found, using init.")
        
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    
    # Single batch step
    batch_size = 16384
    u = torch.rand(batch_size, device=device)
    z = torch.rand(batch_size, device=device) * 2.0 - 1.0
    v = torch.arccos(z) / torch.tensor(3.1415926535, device=device)
    inputs = torch.stack([u, v], dim=-1)
    
    u_idx = (u * (W - 1)).long().clamp(0, W - 1)
    v_idx = (v * (H - 1)).long().clamp(0, H - 1)
    is_land = mask[v_idx, u_idx]
    
    jac = model.get_jacobian(inputs)
    loss = compute_distortion_loss(jac, u, v, is_land, land_weight=50.0)
    
    optimizer.zero_grad()
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    
    # Check Gradients
    ar_grad = model.log_aspect_ratio.grad
    print(f"Log Aspect Ratio: {model.log_aspect_ratio.item():.6f} (AR={torch.exp(model.log_aspect_ratio).item():.4f})")
    
    if ar_grad is not None:
        print(f"AR Gradient: {ar_grad.item():.8f}")
        print(f"AR Gradient Magnitude: {abs(ar_grad.item()):.8f}")
    else:
        print("AR Gradient is None!")
        
    # Check Weight Gradient for comparison
    w_grad = model.layers[0].net[0].weight.grad
    if w_grad is not None:
        print(f"Layer 0 Weight Gradient Mean Abs: {w_grad.abs().mean().item():.8f}")
        print(f"Layer 0 Weight Gradient Max: {w_grad.abs().max().item():.8f}")

    # Check Ratio
    if ar_grad is not None and w_grad is not None:
        ratio = abs(ar_grad.item()) / w_grad.abs().mean().item()
        print(f"AR Grad / Weight Grad Ratio: {ratio:.2f}")

    if ar_grad is None or abs(ar_grad.item()) < 1e-7:
        print("CONCLUSION: AR Gradient is vanishingly small. Parametrization issue confirmed.")
    elif ratio < 0.1:
         print("CONCLUSION: AR is learning much slower than weights.")
    else:
        print("CONCLUSION: AR Gradient looks healthy.")

import os
if __name__ == "__main__":
    debug_gradients()
