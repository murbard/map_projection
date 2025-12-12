
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from src.train import load_land_mask, train_model
from src.model import IdentityAR

# Monkey patch train_model to use IdentityAR if needed, 
# or just copy main logic since train_model hardcodes BijectiveSquareFlow alias.
# Actually, train_model imports BijectiveSquareFlow from model.py
# In model.py, BijectiveSquareFlow = RealNVP.
# So we need a custom training loop or to modify train_model to accept a model class/instance.
# Modifying train.py is cleaner.

def main():
    mask = load_land_mask("world_mask_highres.png")
    
    # We will modify train_model to take a `model_class` or constructing it here.
    # But for now let's just create a custom loop here to be safe and fast.
    
    save_dir = 'identity_test'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    
    model = IdentityAR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2) # High LR for single parameter
    
    print("Starting Identity AR Optimization...")
    H, W = mask.shape
    batch_size = 16384
    
    loss_history = []
    ar_history = []
    
    from src.utils import compute_distortion_loss
    
    for epoch in range(5000): # Should converge fast
        optimizer.zero_grad()
        
        # Sampling
        u = torch.rand(batch_size, device=device)
        z = torch.rand(batch_size, device=device) * 2.0 - 1.0
        v = torch.arccos(z) / torch.tensor(3.1415926535, device=device)
        
        inputs = torch.stack([u, v], dim=-1)
        
        # Land check
        u_idx = (u * (W - 1)).long().clamp(0, W - 1)
        v_idx = (v * (H - 1)).long().clamp(0, H - 1)
        is_land = mask[v_idx, u_idx]
        
        # Jacobian
        jac = model.get_jacobian(inputs)
        
        # Loss
        loss = compute_distortion_loss(jac, u, v, is_land, land_weight=50.0)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            ar = torch.exp(model.log_aspect_ratio).item()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, AR {ar:.4f}")
            loss_history.append(loss.item())
            ar_history.append(ar)
            
    final_ar = torch.exp(model.log_aspect_ratio).item()
    print(f"Final Result: AR = {final_ar:.6f}")

if __name__ == "__main__":
    main()
