
import torch
import sys
import os
from .train import train_model, load_land_mask
import matplotlib.pyplot as plt

def main():
    mask_path = "world_mask_highres.png"
    if not os.path.exists(mask_path):
        print("Mask not found.")
        sys.exit(1)
        
    print(f"Loading mask from {mask_path}...")
    # Load at higher resolution for precision (squeezed to 2048x2048)
    mask = load_land_mask(mask_path, size=2048)
    
    # Invert logic if needed: image usually black on transparent?
    # Imagemagick convert SVG on default transparent BG might make black lines?
    # Or whitelines?
    # Let's check mask statistics quickly or safe-guard.
    # Usually: Land is non-zero.
    
    print("Mask average value:", mask.float().mean().item())
    # If average is very high (most of world is sea?), check logic.
    # Earth is 70% water.
    # If mask is 1 for land, mean should be ~0.3.
    # If mask is 1 for sea, mean should be ~0.7.
    
    # Save processed mask to check (downsampled for quick view/save space?)
    # No, save full ensures we know what we loaded.
    plt.imsave("processed_mask_check.png", mask.cpu().numpy(), cmap='gray')
    
    print("Starting training on Real World Map (Run 9: Massive Scale, L=24, H=1024, W=50, B=4096)...")
    # Epochs: 1,000,000 
    # LR: Differential (Handled in train.py)
    # Weight: 50.0 
    model = train_model(mask, num_epochs=1000000, batch_size=4096, lr=2e-5, 
                        save_dir='world_results_massive', layers=24, num_bins=32, land_weight=50.0, hidden_dim=1024)
    
    ar = torch.exp(model.log_aspect_ratio).item()
    print(f"Training complete. Learned Aspect Ratio: {ar:.4f} (Log: {model.log_aspect_ratio.item():.4f})")
    print("Check 'world_results_ultra'.")

if __name__ == "__main__":
    main()
