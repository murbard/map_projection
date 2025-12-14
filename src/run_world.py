
import torch
import sys
import os
import torch.nn as nn
from src.train import train_model, load_land_mask
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
    
    print("Starting training on Real World Map (Run 11: Optimized SiLU, Equal Weights 1:1)...")
    # Epochs: 10,000 
    # LR: 4.8e-4
    # Weight: 1.0 (Equal)
    model = train_model(mask, num_epochs=10000, lr=4.8e-4, batch_size=15872,
                        save_dir='world_results_optimized_equal', num_bins=32, land_weight=1.0,
                        layers=38, hidden_dim=35, activation_cls=nn.SiLU)
    
    ar = torch.exp(model.log_aspect_ratio).item()
    print(f"Training complete. Learned Aspect Ratio: {ar:.4f} (Log: {model.log_aspect_ratio.item():.4f})")
    print("Check 'world_results_ultra'.")

if __name__ == "__main__":
    main()
