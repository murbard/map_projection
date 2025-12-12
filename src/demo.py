
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .train import train_model

def create_synthetic_mask(size=256):
    """
    Creates a synthetic "Earth" mask with some blobs representing land.
    """
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
    
    # 1. Continent 1 (Top Left)
    c1 = ((x - 0.3)**2 + (y - 0.3)**2) < 0.15**2
    
    # 2. Continent 2 (Bottom Right)
    c2 = ((x - 0.7)**2 + (y - 0.7)**2) < 0.15**2
    
    # 3. "Antarctica" strip at bottom (to test pole behavior)
    c3 = (y > 0.9)
    
    # Combined mask
    mask = (c1 | c2 | c3).float()
    return mask

def main():
    print("Running Map Projection Demo...")
    
    mask = create_synthetic_mask(size=256)
    
    # Save mask for inspection
    plt.imsave("demo_mask.png", mask.numpy(), cmap='gray')
    
    # Run training
    # Fast run for demo
    model = train_model(mask, num_epochs=1000, batch_size=2048, lr=5e-4, 
                        save_dir='demo_results', layers=4, num_bins=8)
    
    print("Demo completed. Check 'demo_results' directory.")

if __name__ == "__main__":
    main()
