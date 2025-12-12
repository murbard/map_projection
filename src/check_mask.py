
import torch
import numpy as np
from src.train import load_land_mask
import sys

def check_mask():
    mask_path = "world_mask_highres.png"
    print(f"Loading {mask_path}...")
    try:
        mask = load_land_mask(mask_path, size=2048)
    except Exception as e:
        print(e)
        return

    H, W = mask.shape
    print(f"Mask Shape: {H}x{W}")
    print(f"Mean Value: {mask.float().mean().item():.4f}")
    
    # Define probes
    probes = {
        "Pacific (Sea)": (0.0, 0.5),      # Left edge, equator
        "Atlantic (Sea)": (0.25, 0.5),    # mid-Atlantic (approx -90)
        "Africa (Land)": (0.55, 0.5),     # East Africa (~20E)
        "Antarctica (Land)": (0.5, 0.95), # South Pole
        "Arctic (Sea)": (0.5, 0.05),      # North Pole
    }
    
    print("\nProbing mask (1=Land, 0=Sea)...")
    for name, (u, v) in probes.items():
        c_x = int(u * (W - 1))
        c_y = int(v * (H - 1))
        val = mask[c_y, c_x].item()
        print(f"{name} (u={u:.2f}, v={v:.2f}) -> {val}")
        
    # Heuristic check
    if mask.float().mean().item() > 0.5:
        print("\nWARNING: Mask mean > 0.5. Unless the map is inverted or not Earth, this is likely wrong.")
        print("Earth is ~30% land.")

if __name__ == "__main__":
    check_mask()
