
import torch
import torch.nn as nn
from src.model import RealNVP
import time

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Massive Config
    batch_size = 16535
    layers = 24
    hidden_dim = 1024
    
    print(f"Config: B={batch_size}, L={layers}, H={hidden_dim}")
    print_gpu_memory("Start")
    
    model = RealNVP(num_layers=layers, hidden_dim=hidden_dim, num_bins=32).to(device)
    print_gpu_memory("Model Loaded")
    
    inputs = torch.rand(batch_size, 2, device=device, requires_grad=True)
    print_gpu_memory("Inputs Created")
    
    # 1. Forward (Jacobian)
    t0 = time.time()
    try:
        jac = model.get_jacobian(inputs)
        torch.cuda.synchronize()
        print(f"Jacobian computed in {time.time() - t0:.2f}s")
        print_gpu_memory("After Jacobian")
    except RuntimeError as e:
        print(f"OOM during Jacobian: {e}")
        return

    # 2. Loss (Dummy)
    loss = jac.sum()
    print_gpu_memory("After Loss")
    
    # 3. Backward
    t0 = time.time()
    try:
        loss.backward()
        torch.cuda.synchronize()
        print(f"Backward computed in {time.time() - t0:.2f}s")
        print_gpu_memory("After Backward")
    except RuntimeError as e:
        print(f"OOM during Backward: {e}")
        return
        
    if model.layers[0].net[0].weight.grad is None:
        print("FAILURE: No gradients on weights! Implementation detached graph incorrectly.")
    else:
        print("Success! Gradients computed.")
        print(f"Grad norm: {model.layers[0].net[0].weight.grad.norm().item()}")

if __name__ == "__main__":
    main()
