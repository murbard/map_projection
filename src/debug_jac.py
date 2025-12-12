
import torch
import torch.optim as optim
from src.model import BijectiveSquareFlow
from src.utils import compute_distortion_loss, compute_sphere_jacobian_to_equirectangular

def debug_optimization():
    print("Initializing Debug Optimization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BijectiveSquareFlow(num_layers=4, hidden_dim=32, num_bins=8).to(device)
    
    # Point near pole: v small
    u = torch.tensor([0.5], device=device)
    v = torch.tensor([0.01], device=device) # Very close to pole
    inputs = torch.stack([u, v], dim=-1)
    
    is_land = torch.tensor([1.0], device=device) # Treat as land to force minimization
    
    # Check initial Jacobian
    jac_sphere = compute_sphere_jacobian_to_equirectangular(u, v)
    print(f"Sphere Jacobian at v=0.01:\n{jac_sphere.cpu().detach().numpy()}")
    # Should be [[Large, 0], [0, Const]]
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Optimizing...")
    for i in range(500):
        optimizer.zero_grad()
        jac_flow = model.get_jacobian(inputs)
        
        # Check flow jacobian
        if i == 0:
            print(f"Initial Flow Jacobian:\n{jac_flow.cpu().detach().numpy()}")
            # Should be close to Identity
            
        loss = compute_distortion_loss(jac_flow, u, v, is_land, land_weight=1.0)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item()}")
            
    print(f"Final Loss: {loss.item()}")
    final_jac_flow = model.get_jacobian(inputs).cpu().detach().numpy()
    print(f"Final Flow Jacobian:\n{final_jac_flow}")
    
    # Check total
    jac_total = torch.bmm(model.get_jacobian(inputs), jac_sphere)
    print(f"Final Total Jacobian:\n{jac_total.cpu().detach().numpy()}")
    # Should be close to rotation/identity
    
    u, s, vh = torch.linalg.svd(jac_total)
    print(f"Singular Values: {s.cpu().detach().numpy()}")

if __name__ == "__main__":
    debug_optimization()
