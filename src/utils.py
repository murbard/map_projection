
import torch
import numpy as np

def compute_sphere_jacobian_to_equirectangular(u, v, epsilon=1e-6):
    """
    Computes the Jacobian of the mapping from Sphere (orthonormal basis) 
    to Equirectangular UV square (coordinate basis).

    The sphere is assumed to have surface area 1, so R = 1/sqrt(4*pi).
    
    Mapping psi: (theta, phi) -> (u, v)
    u = phi / (2*pi)
    v = theta / pi
    
    Orthonormal basis on Sphere:
    e_theta = (1/R) * d/d_theta
    e_phi   = (1/(R*sin(theta))) * d/d_phi
    
    Coordinate basis on UV: d/du, d/dv
    
    J * e_theta = (1/R) * (du/d_theta * d/du + dv/d_theta * d/dv)
                = (1/R) * (0 + (1/pi) * d/dv)
                = 1/(pi*R) * d/dv
                
    J * e_phi   = (1/(R*sin(theta))) * (du/d_phi * d/du + dv/d_phi * d/dv)
                = ... * ((1/(2*pi)) * d/du + 0)
                = 1/(2*pi*R*sin(theta)) * d/du
                
    Matrix J (rows: u, v; cols: e_phi, e_theta) 
    (Note: aligning with standard x,y -> u,v orientation where u is horizontal (phi), v is vertical (theta))
    
    J = [[ 1/(2*pi*R*sin(theta)), 0              ],
         [ 0,                     1/(pi*R)       ]]
         
    With R = 1/(2*sqrt(pi)):
    1/(pi*R) = 2*sqrt(pi)/pi = 2/sqrt(pi)
    1/(2*pi*R) = sqrt(pi)/pi = 1/sqrt(pi)
    
    So J_00 = 1/(sqrt(pi) * sin(theta))
    J_11 = 2/sqrt(pi)
    """
    # u, v are in [0, 1]
    # theta = v * pi
    theta = v * np.pi
    
    # Avoid singular poles
    sin_theta = torch.sin(theta)
    sin_theta = torch.clamp(sin_theta, min=epsilon)
    
    sqrt_pi = np.sqrt(np.pi)
    
    # J_00 corresponds to partial u / partial e_phi
    j_00 = 1.0 / (sqrt_pi * sin_theta)
    
    # J_11 corresponds to partial v / partial e_theta
    j_11 = torch.full_like(u, 2.0 / sqrt_pi)
    
    # Zeros
    zeros = torch.zeros_like(u)
    
    # Stack to shape (Batch, 2, 2)
    # [[du/d_e_phi, du/d_e_theta], [dv/d_e_phi, dv/d_e_theta]]
    # [[j_00, 0], [0, j_11]]
    row1 = torch.stack([j_00, zeros], dim=-1)
    row2 = torch.stack([zeros, j_11], dim=-1)
    jacobian = torch.stack([row1, row2], dim=-2)
    
    return jacobian

def compute_distortion_loss(jacobian_flow, u, v, is_land, land_weight=10.0, geometry_weights=None):
    """
    Computes the weighted distortion loss using Symmetric Frobenius Norms.
    Loss = ||J||_F^(3/2) + ||J^-1||_F^(3/2)
    
    jacobian_flow: (B, 2, 2) Jacobian of the network f: (u,v) -> (x,y)
    u, v: (B,) Coordinates
    is_land: (B,) Boolean or Float mask (1.0 for land, 0.0 for sea)
    """
    
    # 1. Compute Sphere -> UV Jacobian
    # Shape: (B, 2, 2)
    jacobian_sphere = compute_sphere_jacobian_to_equirectangular(u, v)
    
    # 2. Composition: Chain rule
    # J_total = J_flow @ J_sphere
    # J maps Sphere Tangent Plane -> Map Plane
    j_total = torch.bmm(jacobian_flow, jacobian_sphere)
    
    # 3. Compute Singular Values of J_total
    try:
        _, s, _ = torch.linalg.svd(j_total)
    except RuntimeError:
        j_total = j_total + torch.eye(2, device=j_total.device).unsqueeze(0) * 1e-6
        _, s, _ = torch.linalg.svd(j_total)

    # 4. New Loss Calculation
    # ||J||_F^2 = sum(s_i^2)
    # ||J^-1||_F^2 = sum(1/s_i^2)
    
    # Avoid div by zero
    s = s + 1e-8
    
    norm_sq = torch.sum(s**2, dim=-1)
    norm_inv_sq = torch.sum(1.0 / (s**2), dim=-1)
    
    # Power 3/4 applied to the squared norms gives Power 3/2 to the norms
    # ||J||^1.5 = (||J||^2)^0.75
    term1 = torch.pow(norm_sq, 0.75)
    term2 = torch.pow(norm_inv_sq, 0.75)
    
    loss_per_point = term1 + term2
    
    # 5. Weighted Mean
    is_land = is_land.float()
    import_weights = is_land * (land_weight - 1.0) + 1.0
    
    # Clean possible NaNs
    loss_per_point = torch.nan_to_num(loss_per_point, nan=0.0, posinf=0.0, neginf=0.0)
    
    if geometry_weights is not None:
        # Mesh training: Weighted Mean by Area
        total_weights = import_weights * geometry_weights
        weighted_loss = (loss_per_point * total_weights).sum() / (total_weights.sum() + 1e-8)
    else:
        # Standard training
        weighted_loss = (loss_per_point * import_weights).mean()
    
    return weighted_loss
