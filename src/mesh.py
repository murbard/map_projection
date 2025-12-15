
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableMesh(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H = height
        self.W = width
        
        # 1. Initialize Vertices on Regular Grid (Equirectangular)
        # u: [0, 1], v: [0, 1]
        # (H, W, 2)
        v = torch.linspace(0, 1, height)
        u = torch.linspace(0, 1, width)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
        
        # Initial positions: (u, v)
        # Shape (H, W, 2)
        initial_pos = torch.stack([grid_u, grid_v], dim=-1)
        
        # Parameter: Vertices
        self.vertices = nn.Parameter(initial_pos)
        
        # 2. Precompute Connectivity (Indices)
        # Quads -> 2 Triangles
        # (i, j), (i+1, j), (i, j+1) -> T1
        # (i+1, j), (i+1, j+1), (i, j+1) -> T2
        
        # We need efficient indexing.
        # Store indices for p1, p2, p3 for all triangles.
        
        i = torch.arange(height - 1)
        j = torch.arange(width - 1)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        
        # Flatten
        gi = gi.flatten()
        gj = gj.flatten()
        
        # Linear indices: idx = i * W + j
        idx_00 = gi * width + gj
        idx_10 = (gi + 1) * width + gj
        idx_01 = gi * width + (gj + 1)
        idx_11 = (gi + 1) * width + (gj + 1)
        
        # Triangle 1: 00, 10, 01
        # Triangle 2: 10, 11, 01
        
        # Stack indices: (NumQuads, 2, 3) -> (NumTriangles, 3)
        t1 = torch.stack([idx_00, idx_10, idx_01], dim=1)
        t2 = torch.stack([idx_10, idx_11, idx_01], dim=1)
        
        self.indices = torch.cat([t1, t2], dim=0) # (2*TotalQuads, 3)
        
        # 3. Precompute Source Edges (Inverse)
        # Source (u, v) is fixed on regular grid.
        # Triangles are standard right triangles in (u,v) space.
        
        # T1: p1=(0,0), p2=(0, dv), p3=(du, 0) (relative)
        # Edge1 = p2-p1 = (0, dv)
        # Edge2 = p3-p1 = (du, 0)
        # Matrix E_p = [[0, du], [dv, 0]]
        # Inverse E_p^-1 = [[0, 1/dv], [1/du, 0]]
        
        # T2: p1=(0, dv), p2=(du, dv), p3=(du, 0)
        # Edge1 = p2-p1 = (du, 0)
        # Edge2 = p3-p1 = (du, -dv)
        # E_p = [[du, du], [0, -dv]]
        
        # Wait, let's be precise.
        # T1 vertices: (u, v), (u, v+dv), (u+du, v)
        # p2-p1 = (0, dv) -> Column 0 ?? No.
        # Jacobian J maps (du, dv) -> (dx, dy)
        # J * [p2-p1] = [q2-q1]
        
        du = 1.0 / (width - 1)
        dv = 1.0 / (height - 1)
        
        # E_p_inv for T1
        # p2-p1 = (0, dv) (vector in uv)
        # p3-p1 = (du, 0)
        # E_p = [[0, du], [dv, 0]] (Cols are vectors)
        # Inv = 1/(-du*dv) * [[0, -du], [-dv, 0]] = [[0, 1/dv], [1/du, 0]]
        
        # E_p_inv for T2
        # T2 vertices: (u, v+dv), (u+du, v+dv), (u+du, v)
        # p1=(0, dv), p2=(du, dv), p3=(du, 0) (relative to origin)
        # p2-p1 = (du, 0)
        # p3-p1 = (du, -dv)
        # E_p = [[du, du], [0, -dv]]
        # Inv = 1/(-du*dv) * [[-dv, -du], [0, du]] = [[1/du, 1/dv], [0, -1/dv]]
        
        E_p_inv_t1 = torch.tensor([[0, 1/dv], [1/du, 0]])
        E_p_inv_t2 = torch.tensor([[1/du, 1/dv], [0, -1/dv]])
        
        # Replicate for all T1 and T2
        num_quads = idx_00.shape[0]
        
        self.register_buffer('E_p_inv_t1', E_p_inv_t1.unsqueeze(0).repeat(num_quads, 1, 1))
        self.register_buffer('E_p_inv_t2', E_p_inv_t2.unsqueeze(0).repeat(num_quads, 1, 1))
        
        # Combined Buffer: (NumTriangles, 2, 2)
        self.register_buffer('E_p_inv', torch.cat([self.E_p_inv_t1, self.E_p_inv_t2], dim=0))
        
        # Precompute UV centers for weighting/Jacobian sphere calc
        # For T1: Center ~ (i + 1/3, j + 1/3) roughly?
        # Just use (i,j) for simplicity or centroid.
        # Centroid of (0,0), (0,1), (1,0) is (1/3, 1/3).
        
        u_centers = (grid_u.flatten()[idx_00] + grid_u.flatten()[idx_10] + grid_u.flatten()[idx_01]) / 3
        v_centers = (grid_v.flatten()[idx_00] + grid_v.flatten()[idx_10] + grid_v.flatten()[idx_01]) / 3
        
        u_centers_2 = (grid_u.flatten()[idx_10] + grid_u.flatten()[idx_11] + grid_u.flatten()[idx_01]) / 3
        v_centers_2 = (grid_v.flatten()[idx_10] + grid_v.flatten()[idx_11] + grid_v.flatten()[idx_01]) / 3
        
        self.register_buffer('u_centers', torch.cat([u_centers, u_centers_2]))
        self.register_buffer('v_centers', torch.cat([v_centers, v_centers_2]))
        
        
    def get_jacobians(self):
        # 1. Gather Vertex positions
        # (NumTriangles, 3, 2)
        verts_flat = self.vertices.reshape(-1, 2)
        tri_verts = verts_flat[self.indices] # (N_tri, 3, 2)
        
        q1 = tri_verts[:, 0, :]
        q2 = tri_verts[:, 1, :]
        q3 = tri_verts[:, 2, :]
        
        # 2. Target Edges E_q
        # Col 0: q2 - q1
        # Col 1: q3 - q1
        edge1 = q2 - q1
        edge2 = q3 - q1
        
        # Stack cols: (N, 2, 2)
        # [[e1_x, e2_x], [e1_y, e2_y]]
        E_q = torch.stack([edge1, edge2], dim=-1)
        
        # 3. Compute Jacobian J = E_q @ E_p_inv
        # (N, 2, 2) @ (N, 2, 2)
        J = torch.bmm(E_q, self.E_p_inv)
        
        return J, E_q

    
    def check_collisions(self, new_vertices):
        # Compute determinants of E_q for new vertices
        # If any < epsilon, we have a flip or collapse.
        
        verts_flat = new_vertices.reshape(-1, 2)
        tri_verts = verts_flat[self.indices] # (N_tri, 3, 2)
        
        q1 = tri_verts[:, 0, :]
        q2 = tri_verts[:, 1, :]
        q3 = tri_verts[:, 2, :]
        
        edge1 = q2 - q1
        edge2 = q3 - q1
        
        # Det2D = x1*y2 - x2*y1
        # E_q = [[e1x, e2x], [e1y, e2y]]
        # Det = e1x * e2y - e2x * e1y
        det = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        
        # T1 should have positive det? 
        # Source triangle T1: p1(0,0), p2(0,1), p3(1,0). Det = 0*0 - 1*1 = -1? 
        # Source T1 edge1=(0,dv), edge2=(du,0). Det = 0*0 - dv*du = Negative.
        # Wait, if source area is negative (orientation), target must preserve it.
        # We should enforce Sign(Det_q) == Sign(Det_p).
        
        # Let's check Source Det
        # E_p_inv is stored. E_p = inv(E_p_inv).
        # Det(J) = Det(E_q) * Det(E_p_inv) > 0 implies same orientation?
        # Standard requirement: Det(J) > 0.
        
        # J = E_q @ E_p_inv.
        # Det(J) = Det(E_q) * Det(E_p_inv).
        # We strictly want Det(J) > epsilon.
        
        # But we need J efficiently. get_jacobians does it.
        # But we want to do this BEFORE accepting the step.
        pass # Logic handled in optimization loop or here efficiency
