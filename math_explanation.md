# Math Implementation Recap

This document outlines the mathematical framework used in the code to minimize map distortion.

## 1. Composition of the Jacobian
We aim to analyze the local distortion introduced by the map projection $f: S^2 \to \mathbb{R}^2$. To do this, we compute the total Jacobian matrix mapping tangent vectors from the Sphere to the Planar Map.

The map is defined as a composition of coordinate changes and the neural network transformation:
$$ f_{total} = f_{net} \circ \phi $$
where $\phi: S^2 \to [0,1]^2$ is the Equirectangular projection (mapping Sphere to UV square), and $f_{net}: [0,1]^2 \to \mathbb{R}^2$ is the learned Bijective Flow.

The Total Jacobian $J_{total}$ is composed using the Chain Rule:
$$ J_{total} = J_{flow} \cdot J_{sphere \to uv} $$

### Step A: Sphere to UV Jacobian ($J_{sphere \to uv}$)
Implemented in `src/utils.py`: `compute_sphere_jacobian_to_equirectangular`.

We define an orthonormal basis on the sphere $\{e_\phi, e_\theta\}$ and the coordinate basis on the UV plane $\{\frac{\partial}{\partial u}, \frac{\partial}{\partial v}\}$.
With $u = \frac{\phi}{2\pi}$ and $v = \frac{\theta}{\pi}$:

$$ J_{sphere \to uv} = \begin{bmatrix} \frac{\partial u}{\partial e_\phi} & \frac{\partial u}{\partial e_\theta} \\ \frac{\partial v}{\partial e_\phi} & \frac{\partial v}{\partial e_\theta} \end{bmatrix} = \begin{bmatrix} \frac{1}{2\pi R \sin(\theta)} & 0 \\ 0 & \frac{1}{\pi R} \end{bmatrix} $$

In the code, $R = \frac{1}{\sqrt{4\pi}}$ (Unit Area Sphere), leading to the specific constants used.

### Step B: Flow Jacobian ($J_{flow}$)
Implemented in `src/model.py`: `RealNVP.get_jacobian`.

This is the Jacobian of the neural network $f_{net}(u, v) = (x, y)$ with respect to $(u, v)$.
$$ J_{flow} = \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{bmatrix} $$

It is computed efficiently using `torch.vmap(torch.func.jacrev(flow_func))`.

### Step C: End-to-End composition
The matrices are multiplied in `src/utils.py`: `compute_distortion_loss`.
```python
j_total = torch.bmm(jacobian_flow, jacobian_sphere)
```

## 2. Riemannian Metric Error (Distortion Loss)
Implemented in `src/utils.py`: `compute_distortion_loss`.

Ideally, for an isometric (length-preserving) map, we want the transformation of orthonormal vectors to remain orthonormal. This implies $J_{total}^T J_{total} = I$.

We use the Singular Value Decomposition (SVD) of $J_{total}$ to inspect the local distortion:
$$ J_{total} = U \Sigma V^T $$
where $\Sigma = \text{diag}(\sigma_1, \sigma_2)$. The values $\sigma_i$ differ from $1.0$ effectively measure the stretching or squashing along principal axes.

The loss function minimizes the deviation of singular values from 1 in log-space:
$$ \mathcal{L} = \sum_{i} (\log \sigma_i)^2 $$

This is equivalent to the squared geodesic distance on the Riemannian manifold of positive definite matrices (Log-Euclidean metric) between $J^T J$ and $I$.

The loss is weighted by a land mask to prioritize specific regions:
$$ \mathcal{L}_{weighted} = \frac{1}{N} \sum_k w_k \cdot \mathcal{L}_k $$

## 3. Sampling Points
Implemented in `src/train.py`: `train_model`.

To ensure the loss represents the global distortion accurately, we sample points uniformly distributed by **area** on the sphere.

1. **Longitude ($u$)**: Sampled uniformly from $[0, 1]$.
   $$ u \sim U(0, 1) $$
   
2. **Latitude ($v$)**: Sampled to preserve area.
   Since differential area $dA = \sin(\theta) d\theta d\phi = -d(\cos \theta) d\phi$.
   We sample $z = \cos(\theta)$ uniformly from $[-1, 1]$.
   $$ z \sim U(-1, 1) $$
   Then convert back to polar angle $\theta$:
   $$ \theta = \arccos(z) $$
   $$ v = \frac{\theta}{\pi} $$

This Monte Carlo integration ensures that minimizing the expected loss corresponds to minimizing the integral of distortion over the sphere's surface.
