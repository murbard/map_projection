# Map Projection Optimization via Differentiable Mesh

This project minimizes map distortion by optimizing a mesh-based mapping from the Sphere ($S^2$) to the Plane ($\mathbb{R}^2$) using a Differentiable Physics approach.

## Method

We model the map projection as a deformable mesh. The objective is to find a mapping that is locally isometric (preserving lengths and angles).

### Loss Function
We utilize a **Symmetric Frobenius Norm** distortion metric, which penalizes deviation from isometry ($J^TJ = I$).

$$ \mathcal{L} = \|J\|_{F}^{3/2} + \|J^{-1}\|_{F}^{3/2} $$

- **Theoretical Minimum**: For a perfect isometry (singular values $\sigma_1 = \sigma_2 = 1$), the loss is $\approx 3.36$.
- **Area Weighting**: The integral over the sphere is computed using exact spherical area weights for each mesh triangle.

## Experiments

We performed two primary experiments to evaluate the trade-off between global smoothness and local preservation of land masses. Results are saved in `experiments/`.

### 1. Land Priority (50x Weight)
- **Configuration**: Land regions are weighted **50 times** higher than ocean regions.
- **Outcome**: Continents are preserved with very low distortion, while oceans absorb most of the necessary deformation (stretching/tearing).
- **Results**: See `experiments/frobenius_50x/`

### 2. Uniform Weighting (1:1)
- **Configuration**: All surface area is weighted equally.
- **Outcome**: The distortion is distributed more evenly across the globe, leading to a wider aspect ratio and different global topology.
- **Results**: See `experiments/frobenius_uniform/`

## Usage

To train the model:

```bash
# Weighted Run (Default 50x)
python -m src.run_mesh
```

To modify weighting, edit `src/run_mesh.py` and change `land_weight` in the `compute_distortion_loss` call.

## High Resolution Rendering

Use the provided scripts to generate high-res maps and wireframes:

```bash
# Render Map (Textured)
python -m src.gen_mesh_image --model path/to/model.pth --output map.png --width 4000 --texture light_map.png

# Render Wireframe
python -m src.gen_mesh_wireframe --model path/to/model.pth --output wireframe.png
```
