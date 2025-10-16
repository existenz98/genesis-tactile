# tacto6d — 6D Tactile Sensing with FEM + iFEM + Synthetic Camera data generation

- **Forward FEM**: given a dense surface force distribution (Fx, Fy, Fz), solve 3D deformation of a layered silicone block.
- **Inverse (iFEM-style)**: given (measured or simulated) multi-depth displacement, recover the surface force field (with regularization/constraints).
- **End-to-end tests**: generate test forces → forward simulate → sample/“observe” → inverse solve → verify accuracy.
- **Synthetic camera**: render RGB images of random colored particles embedded at multiple depths, before/after load, at camera resolutions (e.g., 640×480).
- With 2D/3D visualization (matplotlib + PyVista).

---

## 1) Features

- Forward FEM with **FEniCSx/dolfinx** (layered material, near-incompressible, fixed bottom & sides, free top).
- Influence-matrix **iFEM-style** inverse with Tikhonov/TV regularization + optional physical constraints (e.g., `tz ≥ 0`, friction cone).
- Visualization:
  - 2D heatmaps & quiver for forces and errors (matplotlib).
  - 3D mesh deformation, slicing, and interactive views (PyVista/VTK).
- End-to-end regression tests (RMSE, correlation, angle error, SSIM/PSNR for `tz`).
- Synthetic **RGB particle renderer** for 3 depth bands (R/G/B), PSF blur, noise, optional vignetting; exports paired frames and (optionally) ground-truth optical flow.

---

## 2) Environment and OS notes

- **FEM (dolfinx) is best on Linux**. Use **WSL2 (Ubuntu)** on Windows for FEM + inverse.  
- **Visualization on Windows** is fully supported (PyVista/VTK); for FEM you can still run in WSL2.
- If using **WSLg** (Windows 11), PyVista GUIs work out-of-the-box in WSL2; otherwise, use off-screen rendering.

FEM (fenics-dolfinx, PETSc/SLEPc) should be installed via conda using on environment.yml Linux or WSL2.
On Windows native, requirements.txt gives you synthetic rendering + plotting, but no FEM.

### Setup in Ubuntu (including WSL2)

```bash
conda config --env --set channel_priority strict
conda env create -f environment.yml
conda activate tacto6d
```

Optional, for extra packages:

```bash
pip install -r requirements.txt
```

### Setup in Windows (no FEM)

Only installs plotting and image deps; FEM via dolfinx is not provided on Windows

```bash
pip install -r requirements.txt
```

## 3) Running scripts

Each task is a directly runnable src/scripts/

### Generate a synthetic surface force:

Pressure (normal forcee)

```bash
python src/scripts/generate_synth_force.py --preset pressure --fz_peak_mpa 0.12 --sigma_mm 3.5   --Lx_mm 40 --Ly_mm 30 --Nx 80 --Ny 60   --out data/input/force_pressure.npz   --preview_prefix data/input/force_pressure
```

Sheer (x,y)

```bash
python src/scripts/generate_synth_force.py --preset shear --Lx_mm 40 --Ly_mm 30 --Nx 80 --Ny 60 --cx_mm 15 --cy_mm 10 --sigma_mm 3.0 --tau_shear_mpa 0.05 --shear_dir_deg 45 --out data/input/force_shear.npz --preview_prefix data/input/force_shear
```

Torque

```bash
python src/scripts/generate_synth_force.py --preset torque --tau_torque_mpa 0.05 --sigma_mm 4.0 --shear_dir_deg 0 --torque_inner_mm 0.5 --out data/input/force_torque.npz --preview_prefix data/input/force_torque
```

Combined (Pressure, Shear, Torque)

```bash
python src/scripts/generate_synth_force.py \
  --preset combo \
  --Lx_mm 40 \
  --Ly_mm 30 \
  --Nx 80 \
  --Ny 60 \
  --cx_mm 15 \
  --cy_mm 10 \
  --sigma_mm 3.0 \
  --fz_peak_mpa 0.10 \
  --tau_shear_mpa 0.05 \
  --tau_torque_mpa 0.05 \
  --shear_dir_deg 0 \
  --torque_inner_mm 0.5 \
  --out data/input/force_combo.npz \
  --preview_prefix data/input/force_combo
```

this will make: A Gaussian normal patch (fz_peak_mpa) + A shear patch (tau_shear_mpa, shear_dir_deg) + A torque field (half of tau_torque_mpa, slightly larger sigma).

### Forward FEM (force → displacement):

```bash
python src/scripts/run_forward.py --force data/input/force_combo.npz   --config src/config/material.yaml   --xdmf data/output/u.xdmf   --viz_prefix data/output/disp_top   --sample_Nx 80 --sample_Ny 60  --ksp cg --pc gamg --ksp_monitor
```

This will load force field data from 'data/input/force_combo.npz', and save FEM result to 'data/output/u.xdmf'.

### Inverse (displacement/observations → force):

```bash
python src/scripts/run_ifem_from_u.py \
  --mesh data/output/u.xdmf \
  --dofs data/output/u.dofs.npz \
  --config src/config/material.yaml \
  --Nx 80 --Ny 60 \
  --pc jacobi --ksp cg --ksp_rtol 1e-7 --ksp_atol 0 --ksp_monitor
```

### Render synthetic images

#### 1. new way : render_sensor_frame.py

Supports **All sensor types, including Particle, Gelsight, Tac3D**

**Particle style sensor**

```bash
python src/scripts/render_sensor_frame.py \
  --sensor particle_vts \
  --config src/config/renderer.yaml \
  --out data/output/render_pts.png \
  --seed 123 \
  --supersample 2
```

**Gelsight style sensor**

```bash
python src/scripts/render_sensor_frame.py \
  --sensor gelsight_style \
  --config src/config/renderer_gelsight.yaml \
  --out data/output/render_gs.png
```

**Tac3D style sensor**

```bash
python src/scripts/render_sensor_frame.py \
  --sensor tac3d \
  --config src/config/renderer_tac3d.yaml \
  --out data/output/render_tac3d.png
```


#### 2. deprecated : render_camera_frame.py

Can only support **Particle style sensor**

```bash
python src/scripts/render_camera_frame.py   --config src/config/renderer.yaml   --out data/output/render.png   --seed 42   --supersample 2
```


**Note:**

To generate generate images with deformed state (gel with force field applied), you need to:

1. run the Forward FEM step to generate deformed mesh `u.xdmf`

2. modify the sensor's corresponding `renderer.yaml` (`renderer_tac3d.yaml`, `renderer_gelsight.yaml`):

from:

```yaml
deformation:
  mode: none
```

to:

```yaml
deformation:
  mode: xdmf
  xdmf_path: <path to u.xdmf>
```

---

### Generate Dataset for training

Single sample test:

```bash
python src/dataset/gen_sample.py \
    --outdir dataset/val/000001 \
    --material src/config/material.yaml \
    --renderer src/config/renderer.yaml \
    --mode shear  \
    --n_balls 1  --seed 41 \
    --save_flow \
    --debug_show
```

Generate dataset (many samples, in parallel):

```bash
python src/dataset/gen_dataset.py \
    --root dataset/train \
    --material src/config/material.yaml \
    --renderer src/config/renderer.yaml \
    --n 1000 --jobs 8 \
    --mode_mix 0.25,0.25,0.25,0.25 \
    --n_balls_min 1 --n_balls_max 2
```

### Training

Traing and infer test see: [src/nn/README.md](src/nn/README.md)

### End-to-end validation (**Not Implemented Yet**):

```bash
python src/scripts/run_end2end.py --preset torque --noise 0.01 --report data/output/report.json
```

## 4) Data layout

data/input/ — Forces, configs, or external inputs.

data/output/ — Displacements, camera frames, recovered forces, reports.

data/cache/ — Precomputed influence matrices (H_sparse.npz), mesh caches.

## 5) PyVista/VTK tips on Windows & WSL2 Ubuntu

1. Windows native Python: pip install pyvista pyvistaqt vtk and ensure a Qt backend (e.g., PySide6).

2. WSL2 with WSLg: GUI should “just work”.

3. Off-screen rendering (CI/headless):

```bash
export PYVISTA_OFF_SCREEN=true
```

Ensure Mesa/EGL OpenGL libs are present (mesa, libgl1).


## 6) TODO

Mesh + BC + forward FEM (linear elastic; near-incompressible; unit tests).

Influence matrix build + regularized least squares inverse (no constraints → add constraints).

Synthetic camera renderer (RGB multi-depth particles; PSF + noise).

End-to-end script + visualizations (2D/3D) + regression tests.

Performance tuning, config polishing, and documentation.
