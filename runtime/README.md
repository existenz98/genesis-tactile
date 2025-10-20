# Vision-Tactile 6D Force Runtime

**Realtime runtime to estimate dense 6-DoF (Fx, Fy, Fz, τx, τy, τz) tactile force fields** from a vision-based tactile sensor.  
It ingests frames from a **real camera** or a **virtual camera** (video / image folder), runs robust **preprocessing** (undistortion, illumination/color compensation, tracker separation, optical flow), and then feeds multiple **force solvers** (iFEM, physics-based, CNN).  
The system includes a **live visualizer** and a **developer SDK** for downstream apps.

---

## Features

- **Unified streaming pipeline**: camera / video / folder sources drive the same code path
- **Photometric compensation**: patchwise brightness + spatial white-balance fields
- **Color unmixing**: separate different tracker populations (e.g., RGB particles) in linear space
- **Per-color optical flow**: DIS / TV-L1 / Farnebäck backends (OpenCV)
- **Modular force estimation** (pluggable):
  - iFEM / inverse optimization *(planned)*
  - Physics approximations (divergence/strain-to-pressure, etc.) *(planned)*
  - CNN regression *(planned)*
- **Live debug windows** (OpenCV HighGUI) and **video export**
- **SDK-style outputs** for integration with external tools (visualization, robotics stacks)

---

## Directory Layout

## Install

```bash
# Python ≥ 3.9 recommended
pip install -U numpy pillow opencv-python opencv-contrib-python imageio imageio-ffmpeg pyyaml
```

### Note

opencv-contrib-python is required for DIS and TV-L1.

## Quick Start

Run from project root (recommended)

```bash
# Show arguments
python -m runtime.run_demo --help
```
### Examples

Video input, gray flow only (no unmix/compensation), show 3D and 2D debug windows

```bash
python -m runtime.run_demo \
  --source video \
  --input data/raw/camera_frames.avi \
  --loop \
  --compensation skip \
  --unmix skip \
  --vis3d --vis2d
```

Video input, per-color flow, use DIS optical flow algo, write videos for debug

```bash
python -m runtime.run_demo \
  --source video \
  --input data/raw/camera_frames.avi \
  --color_flow \
  --write_videos \
  --out outputs \
  --flow dis
```

Folder of frames at 30 FPS

```bash
python -m runtime.run_demo \
  --source folder \
  --input data/raw/frames \
  --fps 30 \
  --color_flow \
  --write_videos

```

USB camera (device 0), live windows

```bash
python -m runtime.run_demo \
  --source camera \
  --device 0 \
  --color_flow
```



## Processing Pipeline

1. **Input source**  
   - Real UVC camera (OpenCV `VideoCapture`), or virtual camera (**video** / **folder**) with pacing.

2. **Camera undistortion** *(TODO)*  
   - Use `camera_model.py` (intrinsics/extrinsics) + `geo_calibrator.py` to undistort and map pixels to metric space.

3. **Photometric compensation**  
   - From the first frame, compute robust patchwise stats → spatial brightness field **`s(x)`** and per-channel white-balance fields **`w_R/G/B(x)`**.  
   - Mask highlights; apply in **linear RGB** to all frames (or per-frame mode).

4. **Color unmixing (tracker separation)**  
   - k-means++ in chromaticity (linear RGB) to estimate R/G/B basis; NNLS-like projection → component maps **`c_R, c_G, c_B`**.  
   - Optional supervision (future): user scribbles to lock basis vectors.

5. **Optical flow (per color &/or raw)**  
   - Backends: **DIS** (fast), **TV-L1** (robust), **Farnebäck** (baseline).  
   - Compute dense flow for each component **vs the first frame** (incremental mode TBD).  
   - Convert to color-coded visualization (HSV: direction→hue, magnitude→value).

6. **Force solvers (6D field)** *(plug-in, WIP)*  
   - **Physics approximation**: displacement/strain → pressure & shear → (Fx, Fy, Fz, τx, τy, τz).  
   - **iFEM / inverse optimization**: recover forces that best explain observed motion under material/BCs.  
   - **CNN regression**: learned mapping from preprocessed inputs to 6D force maps.

7. **Outputs**  
   - Live **debug windows** (input / compensated / segmentation / flows).  
   - **Video export**: `flow_R.mp4`, `flow_G.mp4`, `flow_B.mp4`, optional `flow_raw.mp4`.  
   - **SDK** publish (stub): topic-based messages (e.g., `flow.R`, `force.normal`, `force.tangent`), ready for gRPC/ZeroMQ integration.  **(WIP)**
