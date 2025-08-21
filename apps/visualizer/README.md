# Visualizer (Engineer Mode)

A Qt-based desktop application for **real‑time visualization** of the tactile sensing pipeline.

This app subscribes (locally) to the runtime via **SDK**,
and renders:

- **Camera view** (RGB)
- **Optical Flow** (HSV color + quiver arrows)
- **Force field** (pressure heatmap + shear quiver arrows)
- **3D Force** (PyVista: translucent surface colored by pressure + downward arrows)
- **Live stats** (sequence #, FPS, algorithm, scales)


---

This app depends on the **SDK** package located at `sdk/python` (`tacto6d`).

---

## 2) Prerequisites

- The runtime publishing locally via SHM + IPC with defaults:
  - **Notify**: `ipc:///tmp/tacto6d.frame`
  - **Control**: `ipc:///tmp/tacto6d.ctrl`
  - **SHM name**: `tacto6d_frame0` (the SDK discovers this from notify)
- Install the SDK (editable) so the app can import it:

```bash
pip install -e sdk/python
```

## 3) Install app dependencies

From repo root:

```
pip install -r apps/visualizer/requirements.txt

```

Tested combos: PySide6 + pyvistaqt 0.11 + pyvista >= 0.43.

## 4) Run

Start runtime (must be publishing frames).

Launch the Visualizer:

```
python apps/visualizer/run_visualizer.py
```

The app auto‑connects to the default local endpoints and begins rendering the latest frame.

## 5) Configuration

Edit apps/visualizer/visualizer/config.py

