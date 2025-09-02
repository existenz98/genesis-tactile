# Sensor Plugin Layer

A plugin-like layer so the dataset/simulation pipeline can support multiple **vision tactile sensor imaging models** (particle-based VTS, GelSight-style, etc.) without entangling them with FEM or dataset writers.

## Key Concepts

- **SensorRenderer (base class)** — defines `render_frame()` which produces the
  observables for one frame (e.g., RGB image, dense optical flow).
- **Registry** — a factory to instantiate a renderer by name
  (`make_sensor("particle_vts", **cfg)`).
- **FEMOutputs / Scene** — passed to the renderer.

## File Overview

```
sensors/
├── base.py # abstract API: SensorRenderer + data containers
├── registry.py # register_sensor decorator + factory
└── particle_vts/
    └── renderer.py # adapter over existing synth/* modules
```

## Particle VTS Renderer

`particle_vts/renderer.py` is a *thin adapter* that calls `synth.particles`, `synth.deform`, `synth.camera`, `synth.raster_cv`, and `synth.optical_flow`. 

### Returned Modalities

- `image_rgb` — RGB render of the deformed particle layer
- `flow_dense` (optional) — dense optical flow built from particle motion

## Example usage:

```python
from sensors import FEMOutputs, Scene
from sensors.registry import make_sensor

# fem_results: obtain from fem.forward() (u.dofs, force_top, surface_mesh)
fem_results = FEMOutputs(u_dofs=u_dofs, force_top=force_top, surface_mesh=surface_mesh)

# camera: synth.camera model instance
scene = Scene(camera=camera)

renderer = make_sensor("particle_vts",
                       particles={"num_layers": 3, "density": 0.5},
                       rasterizer={"radius_px": 2.0, "blur_sigma": 0.8},
                       flow={"method": "TPS"})

frame = renderer.render_frame(fem_results, scene)

rgb = frame.modalities["image_rgb"]           # HxWx3 uint8
flow = frame.modalities.get("flow_dense", None)

targets = renderer.export_targets(fem_results)  # {"u_dofs": ..., "force_top": ...}
```


