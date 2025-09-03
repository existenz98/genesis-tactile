# -----------------------------------------------------------------------------
# SPDX-License-Identifier: LicenseRef-YF-Research-NC-1.0
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# Licensed for academic research and non-commercial use only.
# Any Commercial Use (including production use or any use for commercial
# advantage) requires a separate written license from the copyright holder.
# See LICENSE-SRC-RESEARCH-NC for definitions and terms.
#
# Patent Notice:
#   No patent license is granted or implied. Users are responsible for
#   third-party patent clearance.
#
# Citation:
#   Please cite associated publications when available.
# -----------------------------------------------------------------------------


"""
Render a synthetic camera frame via the sensor plugin registry.

This script is a generic variant of render_camera_frame.py:
- Same CLI + one extra flag: --sensor
- Same YAML schema (camera, view_mm, background_bgr, layers, deformation)
- Works with any registered sensor (particle_vts, gelsight_style, ...)

Examples:
  # Particle-based sensor
  python src/scripts/render_sensor_frame.py \
    --sensor particle_vts \
    --config src/config/renderer.yaml \
    --out data/output/render_pts.png \
    --seed 123 \
    --supersample 2

  # GelSight-style sensor
  python src/scripts/render_sensor_frame.py \
      --sensor gelsight_style \
      --config src/config/renderer_gelsight.yaml \
      --out data/output/render_gs.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
import cv2

# Ensure "./src" is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors import FEMOutputs, Scene
from sensors.registry import make_sensor, list_sensors

# Import known sensors so they self-register.
# Safe if a module is missing: we just skip it.
def _try_import(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception:
        pass

_try_import("sensors.particle_vts")
_try_import("sensors.gelsight_style")
_try_import("sensors.tac3d")
# ... here to add future plugin(s)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", type=str, default="particle_vts",
                   help="Registered sensor name (default: particle_vts)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--supersample", type=int, default=2)
    args = p.parse_args()

    # Load config (same schema as existing renderer.yaml)
    cfg = _load_yaml(args.config)

    # Apply CLI overrides to mirror legacy behavior
    cfg = dict(cfg)
    cfg["seed"] = int(args.seed)
    cfg["supersample"] = int(args.supersample)

    # Instantiate the requested sensor
    try:
        renderer = make_sensor(args.sensor, **cfg)
    except KeyError as e:
        available = ", ".join(list_sensors()) or "<none>"
        raise SystemExit(f"Unknown sensor '{args.sensor}'. Available: {available}") from e

    # For this script, FEM is not required (deformation can be loaded via cfg['deformation'])
    fem = FEMOutputs(u_dofs=None, force_top=None)
    scene = Scene(camera=None)  # renderer will build camera from cfg if needed

    # Render a single frame
    frame = renderer.render_frame(fem, scene)

    # Choose the most common modality name for images across sensors.
    # particle_vts returns 'image_bgr'; gelsight_style might return 'image_rgb'.
    # Prefer BGR if present (OpenCV write), fall back to RGB and convert.
    modalities = frame.modalities
    if "image_bgr" in modalities:
        img_bgr = modalities["image_bgr"]
    elif "image_rgb" in modalities:
        import numpy as np
        img_rgb = modalities["image_rgb"]
        # Convert RGB->BGR for OpenCV saving if it's a numpy array
        if hasattr(img_rgb, "shape") and img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
            img_bgr = img_rgb[..., ::-1]
        else:
            raise SystemExit("image_rgb found but it is not an HxWx3 array")
    else:
        raise SystemExit("No image modality found (expected 'image_bgr' or 'image_rgb').")

    # Save image
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img_bgr)

    print(f"[OK] saved {args.out}")


if __name__ == "__main__":
    main()
