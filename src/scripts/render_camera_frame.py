"""
Render synthetic camera frame of layered colored particles.

Example:
  python src/scripts/render_frame.py \
    --config src/config/renderer.yaml \
    --out data/output/render.png \
    --seed 123

The renderer:
- reads camera + layer specs from YAML,
- generates particles per layer,
- loads deformation field (xdmf) data and displaces particles position,
- renders with OpenCV (depth-sorted disks or random polygons),
- saves 8-bit PNG.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import yaml

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from synth.camera import PinholeCamera
from synth.particles import LayerSpec, generate_particles
from synth.deform import DeformField
from synth.raster_cv import render_frame_cv



def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_layers(cfg: dict) -> list[LayerSpec]:
    out = []
    for L in cfg["layers"]:
        out.append(LayerSpec(
            name=L["name"],
            z_min_mm=float(L["z_mm"][0]),
            z_max_mm=float(L["z_mm"][1]),
            color_bgr=tuple(int(c) for c in L["color_bgr"]),
            density_per_mm2=float(L["density_per_mm2"]),
            radius_mm=float(L["radius_mm"]),
            depth_atten=L.get("depth_brightness", "exp"),
            depth_beta=float(L.get("depth_beta", 0.2)),
            shape=L.get("shape", "disk"),
            poly_verts=int(L.get("poly_vertices", 0)),
        ))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--supersample", type=int, default=2)
    args = p.parse_args()

    cfg = load_yaml(args.config)

    # Camera
    cam_cfg = cfg["camera"]
    img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])
    Lx_mm, Ly_mm = float(cfg["view_mm"][0]), float(cfg["view_mm"][1])
    Z_ref_mm = float(cam_cfg.get("z_ref_mm", 5.0))
    z_cam_mm = float(cam_cfg.get("z_cam_mm", 5.0))
    cam = PinholeCamera.from_viewbox(img_w, img_h, Lx_mm, Ly_mm, Z_ref_mm, z_cam_mm)

    # Background color (BGR)
    bg_bgr = tuple(int(c) for c in cfg.get("background_bgr", [0, 0, 0]))

    # Layers
    layers = _load_layers(cfg)

    # Particles (initial)
    rng = np.random.default_rng(args.seed)
    parts = generate_particles(layers, Lx_mm, Ly_mm, rng)

    # Deformation
    def_cfg = cfg.get("deformation", {"mode": "none"})
    field = DeformField(mode=str(def_cfg.get("mode", "none")),
                        xdmf_path=def_cfg.get("xdmf_path"))
    if field.mode == "xdmf":
        field.load()
    
    if parts.xyz_mm.shape[0] > 0:
        disp = field.sample(parts.xyz_mm)
        parts.xyz_mm = parts.xyz_mm + disp  # apply deformation
    

    # Render
    img = render_frame_cv(cam, parts, layers, Lx_mm, Ly_mm, supersample=args.supersample, bg_color=bg_bgr)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(args.out), img)

    print(f"[OK] saved {args.out}  (N={parts.xyz_mm.shape[0]} particles)")



if __name__ == "__main__":
    main()