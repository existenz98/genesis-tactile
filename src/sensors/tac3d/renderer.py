# -----------------------------------------------------------------------------
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.
#
# Patent Notice:
#   This software is provided under copyright only.
#   No license to any patents is granted or implied.
#   Users are responsible for ensuring that their use of this software,
#   especially in commercial applications, does not infringe on any
#   third-party patents (e.g., tactile sensor hardware, methods).
#
# Citation:
#   If you use this code in academic work, please cite the associated
#   publication(s) when available.
# -----------------------------------------------------------------------------


"""
Acorn Tac3D style renderer

- big circular dot grid; Z from perspective size.
- optional mirror-stereo mode (two panels in one image)

How to use:
python src/scripts/render_sensor_frame.py \
  --sensor tac3d \
  --config src/config/renderer_tac3d.yaml \
  --out data/output/tac3d_frame.png

"""



from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

from ..base import FEMOutputs, FrameBundle, Scene, SensorRenderer
from ..registry import register_sensor

# Reuse your existing camera + FEM sampler
from synth.camera import PinholeCamera
from synth.deform import DeformField

log = logging.getLogger(__name__)


# ------------------------------- small utils --------------------------------
def _build_camera_from_cfg(cam_cfg: Dict[str, Any], view_mm: Tuple[float, float]) -> PinholeCamera:
    img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])
    Lx_mm, Ly_mm = float(view_mm[0]), float(view_mm[1])
    Z_ref_mm = float(cam_cfg.get("z_ref_mm", 5.0))
    z_cam_mm = float(cam_cfg.get("z_cam_mm", 5.0))
    return PinholeCamera.from_viewbox(img_w, img_h, Lx_mm, Ly_mm, Z_ref_mm, z_cam_mm)


def _build_deform_field_from_cfg(def_cfg: Dict[str, Any]) -> DeformField:
    mode = str(def_cfg.get("mode", "none"))
    xdmf_path = def_cfg.get("xdmf_path")
    field = DeformField(mode=mode, xdmf_path=xdmf_path)
    if field.mode == "xdmf":
        field.load()
    return field


def _regular_dot_grid_xy0(
    Lx_mm: float, Ly_mm: float, spacing_mm: float, margin_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regular grid within [0, Lx] × [0, Ly] with 'margin_mm' keepout from edges.
    Returns (Xg, Yg) shaped (Ny, Nx), in mm, origin at top-left (x right, y down).
    """
    x0 = 0.0 + margin_mm
    y0 = 0.0 + margin_mm
    x1 = Lx_mm - margin_mm
    y1 = Ly_mm - margin_mm
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid margin: reduces view box to non-positive extent.")
    Nx = max(2, int(np.floor((x1 - x0) / spacing_mm)) + 1)
    Ny = max(2, int(np.floor((y1 - y0) / spacing_mm)) + 1)
    xs = x0 + spacing_mm * np.arange(Nx, dtype=np.float32)
    ys = y0 + spacing_mm * np.arange(Ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    return Xg, Yg


def _render_panel_image(
    cam: PinholeCamera,
    img_w: int,
    img_h: int,
    def_px: np.ndarray,
    rad_px_def: np.ndarray,
    dot_color: Tuple[int, int, int],
    bg_bgr: Tuple[int, int, int],
    supersample: int,
) -> np.ndarray:
    """Draw a dot panel into an image with optional supersampling."""
    ss = max(1, int(supersample))
    Wss, Hss = img_w * ss, img_h * ss
    canvas = np.zeros((Hss, Wss, 3), dtype=np.uint8)
    canvas[...] = np.array(bg_bgr, dtype=np.uint8).reshape(1, 1, 3)

    def draw_circle(u: float, v: float, r: float):
        uu = int(round(u * ss))
        vv = int(round(v * ss))
        rr = max(1, int(round(r * ss)))
        if 0 <= uu < Wss and 0 <= vv < Hss:
            cv2.circle(canvas, (uu, vv), rr, dot_color, thickness=-1, lineType=cv2.LINE_AA)

    for (u, v), rpx in zip(def_px, rad_px_def):
        draw_circle(float(u), float(v), float(rpx))

    if ss > 1:
        return cv2.resize(canvas, (img_w, img_h), interpolation=cv2.INTER_AREA)
    return canvas


# ----------------------------- renderer class --------------------------------
@register_sensor("tac3d")
class Tac3DRenderer(SensorRenderer):
    """
    Acorn Tac3D-style renderer:
      - Big circular dots on a regular grid at membrane plane z = surface.z_mm
      - XY from dot-center displacement (pixels)
      - Z from dot size via perspective scaling: r_px ∝ 1 / Z (non-telecentric pinhole)

    Mirror-stereo mode:
      - Two virtual viewpoints with a horizontal baseline are rendered as two panels
        (left and right) in a single composite image.

    Modalities (superset; some keys present only when mirror.enabled=true):
      image_bgr                           : (H, W_out, 3) uint8
      dots_rest_px / def_px               : (N,2) float32         [single-panel mode]
      dots_rest_xyz / def_xyz             : (N,3) float32 (mm)
      dot_radius_px_rest/def              : (N,)  float32

      dots_rest_px_left / def_px_left     : (N,2) float32         [mirror mode]
      dots_rest_px_right / def_px_right   : (N,2) float32
      dot_radius_px_rest_left/def_left    : (N,)  float32
      dot_radius_px_rest_right/def_right  : (N,)  float32
    """

    def name(self) -> str:
        return "tac3d"

    def version(self) -> str:
        return "0.2.0"

    def modalities(self) -> Dict[str, str]:
        # Superset description; not all keys appear every time.
        return {
            "image_bgr": "HxWx3 uint8 (OpenCV BGR, single or composite)",
            "dots_rest_px": "Nx2 float32",
            "dots_def_px": "Nx2 float32",
            "dots_rest_xyz": "Nx3 float32 (mm)",
            "dots_def_xyz": "Nx3 float32 (mm)",
            "dot_radius_px_rest": "N float32",
            "dot_radius_px_def": "N float32",
            "dots_rest_px_left": "Nx2 float32",
            "dots_def_px_left": "Nx2 float32",
            "dots_rest_px_right": "Nx2 float32",
            "dots_def_px_right": "Nx2 float32",
            "dot_radius_px_rest_left": "N float32",
            "dot_radius_px_def_left": "N float32",
            "dot_radius_px_rest_right": "N float32",
            "dot_radius_px_def_right": "N float32",
        }

    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        cfg = dict(self.cfg or {})

        # --- Camera & view ---
        cam: Optional[PinholeCamera] = getattr(scene, "camera", None)
        view_wh = cfg.get("view_mm", [40.0, 30.0])
        Lx_mm, Ly_mm = float(view_wh[0]), float(view_wh[1])

        if cam is None:
            cam_cfg = cfg.get("camera", {})
            if not cam_cfg:
                raise ValueError("Missing 'camera' config and Scene.camera is None.")
            cam = _build_camera_from_cfg(cam_cfg, (Lx_mm, Ly_mm))
        else:
            cam_cfg = cfg.get("camera", {"img_wh": [cam.img_w, cam.img_h]})

        img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])

        # --- Surface (membrane) Z ---
        surf_cfg = cfg.get("surface", {})
        z_surf_mm = float(surf_cfg.get("z_mm", 0.0))  # e.g., 4.0 mm (top surface)

        # --- Dots & colors ---
        dots_cfg = cfg.get("dots", {})
        spacing_mm = float(dots_cfg.get("spacing_mm", 1.6))
        diameter_mm = float(dots_cfg.get("diameter_mm", 0.9))
        margin_mm = float(dots_cfg.get("margin_mm", 0.8))
        dot_color = tuple(int(c) for c in dots_cfg.get("color_bgr", [245, 245, 255]))  # bright slightly bluish white

        bg_bgr = tuple(int(c) for c in cfg.get("background_bgr", [20, 20, 60]))
        supersample = int(cfg.get("supersample", 2))

        # --- Mirror-stereo config ---
        mirror_cfg = cfg.get("mirror", {})
        mirror_enabled = bool(mirror_cfg.get("enabled", False))
        baseline_mm = float(mirror_cfg.get("baseline_mm", 6.0))   # distance between (virtual) left/right camera centers
        panel_gap_px = int(mirror_cfg.get("panel_gap_px", 12))
        flip_left = bool(mirror_cfg.get("flip_left", False))      # optional: left panel horizontal flip

        # --- Deformation field ---
        def_cfg = cfg.get("deformation", {"mode": "none"})
        field = _build_deform_field_from_cfg(def_cfg)

        # --- Build dot grid (rest) at Z=z_surf_mm in [0,Lx]×[0,Ly] coords ---
        Xg, Yg = _regular_dot_grid_xy0(Lx_mm, Ly_mm, spacing_mm, margin_mm)  # (Ny,Nx)
        Ny, Nx = Xg.shape
        N = Ny * Nx

        rest_xyz = np.stack(
            [Xg.ravel(), Yg.ravel(), np.full(N, z_surf_mm, dtype=np.float32)],
            axis=-1
        ).astype(np.float32)

        if field.mode != "none":
            disp = field.sample(rest_xyz).astype(np.float32)  # (N,3) mm
        else:
            disp = np.zeros_like(rest_xyz, dtype=np.float32)
        def_xyz = rest_xyz + disp

        # -------------------- Single-panel (legacy) path ---------------------
        if not mirror_enabled:
            # Project via the main camera
            rest_px, Z_rest = cam.project_mm(rest_xyz)  # (N,2), (N,)
            def_px,  Z_def  = cam.project_mm(def_xyz)   # (N,2), (N,)

            r_mm = np.full((N,), 0.5 * diameter_mm, dtype=np.float32)
            rad_px_rest = cam.radius_mm_to_px(r_mm, Z_rest).astype(np.float32)
            rad_px_def  = cam.radius_mm_to_px(r_mm, Z_def ).astype(np.float32)

            img_bgr = _render_panel_image(cam, img_w, img_h, def_px, rad_px_def, dot_color, bg_bgr, supersample)

            modalities: Dict[str, Any] = {
                "image_bgr": img_bgr,
                "dots_rest_px": rest_px.astype(np.float32),
                "dots_def_px": def_px.astype(np.float32),
                "dots_rest_xyz": rest_xyz.astype(np.float32),
                "dots_def_xyz": def_xyz.astype(np.float32),
                "dot_radius_px_rest": rad_px_rest,
                "dot_radius_px_def": rad_px_def,
            }

            meta = {
                "renderer": self.name(),
                "version": self.version(),
                "mode": "single_panel",
                "image_size": (img_w, img_h),
                "grid_shape": (Ny, Nx),
                "spacing_mm": spacing_mm,
                "diameter_mm": diameter_mm,
                "surface_z_mm": z_surf_mm,
                "supersample": supersample,
            }
            return FrameBundle(modalities=modalities, metadata=meta, aux={})

        # -------------------- Mirror-stereo (two panels) ---------------------
        # Effective virtual cameras via world-X shift ±B/2
        half_B = 0.5 * baseline_mm

        def shiftX(arr: np.ndarray, dx_mm: float) -> np.ndarray:
            out = arr.copy()
            out[:, 0] = out[:, 0] + dx_mm
            return out

        # Left panel
        rest_xyz_L = shiftX(rest_xyz, +half_B)
        def_xyz_L  = shiftX(def_xyz,  +half_B)
        rest_px_L, Z_rest_L = cam.project_mm(rest_xyz_L)
        def_px_L,  Z_def_L  = cam.project_mm(def_xyz_L)
        r_mm = np.full((N,), 0.5 * diameter_mm, dtype=np.float32)
        rad_px_rest_L = cam.radius_mm_to_px(r_mm, Z_rest_L).astype(np.float32)
        rad_px_def_L  = cam.radius_mm_to_px(r_mm, Z_def_L ).astype(np.float32)

        # Right panel
        rest_xyz_R = shiftX(rest_xyz, -half_B)
        def_xyz_R  = shiftX(def_xyz,  -half_B)
        rest_px_R, Z_rest_R = cam.project_mm(rest_xyz_R)
        def_px_R,  Z_def_R  = cam.project_mm(def_xyz_R)
        rad_px_rest_R = cam.radius_mm_to_px(r_mm, Z_rest_R).astype(np.float32)
        rad_px_def_R  = cam.radius_mm_to_px(r_mm, Z_def_R ).astype(np.float32)

        # Optional mirror flip for the left panel (cosmetic realism)
        if flip_left:
            def_px_L = def_px_L.copy()
            def_px_L[:, 0] = (img_w - 1) - def_px_L[:, 0]
            rest_px_L = rest_px_L.copy()
            rest_px_L[:, 0] = (img_w - 1) - rest_px_L[:, 0]

        # Render panels separately
        img_L = _render_panel_image(cam, img_w, img_h, def_px_L, rad_px_def_L, dot_color, bg_bgr, supersample)
        img_R = _render_panel_image(cam, img_w, img_h, def_px_R, rad_px_def_R, dot_color, bg_bgr, supersample)

        # Compose side-by-side with a gap
        W_out = img_w * 2 + panel_gap_px
        H_out = img_h
        out = np.zeros((H_out, W_out, 3), dtype=np.uint8)
        out[...] = np.array(bg_bgr, dtype=np.uint8).reshape(1, 1, 3)

        # Paste
        out[:, 0:img_w, :] = img_L
        out[:, img_w + panel_gap_px : img_w + panel_gap_px + img_w, :] = img_R

        # Offsets of panels in the composite (for downstream use)
        panel_offsets_px = {
            "left":  (0, 0),
            "right": (img_w + panel_gap_px, 0),
        }

        # Shift pixel coordinates into composite image space (so they match the image)
        def shift_px_to_composite(px: np.ndarray, panel: str) -> np.ndarray:
            px2 = px.copy()
            off = panel_offsets_px[panel]
            px2[:, 0] += off[0]
            px2[:, 1] += off[1]
            return px2

        def_px_L_comp  = shift_px_to_composite(def_px_L,  "left")
        rest_px_L_comp = shift_px_to_composite(rest_px_L, "left")
        def_px_R_comp  = shift_px_to_composite(def_px_R,  "right")
        rest_px_R_comp = shift_px_to_composite(rest_px_R, "right")

        # Package outputs
        modalities: Dict[str, Any] = {
            "image_bgr": out,
            "dots_rest_xyz": rest_xyz.astype(np.float32),
            "dots_def_xyz": def_xyz.astype(np.float32),

            "dots_rest_px_left":  rest_px_L_comp.astype(np.float32),
            "dots_def_px_left":   def_px_L_comp.astype(np.float32),
            "dot_radius_px_rest_left": rad_px_rest_L,
            "dot_radius_px_def_left":  rad_px_def_L,

            "dots_rest_px_right": rest_px_R_comp.astype(np.float32),
            "dots_def_px_right":  def_px_R_comp.astype(np.float32),
            "dot_radius_px_rest_right": rad_px_rest_R,
            "dot_radius_px_def_right":  rad_px_def_R,
        }

        meta = {
            "renderer": self.name(),
            "version": self.version(),
            "mode": "mirror_stereo",
            "image_size": (H_out, W_out),
            "panel_image_size": (img_h, img_w),
            "panel_gap_px": panel_gap_px,
            "panel_offsets_px": panel_offsets_px,
            "grid_shape": (Ny, Nx),
            "spacing_mm": spacing_mm,
            "diameter_mm": diameter_mm,
            "surface_z_mm": z_surf_mm,
            "baseline_mm": baseline_mm,
            "supersample": supersample,
            "flip_left": flip_left,
        }
        return FrameBundle(modalities=modalities, metadata=meta, aux={})