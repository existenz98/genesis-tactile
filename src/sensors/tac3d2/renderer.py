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

import cv2
import numpy as np

from ..base import FEMOutputs, FrameBundle, Scene, SensorRenderer, TargetBundle
from ..registry import register_sensor

from .geometry import (
    euler_ypr_deg,
    reflect_points_x_plane,
    reflect_dirs_x_plane,
    project_pinhole,
    jacobian_uv_wrt_panel_uv_at_point,
    ellipse_from_jacobian,
)
from .grid import GridSpec, make_grid_uv
from .draw import draw_ellipses, draw_circles

log = logging.getLogger(__name__)


def _read_cfg(cfg: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_panel_disp_grid(path: str, rows: int, cols: int) -> np.ndarray:
    """Load panel-frame displacement grid (rows x cols x 3) in mm.
    Supports .npz (look for keys: 'disp', 'panel_uv_disp_mm', 'data') or .npy.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        key = None
        for k in ("panel_uv_disp_mm", "disp", "data"):
            if k in data:
                key = k
                break
        if key is None:
            raise ValueError(f"No displacement array found in {path}; tried keys 'panel_uv_disp_mm','disp','data'.")
        arr = np.array(data[key])
    elif path.endswith(".npy"):
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported grid deformation file: {path}")
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (rows, cols, 3):
        raise ValueError(f"Expected shape (rows, cols, 3)=({rows},{cols},3); got {arr.shape}")
    return arr


def _maybe_load_xdmf_sampler(path: Optional[str]):
    """
    Build a panel-frame sampler from XDMF:
    input: uvw in panel coords (mm), output: (Δu,Δv,Δw) in panel coords (mm).
    """
    if path is None:
        raise ValueError("deformation.mode='xdmf' requires deformation.path (XDMF)")
    try:
        from synth.deform import DeformField  # type: ignore
    except Exception as e:
        raise ImportError("Tac3D xdmf mode requires synth.deform.DeformField to be available") from e
    field = DeformField(mode="xdmf", xdmf_path=path)
    field.load()

    def sample_panel_disp(uvw_panel: np.ndarray):
        """
        Sample panel-frame displacement at given panel-frame uvw coordinates (e.g. dot centers).
        Returns
        - disp_panel: (N,3) float displacements in panel frame (du,dv,dw) in mm
        - valid_mask: (N,) bool valid samples
        """
        for fn in ("sample_panel", "sample_uv", "sample"):
            if hasattr(field, fn):
                sampler = getattr(field, fn)
                disp = sampler(uvw_panel)  # panel-frame mm, shape (N,3)
                disp = np.asarray(disp, dtype=float)
                valid = np.all(np.isfinite(disp), axis=1)
                disp[~valid] = 0.0
                return disp, valid
        raise AttributeError("DeformField lacks a panel-coordinate sampler")

    return sample_panel_disp


@register_sensor("tac3d2")
class Tac3D2Renderer(SensorRenderer):
    """Tac3D mirror-stereo dot-grid renderer.

    Config schema (YAML):
    ----------------------
    camera:
      model: pinhole
      img_wh: [640, 480]
      fx_px: 180.0
      fy_px: 180.0
      cx_px: 320.0
      cy_px: 240.0
    mirror:
      distance_mm: 5.0        # mirror plane x = -distance_mm (normal +X)
    panel:
      rows: 20
      cols: 20
      spacing_mm: 1.5
      dot_radius_mm: 0.3
      pose_rest:
        center_mm: [5.0, 0.0, 20.0]
        yaw_deg: 0.0
        pitch_deg: 0.0
        roll_deg: 0.0
    appearance:
      background_bgr: [10, 20, 60]
      dot_color_bgr:  [250, 250, 255]
      ellipses: true
    deformation:
      mode: none              # none | xdmf
      path: null              # for xdmf: path to XDMF; requires synth.deform.DeformField

    Notes:
    ------
    - All displacements are in the panel frame (du, dv, dw) in mm.
    - Rendering composes mirrored (left) and real (right) views onto one image.
    """

    def name(self) -> str:
        return "tac3d2"

    def version(self) -> str:
        return "0.1.0"

    def modalities(self) -> Dict[str, str]:
        return {"image_bgr": "HxWx3 uint8 (OpenCV BGR)"}

    # ---- Core rendering ----
    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        cfg = dict(self.cfg or {})

        # ---- Camera configuration ----
        cam_cfg = cfg.get("camera", {})
        img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])
        fx = float(cam_cfg.get("fx_px", 180.0))
        fy = float(cam_cfg.get("fy_px", fx))
        cx = float(cam_cfg.get("cx_px", img_w / 2.0))
        cy = float(cam_cfg.get("cy_px", img_h / 2.0))

        # ---- Mirror plane ----
        mirror_cfg = cfg.get("mirror", {})
        d_mm = float(mirror_cfg.get("distance_mm", 5.0))  # plane x = -d

        # ---- Panel / grid ----
        panel_cfg = cfg.get("panel", {})
        rows = int(panel_cfg.get("rows", 20))
        cols = int(panel_cfg.get("cols", 20))
        spacing = float(panel_cfg.get("spacing_mm", 1.5))
        r_dot = float(panel_cfg.get("dot_radius_mm", 0.3))

        pose = panel_cfg.get("pose_rest", {})
        tx, ty, tz = [float(v) for v in pose.get("center_mm", [5.0, 0.0, 20.0])]
        yaw, pitch, roll = float(pose.get("yaw_deg", 0.0)), float(pose.get("pitch_deg", 0.0)), float(pose.get("roll_deg", 0.0))
        R_p = euler_ypr_deg(yaw, pitch, roll)  # 3x3
        t_p = np.array([tx, ty, tz], dtype=float)  # 3,

        # Tangent directions (world) for +u_p and +v_p (columns of R_p)
        r1 = R_p[:, 0].reshape(1, 3)   # shape (1,3) for broadcasting
        r2 = R_p[:, 1].reshape(1, 3)

        # ---- Generate grid in panel frame ----
        uv_panel, rc = make_grid_uv(GridSpec(rows=rows, cols=cols, spacing_mm=spacing, origin_center=True))  # (N,2), (N,2)
        N = uv_panel.shape[0]
        uvw_panel = np.concatenate([uv_panel, np.zeros((N, 1), dtype=float)], axis=1)  # (N,3)

        # ---- Deformation (panel frame) ----
        def_cfg = cfg.get("deformation", {"mode": "none"})
        mode = str(def_cfg.get("mode", "none")).lower()

        # Panel-frame displacement at dot centers: (du, dv, dw) in mm
        disp_panel = np.zeros((N, 3), dtype=float)
        disp_valid = np.ones((N,), dtype=bool)

        if mode == "xdmf":
            path = def_cfg.get("path", None) or def_cfg.get("xdmf_path", None)
            sample_panel_disp = _maybe_load_xdmf_sampler(path)  # returns (disp, valid)
            disp_panel, disp_valid = sample_panel_disp(uvw_panel)  # queries in panel coords
            # sanitize
            good = np.isfinite(disp_panel).all(axis=1)
            disp_panel[~good] = 0.0
            disp_valid &= good
        elif mode == "none" or mode == "":
            pass
        else:
            raise ValueError(f"Unsupported deformation.mode: {mode}")

        # Deformed panel points in panel frame
        uvw_panel_def = uvw_panel + disp_panel  # (N,3)

        # Convert to world (camera) frame
        X_def = (R_p @ uvw_panel_def.T).T + t_p.reshape(1, 3)  # (N,3)
        X_rest = (R_p @ uvw_panel.T).T + t_p.reshape(1, 3)     # (N,3)

        # Mirror the deformed positions for the left view
        X_def_L = reflect_points_x_plane(X_def, d_mm)  # (N,3)

        # Project both views
        uv_R, valid_R = project_pinhole(X_def, fx, fy, cx, cy)  # (N,2), (N,)
        uv_L, valid_L = project_pinhole(X_def_L, fx, fy, cx, cy)

        # In-FOV masks
        in_img_R = (uv_R[:, 0] >= 0) & (uv_R[:, 0] < img_w) & (uv_R[:, 1] >= 0) & (uv_R[:, 1] < img_h)
        in_img_L = (uv_L[:, 0] >= 0) & (uv_L[:, 0] < img_w) & (uv_L[:, 1] >= 0) & (uv_L[:, 1] < img_h)
        mask_R = valid_R & in_img_R
        mask_L = valid_L & in_img_L

        # Ellipse parameters (optional foreshortening)
        app = cfg.get("appearance", {})
        ellipses = bool(app.get("ellipses", True))
        color_bgr = tuple(int(c) for c in app.get("dot_color_bgr", [250, 250, 255]))
        bg_bgr = tuple(int(c) for c in app.get("background_bgr", [10, 20, 60]))

        ss = max(1, int(app.get("supersample", 2))) # supersampling factor
        blur_sigma = float(app.get("blur_sigma_px", 1.0))

        import numpy as _np
        W_ss, H_ss = img_w * ss, img_h * ss
        img_ss = _np.zeros((H_ss, W_ss, 3), dtype=_np.uint8)
        img_ss[:, :] = _np.array(bg_bgr, dtype=_np.uint8).reshape(1, 1, 3)
        #img = _np.zeros((img_h, img_w, 3), dtype=_np.uint8)
        #img[:, :] = _np.array(bg_bgr, dtype=_np.uint8).reshape(1, 1, 3)

        if ellipses:
            # World tangent directions for real and mirrored views
            r1_world = r1.repeat(N, axis=0)  # (N,3)
            r2_world = r2.repeat(N, axis=0)
            # For mirrored view, reflect the tangent directions across +X plane (no translation)
            r1_m = reflect_dirs_x_plane(r1_world)
            r2_m = reflect_dirs_x_plane(r2_world)

            # Compute (a,b,angle) per dot for each view
            axes_R = _np.zeros((N, 2), dtype=float)
            ang_R = _np.zeros((N,), dtype=float)
            axes_L = _np.zeros((N, 2), dtype=float)
            ang_L = _np.zeros((N,), dtype=float)

            for i in range(N):
                if mask_R[i]:
                    X, Y, Z = float(X_def[i, 0]), float(X_def[i, 1]), float(X_def[i, 2])
                    J = jacobian_uv_wrt_panel_uv_at_point(X, Y, Z, r1_world[i], r2_world[i], fx, fy)
                    a, b, ang = ellipse_from_jacobian(r_dot, J)
                    axes_R[i, 0], axes_R[i, 1], ang_R[i] = a, b, ang
                if mask_L[i]:
                    X, Y, Z = float(X_def_L[i, 0]), float(X_def_L[i, 1]), float(X_def_L[i, 2])
                    J = jacobian_uv_wrt_panel_uv_at_point(X, Y, Z, r1_m[i], r2_m[i], fx, fy)
                    a, b, ang = ellipse_from_jacobian(r_dot, J)
                    axes_L[i, 0], axes_L[i, 1], ang_L[i] = a, b, ang

            # Draw mirrored (left) first, then real (right) on top — in supersampled space
            try:
                #draw_ellipses(img, uv_L, axes_L, ang_L, color_bgr=color_bgr, in_fov_mask=mask_L, thickness=-1)
                #draw_ellipses(img, uv_R, axes_R, ang_R, color_bgr=color_bgr, in_fov_mask=mask_R, thickness=-1)
                uv_L_ss = uv_L * ss
                uv_R_ss = uv_R * ss
                axes_L_ss = axes_L * ss
                axes_R_ss = axes_R * ss
                draw_ellipses(img_ss, uv_L_ss, axes_L_ss, ang_L, color_bgr=color_bgr, in_fov_mask=mask_L, thickness=-1)
                draw_ellipses(img_ss, uv_R_ss, axes_R_ss, ang_R, color_bgr=color_bgr, in_fov_mask=mask_R, thickness=-1)
            except Exception as e:
                log.exception("Failed to draw ellipses; falling back to circles.")
                ellipses = False  # fallback

        if not ellipses:
            # Fallback: isotropic radius by average focal scaling
            Z_R = np.maximum(X_def[:, 2], 1e-6)
            Z_L = np.maximum(X_def_L[:, 2], 1e-6)
            r_R = 0.5 * (fx + fy) * (r_dot / Z_R)
            r_L = 0.5 * (fx + fy) * (r_dot / Z_L)
            #draw_circles(img, uv_L, r_L, color_bgr=color_bgr, in_fov_mask=mask_L, thickness=-1)
            #draw_circles(img, uv_R, r_R, color_bgr=color_bgr, in_fov_mask=mask_R, thickness=-1)
            draw_circles(img_ss, uv_L * ss, r_L * ss, color_bgr=color_bgr, in_fov_mask=mask_L, thickness=-1)
            draw_circles(img_ss, uv_R * ss, r_R * ss, color_bgr=color_bgr, in_fov_mask=mask_R, thickness=-1)

        # ---- Pack outputs ----

        # Gaussian blur at supersampled resolution
        if blur_sigma > 0:
            img_ss = cv2.GaussianBlur(img_ss, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma, borderType=cv2.BORDER_DEFAULT)

        # Downscale to target resolution (bicubic)
        if ss > 1:
            img = cv2.resize(img_ss, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        else:
            img = img_ss

        # ---- Pack outputs ----
        modalities = {"image_bgr": img}

        meta = {
            "renderer": self.name(),
            "version": self.version(),
            "image_size": (img_w, img_h),
            "camera": {"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy},
            "mirror": {"distance_mm": d_mm},
            "panel": {
                "rows": rows, "cols": cols, "spacing_mm": spacing, "dot_radius_mm": r_dot,
                "pose_rest": {"center_mm": [tx, ty, tz], "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll},
            },
            "deformation_mode": mode,
            "counts": {"visible_L": int(mask_L.sum()), "visible_R": int(mask_R.sum())},
        }

        # Aux for debugging / inspection
        aux: Dict[str, Any] = {
            "dot_uv_panel": uv_panel.reshape(rows, cols, 2),
            "dot_world_rest": X_rest.reshape(rows, cols, 3),
            "dot_world_def": X_def.reshape(rows, cols, 3),
            "proj_L": uv_L.reshape(rows, cols, 2),
            "proj_R": uv_R.reshape(rows, cols, 2),
            "mask_in_fov_L": mask_L.reshape(rows, cols),
            "mask_in_fov_R": mask_R.reshape(rows, cols),
            "disp_valid_mask": disp_valid.reshape(rows, cols),
        }

        # Copiable ground-truth for training/analysis
        if mode in ("xdmf"):
            aux["panel_uv_disp_mm"] = (disp_panel.reshape(rows, cols, 3))

        return FrameBundle(modalities=modalities, metadata=meta, aux=aux)

    # ---- Targets export ----
    def export_targets(self, fem: FEMOutputs) -> TargetBundle:
        # For Tac3D, we expose panel-frame displacements if provided in fem.meta or elsewhere.
        t: Dict[str, Any] = {}
        if getattr(fem, "u_dofs", None) is not None:
            t["u_dofs"] = fem.u_dofs
        if getattr(fem, "force_top", None) is not None:
            t["force_top"] = fem.force_top
        # If the FEMOutputs.meta contains panel-frame per-dot disp, pass it through as canonical GT
        if isinstance(getattr(fem, "meta", None), dict):
            for k in ("panel_uv_disp_mm", "panel_disp_uv_mm"):
                if k in fem.meta:
                    t["panel_uv_disp_mm"] = fem.meta[k]
                    break
        return TargetBundle(targets=t)
