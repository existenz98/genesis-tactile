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
Dual-Mirror style (e.g. Acorn Tac3D) synthetic renderer

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
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

from ..base import FEMOutputs, FrameBundle, Scene, SensorRenderer, TargetBundle
from ..registry import register_sensor

from .geometry import (
    rot_yaw_pitch_roll_deg,
    world_to_camera,
    reflect_points_plane,
    reflect_dirs_plane,
    project_pinhole,
    jacobian_uv_wrt_panel_uv_at_point_cam,
    ellipse_from_jacobian,
)
from .grid import GridSpec, make_grid_uv
from .draw import draw_ellipses, draw_circles

log = logging.getLogger(__name__)


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


def _build_deform_field_from_cfg(def_cfg: Dict[str, Any]):
    """
    Create a DeformField
    panel/world coords.
    """
    from synth.deform import DeformField  # type: ignore
    mode = str(def_cfg.get("mode", "none")).lower()
    xdmf_path = def_cfg.get("path", None) or def_cfg.get("xdmf_path", None)
    field = DeformField(mode=mode, xdmf_path=xdmf_path)
    if field.mode == "xdmf":
        field.load()
    return field


@register_sensor("tac3d2")
class Tac3D2Renderer(SensorRenderer):
    """Dual Mirror Stereo dot-grid renderer.

    World/panel frame: gel plane z=0, normal +Z; dots on z=0.

    Notes:
    ------
    - All displacements are in the panel frame (du, dv, dw) in mm.
    - Rendering composes two mirrored views into one image.
    """

    def name(self) -> str:
        return "tac3d2"

    def version(self) -> str:
        return "0.2.0"

    def modalities(self) -> Dict[str, str]:
        return {"image_bgr": "HxWx3 uint8 (OpenCV BGR)"}

    # ---- Core rendering ----
    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        cfg = dict(self.cfg or {})

        # ---- Intrinsics & canvas ----
        cam_cfg = cfg.get("camera", {})
        W, H = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])
        #W, H = int(_read_cfg(cam_cfg, ("img_wh", 0), 640)), int(_read_cfg(cam_cfg, ("img_wh", 1), 480))
        fx = float(cam_cfg.get("fx_px", 180.0))
        fy = float(cam_cfg.get("fy_px", fx))
        cx = float(cam_cfg.get("cx_px", W / 2.0))
        cy = float(cam_cfg.get("cy_px", H / 2.0))

        # ---- Camera pose (world) ----
        pose = cam_cfg.get("pose", {})
        Cx = float(pose.get("x_mm", 0.0))
        Cy = float(pose.get("y_mm", 0.0))
        Cz = float(pose.get("z_mm", -60.0))  # by default place camera at negative z looking +Z
        yaw_c = float(pose.get("yaw_deg", 0.0))
        pitch_c = float(pose.get("pitch_deg", 0.0))
        roll_c = float(pose.get("roll_deg", 0.0))
        R_wc = rot_yaw_pitch_roll_deg(yaw_c, pitch_c, roll_c)  # camera->world
        C_w = np.array([Cx, Cy, Cz], dtype=float)

        # ---- Mirrors ----
        mirrors_cfg: List[Dict[str, Any]] = cfg.get("mirrors", [])
        if not mirrors_cfg or len(mirrors_cfg) < 2:
            log.warning("Tac3D: expected two mirrors, got %d; proceeding with %d", len(mirrors_cfg), len(mirrors_cfg))

        mirrors: List[Dict[str, Any]] = []
        for m in mirrors_cfg:
            name = str(m.get("name", "mirror"))
            cxm, cym, czm = [float(v) for v in m.get("center_mm", [0.0, 0.0, 20.0])]
            yaw_m = float(m.get("yaw_deg", 0.0))
            pitch_m = float(m.get("pitch_deg", 0.0))
            roll_m = float(m.get("roll_deg", 0.0))
            R_m = rot_yaw_pitch_roll_deg(yaw_m, pitch_m, roll_m)
            #n_m = (R_m @ np.array([1.0, 0.0, 0.0], dtype=float))  # normal in +X, then rotate by yaw/pitch/roll
            n_m = (R_m @ np.array([0.0, 0.0, 1.0], dtype=float))  # normal in +Z, then rotate by yaw/pitch/roll
            p0_m = np.array([cxm, cym, czm], dtype=float)
            mirrors.append({"name": name, "n": n_m, "p0": p0_m, "R": R_m})

        # ---- Panel grid (world=panel) ----
        panel_cfg = cfg.get("panel", {})
        rows = int(panel_cfg.get("rows", 20))
        cols = int(panel_cfg.get("cols", 20))
        spacing = float(panel_cfg.get("spacing_mm", 1.5))
        r_dot = float(panel_cfg.get("dot_radius_mm", 0.3))
        # Panel physical spans
        half_w = 0.5 * (cols - 1) * spacing    # mm
        half_h = 0.5 * (rows - 1) * spacing    # mm

        # 2D grid points (location of dot centers) in panel frame
        uv_panel, rc = make_grid_uv(GridSpec(rows=rows, cols=cols, spacing_mm=spacing, origin_center=True))
        N = uv_panel.shape[0]   # number of dots
        # 3D in world/panel frame (u,v) -> (x,y,z)
        uvw_panel = np.concatenate([uv_panel, np.zeros((N,1), dtype=float)], axis=1)

        # ---- Deformation sampling (panel frame) ----
        def_cfg = cfg.get("deformation", {"mode": "none"})
        mode = str(def_cfg.get("mode", "none")).lower()


        grid_displacement = np.zeros((N,3), dtype=float)
        grid_valid = np.ones((N,), dtype=bool)

        if mode == "xdmf":
            # load deformation (FEM's output)
            field = _build_deform_field_from_cfg(def_cfg)

            # --- Sample at panel grid points ---
            # Membrane surface z (mm) in FEM
            surf_cfg = cfg.get("surface", {})
            z_surf_mm = float(surf_cfg.get("z_mm", 1.0))
            # apply offset to center the panel, because in FEM the mesh coordinate is put gel's corner is at (0,0),
            # instead of put gel's center at (0,0).
            ox, oy, oz = half_w, half_h, z_surf_mm
            q = uvw_panel.copy()
            q[:, 0] += ox
            q[:, 1] += oy
            q[:, 2] += oz

            # Query displacement, result is (N,3) in mm, same frame as mesh (panel/world)
            disp = field.sample(q)
            grid_displacement = np.asarray(disp, dtype=float)

            # handle invalid points (rarely happens)
            grid_valid = np.isfinite(grid_displacement).all(axis=1)
            grid_displacement[~grid_valid] = 0.0
        elif mode == "none" or mode == "":
            # grid_displacement as zero
            pass
        else:
            raise ValueError(f"Unsupported deformation.mode={mode}; use 'none' or 'xdmf'.")

        uvw_def = uvw_panel + grid_displacement  # world (panel) coords

        # ---- Appearance & supersampling ----
        app = cfg.get("appearance", {})
        color_bgr = tuple(int(c) for c in app.get("dot_color_bgr", [250, 250, 255]))
        bg_bgr = tuple(int(c) for c in app.get("background_bgr", [10, 20, 60]))
        ellipses = bool(app.get("ellipses", True))

        render_cfg = cfg.get("render", {})
        ss = max(1, int(render_cfg.get("supersample", 2)))
        blur_sigma = float(render_cfg.get("blur_sigma_px", 1.0))

        H_ss, W_ss = H * ss, W * ss
        img_ss = np.zeros((H_ss, W_ss, 3), dtype=np.uint8)
        img_ss[:, :] = np.array(bg_bgr, dtype=np.uint8).reshape(1,1,3)

        # Panel tangent directions in world (constant over the plane)
        r1_world = np.array([1.0, 0.0, 0.0], dtype=float)  # +u = +X
        r2_world = np.array([0.0, 1.0, 0.0], dtype=float)  # +v = +Y

        # ---- For each mirror, reflect and draw ----
        # We do NOT render the direct gel.
        all_proj = {}
        all_masks = {}
        for idx, m in enumerate(mirrors):
            n_m = m["n"]; p0_m = m["p0"]; name = m.get("name", f"mirror{idx+1}")

            # Reflect points and tangents in world frame
            X_ref = reflect_points_plane(uvw_def, n_m, p0_m)  # (N,3)
            r1_ref = reflect_dirs_plane(r1_world, n_m)        # (3,)
            r2_ref = reflect_dirs_plane(r2_world, n_m)        # (3,)

            # To camera frame
            X_cam = world_to_camera(X_ref, C_w, R_wc)         # (N,3)
            r1_cam = (R_wc.T @ r1_ref.reshape(3,1)).reshape(3,)  # rotate only (no translation)
            r2_cam = (R_wc.T @ r2_ref.reshape(3,1)).reshape(3,)

            # Project
            uv, valid = project_pinhole(X_cam, fx, fy, cx, cy)
            in_img = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
            mask = valid & in_img

            # Ellipse params
            if ellipses:
                axes = np.zeros((N,2), dtype=float)
                ang  = np.zeros((N,), dtype=float)
                for i in range(N):
                    if not mask[i]: continue
                    X, Y, Z = float(X_cam[i,0]), float(X_cam[i,1]), float(X_cam[i,2])
                    J = jacobian_uv_wrt_panel_uv_at_point_cam(X, Y, Z, r1_cam, r2_cam, fx, fy)
                    a, b, theta = ellipse_from_jacobian(r_dot, J)
                    axes[i,0], axes[i,1], ang[i] = a*ss, b*ss, theta
                draw_ellipses(img_ss, uv*ss, axes, ang, color_bgr=color_bgr, in_fov_mask=mask, thickness=-1)
            else:
                Zc = np.maximum(X_cam[:,2], 1e-6)
                r_px = 0.5*(fx+fy)*(r_dot / Zc) * ss
                draw_circles(img_ss, uv*ss, r_px, color_bgr=color_bgr, in_fov_mask=mask, thickness=-1)

            all_proj[name] = uv.reshape(rows, cols, 2)
            all_masks[name] = mask.reshape(rows, cols)

        # ---- Blur and downscale ----
        if blur_sigma > 0 and cv2 is not None:
            img_ss = cv2.GaussianBlur(img_ss, ksize=(0,0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        if cv2 is not None and ss > 1:
            img = cv2.resize(img_ss, (W,H), interpolation=cv2.INTER_CUBIC)
        else:
            img = img_ss if ss == 1 else img_ss[::ss, ::ss][:H, :W]

        # ---- Pack outputs ----
        modalities = {"image_bgr": img}
        meta = {
            "renderer": self.name(),
            "version": self.version(),
            "image_size": (W, H),
            "camera": {
                "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
                "pose": {"x_mm": Cx, "y_mm": Cy, "z_mm": Cz, "yaw_deg": yaw_c, "pitch_deg": pitch_c, "roll_deg": roll_c},
            },
            "mirrors": [
                {"name": m.get("name", f"mirror{idx+1}"),
                 "center_mm": m["p0"].tolist(),
                 "normal": (m["n"]/(np.linalg.norm(m["n"]) or 1.0)).tolist()}
                for idx, m in enumerate(mirrors)
            ],
            "panel": {"rows": rows, "cols": cols, "spacing_mm": spacing, "dot_radius_mm": r_dot},
            "deformation_mode": mode,
        }

        aux: Dict[str, Any] = {
            "dot_uv_panel": uv_panel.reshape(rows, cols, 2),
            "dot_world_def": uvw_def.reshape(rows, cols, 3),
            "disp_valid_mask": grid_valid.reshape(rows, cols),
        }
        for name, uv_rc in all_proj.items():
            aux[f"proj_{name}"] = uv_rc
            aux[f"mask_in_fov_{name}"] = all_masks[name]

        if mode in ("xdmf"):
            aux["panel_uv_disp_mm"] = grid_displacement.reshape(rows, cols, 3)

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
