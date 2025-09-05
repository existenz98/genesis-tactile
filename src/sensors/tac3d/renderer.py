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

import numpy as np
import cv2

from ..base import FEMOutputs, FrameBundle, Scene, SensorRenderer
from ..registry import register_sensor

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


def _euler_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float, order: str = "ZYX") -> np.ndarray:
    """Rotation matrix R from Euler angles in degrees. Default order: R = Rz * Ry * Rx."""
    rx = np.deg2rad(rx_deg); ry = np.deg2rad(ry_deg); rz = np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    M = {"X": Rx, "Y": Ry, "Z": Rz}
    R = np.eye(3, dtype=np.float32)
    for ax in order:
        R = M[ax] @ R
    return R.astype(np.float32)


def _apply_surface_pose(Xg: np.ndarray, Yg: np.ndarray, z_surf: float, Lx: float, Ly: float,
                        R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Map gel-frame top-surface points (x,y,z_surf) to world frame with pose (R,t).
    Xg, Yg: (Ny,Nx). Returns (N,3) world coords.
    """
    # local coords relative to the gel-frame center (x_c, y_c, z_surf)
    x0 = (Xg - 0.5 * Lx).ravel()
    y0 = (Yg - 0.5 * Ly).ravel()
    z0 = np.zeros_like(x0, dtype=np.float32)
    P_local = np.stack([x0, y0, z0], axis=-1).astype(np.float32)  # (N,3)
    Pw = (P_local @ R.T) + t[None, :]  # (N,3)
    return Pw.astype(np.float32)


def _apply_surface_pose_with_disp(Xg: np.ndarray, Yg: np.ndarray, z_surf: float, Lx: float, Ly: float,
                                  R: np.ndarray, t: np.ndarray,
                                  disp_gel: np.ndarray) -> np.ndarray:
    """
    Map gel-frame displaced points (x+ux, y+uy, z_surf+uz) to world frame.
    disp_gel: (N,3) displacement in gel frame at (x,y,z_surf).
    """
    x0 = (Xg - 0.5 * Lx).ravel()
    y0 = (Yg - 0.5 * Ly).ravel()
    # local displaced vector in gel frame
    P_local_def = np.stack([x0, y0, np.zeros_like(x0)], axis=-1).astype(np.float32) + disp_gel
    Pw = (P_local_def @ R.T) + t[None, :]
    return Pw.astype(np.float32)


def _reflect_points_across_plane(P: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Reflect 3D points P across a plane (n•(x - p) = 0).
    plane_normal must be unit length, shapes: P (N,3), p (3,), n (3,)
    """
    n = plane_normal.astype(np.float32)
    n = n / max(1e-12, float(np.linalg.norm(n)))
    v = P - plane_point[None, :]
    d = (v @ n)[:, None]  # (N,1)
    return (P - 2.0 * d * n[None, :]).astype(np.float32)


def _render_panel(img_w: int, img_h: int, def_px: np.ndarray, rad_px: np.ndarray,
                  dot_color: Tuple[int, int, int], bg_bgr: Tuple[int, int, int], supersample: int) -> np.ndarray:
    """Draw a dot panel into an image with optional supersampling."""
    ss = max(1, int(supersample))
    Wss, Hss = img_w * ss, img_h * ss
    canvas = np.full((Hss, Wss, 3), np.array(bg_bgr, dtype=np.uint8).reshape(1, 1, 3), dtype=np.uint8)

    def draw_circle(u: float, v: float, r: float):
        uu = int(round(u * ss))
        vv = int(round(v * ss))
        rr = max(1, int(round(r * ss)))
        if 0 <= uu < Wss and 0 <= vv < Hss:
            cv2.circle(canvas, (uu, vv), rr, dot_color, thickness=-1, lineType=cv2.LINE_AA)

    for (u, v), rpx in zip(def_px, rad_px):
        draw_circle(float(u), float(v), float(rpx))

    return cv2.resize(canvas, (img_w, img_h), interpolation=cv2.INTER_AREA) if ss > 1 else canvas


# ----------------------------- renderer class --------------------------------
@register_sensor("tac3d")
class Tac3DRenderer(SensorRenderer):
    """
    Acorn Tac3D-style renderer with:
      - Big circular dot grid on a membrane
      - Optional planar mirror → two-panel catadioptric stereo
      - Optional membrane tilt (Euler ZYX, deg)

    Geometry is physically consistent so that later triangulation using the
    provided metadata (intrinsics, surface pose, mirror plane, camera centers)
    reproduces the same 3D points.

    Modalities (superset; some keys present only in mirror mode):
      image_bgr                        : (H, W_out, 3) uint8
      dots_rest_xyz / dots_def_xyz     : (N,3) world (mm)
      dots_rest_px / dots_def_px       : (N,2) float32 (single-panel)
      dot_radius_px_rest/def           : (N,)  float32

      dots_*_px_left[_panel], dots_*_px_right[_panel] : (N,2) float32 (mirror mode)
      dot_radius_px_*_left/right        : (N,) float32
    """

    def name(self) -> str:
        return "tac3d"

    def version(self) -> str:
        return "0.3.0"

    # ---------------------------------------------------------------------
    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        cfg = dict(self.cfg or {})

        # ---- Camera & view ----
        cam: Optional[PinholeCamera] = getattr(scene, "camera", None)
        view_wh = cfg.get("view_mm", [40.0, 30.0])
        Lx, Ly = float(view_wh[0]), float(view_wh[1])

        if cam is None:
            cam_cfg = cfg.get("camera", {})
            if not cam_cfg:
                raise ValueError("Missing 'camera' config and Scene.camera is None.")
            cam = _build_camera_from_cfg(cam_cfg, (Lx, Ly))
        else:
            cam_cfg = cfg.get("camera", {"img_wh": [cam.img_w, cam.img_h]})

        img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])

        # Real camera center in world coords (matches PinholeCamera.project_mm)
        C = np.array([0.5 * Lx, 0.5 * Ly, -float(cam.z_cam_mm)], dtype=np.float32)

        # ---- Surface pose ----
        surf_cfg = cfg.get("surface", {})
        translation_mm = np.array(surf_cfg.get("translation_mm", [0.0, 0.0, 0.0]), dtype=np.float32)
        z_surf = float(surf_cfg.get("z_mm", 0.0))
        tilt_cfg = surf_cfg.get("tilt", {})

        # Rotation matrix of the surface
        rx = float(tilt_cfg.get("rx_deg", 0.0))
        ry = float(tilt_cfg.get("ry_deg", 0.0))
        rz = float(tilt_cfg.get("rz_deg", 0.0))
        R = _euler_deg_to_R(rx, ry, rz, order="ZYX")

        # Center of the surface (before any movement)
        surface_center = np.array([0.5 * Lx, 0.5 * Ly, z_surf], dtype=np.float32)

        # Apply translation to move the center of the surface (e.g., move it 3 mm to the left)
        new_surface_center = surface_center + translation_mm

        # ---- Dots & appearance ----
        dots_cfg = cfg.get("dots", {})
        spacing_mm = float(dots_cfg.get("spacing_mm", 1.6))
        diameter_mm = float(dots_cfg.get("diameter_mm", 0.9))
        margin_mm = float(dots_cfg.get("margin_mm", 0.8))
        dot_color = tuple(int(c) for c in dots_cfg.get("color_bgr", [245, 245, 255]))

        bg_bgr = tuple(int(c) for c in cfg.get("background_bgr", [20, 20, 60]))
        supersample = int(cfg.get("supersample", 2))

        # --- Mirror-stereo config ---
        mirror_cfg = cfg.get("mirror", {})
        mirror_enabled = bool(mirror_cfg.get("enabled", False))
        plane_n = np.asarray(mirror_cfg.get("plane_normal", [1.0, 0.0, 0.0]), dtype=np.float32)
        plane_p = np.asarray(mirror_cfg.get("plane_point_mm", [0.5 * Lx + 3.0, 0.5 * Ly, 0.0]), dtype=np.float32)
        # normalize
        if mirror_enabled:
            plane_n = plane_n / max(1e-12, float(np.linalg.norm(plane_n)))

        # ---- FEM deformation ----
        def_cfg = cfg.get("deformation", {"mode": "none"})
        field = _build_deform_field_from_cfg(def_cfg)

        # Build dot grid in gel frame (top surface z=z_surf)
        Xg, Yg = _regular_dot_grid_xy0(Lx, Ly, spacing_mm, margin_mm)  # (Ny,Nx)
        Ny, Nx = Xg.shape
        N = Ny * Nx

        # Sample displacement in GEL frame at (x,y,z_surf)
        P_gel = np.stack([Xg.ravel(), Yg.ravel(), np.full(N, z_surf, dtype=np.float32)], axis=-1)
        if field.mode != "none":
            disp_gel = field.sample(P_gel).astype(np.float32)  # (N,3) in gel axes
        else:
            disp_gel = np.zeros_like(P_gel, dtype=np.float32)

        # World positions (rest/def) with surface pose
        rest_xyz_w = _apply_surface_pose(Xg, Yg, z_surf, Lx, Ly, R, new_surface_center)                        # (N,3)
        def_xyz_w  = _apply_surface_pose_with_disp(Xg, Yg, z_surf, Lx, Ly, R, new_surface_center, disp_gel)    # (N,3)

        # ----------------------------- single panel -----------------------------
        if not mirror_enabled:
            rest_px, Z_rest = cam.project_mm(rest_xyz_w)
            def_px,  Z_def  = cam.project_mm(def_xyz_w)

            r_mm = np.full((N,), 0.5 * diameter_mm, dtype=np.float32)
            rad_px_rest = cam.radius_mm_to_px(r_mm, Z_rest).astype(np.float32)
            rad_px_def  = cam.radius_mm_to_px(r_mm, Z_def ).astype(np.float32)

            img = _render_panel(img_w, img_h, def_px, rad_px_def, dot_color, bg_bgr, supersample)

            modalities: Dict[str, Any] = {
                "image_bgr": img,
                "dots_rest_px": rest_px.astype(np.float32),
                "dots_def_px":  def_px.astype(np.float32),
                "dots_rest_xyz": rest_xyz_w.astype(np.float32),
                "dots_def_xyz":  def_xyz_w.astype(np.float32),
                "dot_radius_px_rest": rad_px_rest,
                "dot_radius_px_def":  rad_px_def,
            }
            meta = {
                "renderer": self.name(), "version": self.version(), "mode": "single_panel",
                "image_size": (img_h, img_w), "grid_shape": (Ny, Nx),
                "spacing_mm": spacing_mm, "diameter_mm": diameter_mm,
                "surface": {"z_mm": z_surf, "R": R.tolist(), "t": t.tolist(),
                            "tilt_deg": {"rx": rx, "ry": ry, "rz": rz}},
                "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                           "z_cam_mm": cam.z_cam_mm, "center_mm": C.tolist()},
                "supersample": supersample,
            }
            return FrameBundle(modalities=modalities, metadata=meta, aux={})

        # ----------------------------- mirror stereo ----------------------------
        # Reflect scene points to create the left panel view
        rest_xyz_m = _reflect_points_across_plane(rest_xyz_w, plane_p, plane_n)
        def_xyz_m  = _reflect_points_across_plane(def_xyz_w,  plane_p, plane_n)

        # Project both panels
        rest_px_R, Z_rest_R = cam.project_mm(rest_xyz_w)
        def_px_R,  Z_def_R  = cam.project_mm(def_xyz_w)
        rest_px_L, Z_rest_L = cam.project_mm(rest_xyz_m)
        def_px_L,  Z_def_L  = cam.project_mm(def_xyz_m)

        # Radii (size cue) from true depths to the (real) camera center
        r_mm = np.full((N,), 0.5 * diameter_mm, dtype=np.float32)
        rad_px_rest_R = cam.radius_mm_to_px(r_mm, Z_rest_R).astype(np.float32)
        rad_px_def_R  = cam.radius_mm_to_px(r_mm, Z_def_R ).astype(np.float32)
        rad_px_rest_L = cam.radius_mm_to_px(r_mm, Z_rest_L).astype(np.float32)
        rad_px_def_L  = cam.radius_mm_to_px(r_mm, Z_def_L ).astype(np.float32)

        # Render individual panels
        img_R = _render_panel(img_w, img_h, def_px_R, rad_px_def_R, dot_color, bg_bgr, supersample)
        img_L = _render_panel(img_w, img_h, def_px_L, rad_px_def_L, dot_color, bg_bgr, supersample)

        # Compose side-by-side
        W_out = img_w * 2
        H_out = img_h
        out = np.full((H_out, W_out, 3), np.array(bg_bgr, dtype=np.uint8).reshape(1, 1, 3), dtype=np.uint8)
        off_left  = (0, 0)
        off_right = (img_w, 0)
        out[0:img_h, 0:img_w, :] = img_L
        out[0:img_h, off_right[0]:off_right[0]+img_w, :] = img_R

        # Pixel coordinates in raw panel frames and in the composite frame
        def _shift(px: np.ndarray, off: Tuple[int, int]) -> np.ndarray:
            px2 = px.copy(); px2[:, 0] += off[0]; px2[:, 1] += off[1]; return px2

        rest_px_L_comp = _shift(rest_px_L, off_left)
        def_px_L_comp  = _shift(def_px_L,  off_left)
        rest_px_R_comp = _shift(rest_px_R, off_right)
        def_px_R_comp  = _shift(def_px_R,  off_right)

        # Virtual camera center (reflection of C)
        C_virtual = _reflect_points_across_plane(C[None, :], plane_p, plane_n)[0]
        baseline = float(np.linalg.norm(C_virtual - C))

        modalities: Dict[str, Any] = {
            "image_bgr": out,
            "dots_rest_xyz": rest_xyz_w.astype(np.float32),
            "dots_def_xyz":  def_xyz_w.astype(np.float32),

            "dots_rest_px_left_panel":  rest_px_L.astype(np.float32),
            "dots_def_px_left_panel":   def_px_L.astype(np.float32),
            "dots_rest_px_right_panel": rest_px_R.astype(np.float32),
            "dots_def_px_right_panel":  def_px_R.astype(np.float32),

            "dots_rest_px_left":  rest_px_L_comp.astype(np.float32),
            "dots_def_px_left":   def_px_L_comp.astype(np.float32),
            "dots_rest_px_right": rest_px_R_comp.astype(np.float32),
            "dots_def_px_right":  def_px_R_comp.astype(np.float32),

            "dot_radius_px_rest_left":  rad_px_rest_L,
            "dot_radius_px_def_left":   rad_px_def_L,
            "dot_radius_px_rest_right": rad_px_rest_R,
            "dot_radius_px_def_right":  rad_px_def_R,
        }

        meta = {
            "renderer": self.name(), "version": self.version(), "mode": "mirror_stereo",
            "image_size": (H_out, W_out),
            "panel_image_size": (img_h, img_w),
            "panel_offsets_px": {"left": off_left, "right": off_right},

            "surface": {"z_mm": z_surf, "R": R.tolist(), "t": new_surface_center.tolist(),
                        "tilt_deg": {"rx": rx, "ry": ry, "rz": rz},
                        "size_mm": [Lx, Ly]},

            "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                       "z_cam_mm": cam.z_cam_mm, "center_mm": C.tolist()},

            "mirror": {"plane_normal": plane_n.tolist(), "plane_point_mm": plane_p.tolist(),
                       "virtual_center_mm": C_virtual.tolist()},

            "grid_shape": (Ny, Nx),
            "spacing_mm": spacing_mm, "diameter_mm": diameter_mm,
            "supersample": supersample,
        }
        return FrameBundle(modalities=modalities, metadata=meta, aux={})
