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
GelSight-style renderer (photometric normals + marker grid).

Minimal model:
- Build a per-pixel height map z'(x,y) from the FEM deformation sampled at z=0,
- Compute per-pixel normals via finite differences,
- Lambertian shading under 3 colored lights (R,G,B) → neutral gray at rest,
- Overlay a regular marker grid that moves with the surface,
- (Optional: dense optical flow from the marker grid via bilinear interpolation.)

Kept intentionally simple & fast,  No gamma/noise/speculars/shadows. 

Usage:
  python src/scripts/render_sensor_frame.py \
      --sensor gelsight_style \
      --config src/config/renderer_gelsight.yaml \
      --out data/output/render_gs.png


Expected YAML keys (see gelsight_renderer.yaml sample at the end of this file):
    camera: { img_wh, z_ref_mm, z_cam_mm }
    view_mm: [Lx, Ly]
    background_gray: 180
    lighting: { dir_r, dir_g, dir_b, gains_rgb: null|[gr,gg,gb] }
    markers: { spacing_mm, radius_px, color_bgr, jitter_px }
    deformation: { mode: "none"|"xdmf", xdmf_path: "..." }
    render: { grid_downsample, save_normal_map, save_depth_map, save_flow_from_markers }

Outputs (FrameBundle.modalities):
    image_bgr (HxWx3 uint8)
    markers_rest_px (M x 2 float32)
    markers_def_px  (M x 2 float32)
    markers_rest_xyz (M x 3 float32, mm)
    markers_def_xyz  (M x 3 float32, mm)
    flow_from_markers_dense (HxWx2 float32, px)          [optional]
    flow_from_markers_mask  (HxW uint8 in {0,255})       [optional]
    normal_map (HxWx3 float32, unit)                     [optional]
    depth_map  (HxW float32, mm)                         [optional]
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


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def _mm_grid_for_image(H: int, W: int, Lx_mm: float, Ly_mm: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Per-pixel membrane coordinates in mm, in the same convention expected by
    PinholeCamera.project_mm(): X in [0, Lx], Y in [0, Ly].
    +x right, +y down (image convention).
    """
    dx = Lx_mm / float(W)
    dy = Ly_mm / float(H)
    xs = (np.arange(W, dtype=np.float32) + 0.5) * dx         # 0 .. Lx
    ys = (np.arange(H, dtype=np.float32) + 0.5) * dy         # 0 .. Ly
    X_mm, Y_mm = np.meshgrid(xs, ys, indexing="xy")
    return X_mm, Y_mm, dx, dy


def _finite_diff_normals(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute normals from a height map z(x,y) using central differences.
    Returns (H, W, 3) unit normals with +z up when surface is flat.
    """
    H, W = z.shape
    dzdx = np.zeros_like(z, dtype=np.float32)
    dzdy = np.zeros_like(z, dtype=np.float32)

    # central differences interior
    dzdx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * dx)
    dzdy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2.0 * dy)

    # forward/backward at borders
    dzdx[:, 0] = (z[:, 1] - z[:, 0]) / dx
    dzdx[:, -1] = (z[:, -1] - z[:, -2]) / dx
    dzdy[0, :] = (z[1, :] - z[0, :]) / dy
    dzdy[-1, :] = (z[-1, :] - z[-2, :]) / dy

    # n ~ (-dz/dx, -dz/dy, 1)
    n = np.stack([-dzdx, -dzdy, np.ones_like(z, dtype=np.float32)], axis=-1)
    n = _normalize(n, axis=-1)
    return n


def _lambert_image_bgr(
    normals: np.ndarray,
    dir_r: np.ndarray,
    dir_g: np.ndarray,
    dir_b: np.ndarray,
    base_gray: float,
    gains_rgb: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Lambert shading for three colored lights.
    Returns uint8 BGR image (OpenCV convention).
    """
    n = normals  # (H,W,3), assumed unit
    l_r = _normalize(dir_r.reshape(1, 1, 3).astype(np.float32))
    l_g = _normalize(dir_g.reshape(1, 1, 3).astype(np.float32))
    l_b = _normalize(dir_b.reshape(1, 1, 3).astype(np.float32))

    if gains_rgb is None:
        # Default gains so that flat n0=[0,0,1] leads to base_gray per channel
        n0 = np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 3)
        eps = 1e-6
        g_r = float(base_gray) / float(max(eps, float((l_r * n0).sum())))
        g_g = float(base_gray) / float(max(eps, float((l_g * n0).sum())))
        g_b = float(base_gray) / float(max(eps, float((l_b * n0).sum())))
    else:
        g_r, g_g, g_b = gains_rgb

    # I_c = g_c * max(0, l_c · n)
    Ir = g_r * np.maximum(0.0, (n * l_r).sum(axis=-1))
    Ig = g_g * np.maximum(0.0, (n * l_g).sum(axis=-1))
    Ib = g_b * np.maximum(0.0, (n * l_b).sum(axis=-1))

    img = np.stack([Ib, Ig, Ir], axis=-1)  # BGR
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _regular_marker_grid(Lx_mm: float, Ly_mm: float, spacing_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regular grid fully inside the view box, coordinates in [0, L].
    """
    Nx = max(2, int(np.floor(Lx_mm / spacing_mm)))
    Ny = max(2, int(np.floor(Ly_mm / spacing_mm)))
    # center markers inside cells
    xs = np.linspace(spacing_mm * 0.5, Lx_mm - spacing_mm * 0.5, Nx, dtype=np.float32)
    ys = np.linspace(spacing_mm * 0.5, Ly_mm - spacing_mm * 0.5, Ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    return Xg, Yg


def _project_mm_to_px(cam: PinholeCamera, X_mm: np.ndarray, Y_mm: np.ndarray, Z_mm: np.ndarray) -> np.ndarray:
    pts = np.stack([X_mm.ravel(), Y_mm.ravel(), Z_mm.ravel()], axis=-1)
    uv, _ = cam.project_mm(pts)
    return uv.reshape(*X_mm.shape, 2).astype(np.float32)


def _bilinear_flow_from_marker_grid(
    flow_gx: np.ndarray,  # (Ny, Nx) flow u (px)
    flow_gy: np.ndarray,  # (Ny, Nx) flow v (px)
    Xg: np.ndarray,       # (Ny, Nx) mm
    Yg: np.ndarray,       # (Ny, Nx) mm
    X_mm: np.ndarray,     # (H, W)   mm
    Y_mm: np.ndarray,     # (H, W)   mm
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation of flow defined on a regular marker grid onto the image grid.
    Returns:
        flow_uv (H,W,2) float32
        mask (H,W) uint8 in {0,255} valid region (inside the grid cells)
    """
    Ny, Nx = Xg.shape
    H, W = X_mm.shape
    assert Ny >= 2 and Nx >= 2, "Marker grid must be at least 2x2 for bilinear interpolation."

    # Grid spacing (assume uniform)
    dx = Xg[0, 1] - Xg[0, 0] if Nx > 1 else 1.0
    dy = Yg[1, 0] - Yg[0, 0] if Ny > 1 else 1.0
    x0 = Xg[0, 0]
    y0 = Yg[0, 0]

    # Fractional indices in grid space
    sx = (X_mm - x0) / dx
    sy = (Y_mm - y0) / dy
    i0 = np.floor(sx).astype(np.int32)
    j0 = np.floor(sy).astype(np.int32)
    i1 = i0 + 1
    j1 = j0 + 1

    valid = (i0 >= 0) & (i1 < Nx) & (j0 >= 0) & (j1 < Ny)
    # Clamp for safe indexing
    i0c = np.clip(i0, 0, Nx - 2)
    j0c = np.clip(j0, 0, Ny - 2)
    i1c = i0c + 1
    j1c = j0c + 1

    wx = np.clip(sx - i0c.astype(np.float32), 0.0, 1.0)
    wy = np.clip(sy - j0c.astype(np.float32), 0.0, 1.0)

    # Gather the 4 corners
    f00x = flow_gx[j0c, i0c]; f10x = flow_gx[j0c, i1c]
    f01x = flow_gx[j1c, i0c]; f11x = flow_gx[j1c, i1c]
    f00y = flow_gy[j0c, i0c]; f10y = flow_gy[j0c, i1c]
    f01y = flow_gy[j1c, i0c]; f11y = flow_gy[j1c, i1c]

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    fx = w00 * f00x + w10 * f10x + w01 * f01x + w11 * f11x
    fy = w00 * f00y + w10 * f10y + w01 * f01y + w11 * f11y

    flow_uv = np.stack([fx, fy], axis=-1).astype(np.float32)
    mask = np.where(valid, 255, 0).astype(np.uint8)
    return flow_uv, mask


# ----------------------------- renderer class --------------------------------
@register_sensor("gelsight_style")
class GelSightStyleRenderer(SensorRenderer):
    """
    GelSight-style renderer (plugin).
    """

    def name(self) -> str:
        return "gelsight_style"

    def version(self) -> str:
        return "0.1.0"

    def modalities(self) -> Dict[str, str]:
        return {
            "image_bgr": "HxWx3 uint8 (OpenCV BGR)",
            "markers_rest_px": "Mx2 float32",
            "markers_def_px": "Mx2 float32",
            "markers_rest_xyz": "Mx3 float32 (mm)",
            "markers_def_xyz": "Mx3 float32 (mm)",
            # Optional:
            "flow_from_markers_dense": "HxWx2 float32 (px)",
            "flow_from_markers_mask": "HxW uint8 {0,255}",
            "normal_map": "HxWx3 float32 (unit normals)",
            "depth_map": "HxW float32 (mm)",
        }

    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        cfg = dict(self.cfg or {})

        # --- Camera & view ---
        cam: Optional[Any] = getattr(scene, "camera", None)
        view_wh = cfg.get("view_mm", [40.0, 30.0])
        Lx_mm, Ly_mm = float(view_wh[0]), float(view_wh[1])

        if cam is None:
            cam_cfg = cfg.get("camera", {})
            if not cam_cfg:
                raise ValueError("Missing 'camera' config and Scene.camera is None.")
            cam = _build_camera_from_cfg(cam_cfg, (Lx_mm, Ly_mm))

        img_w, img_h = int(cam_cfg["img_wh"][0]), int(cam_cfg["img_wh"][1])

        # Membrane/surface z (mm) where the reflective layer and markers live
        surf_cfg = cfg.get("surface", {})
        z_surf_mm = float(surf_cfg.get("z_mm", 1.0))

        # --- Deformation field ---
        def_cfg = cfg.get("deformation", {"mode": "none"})
        field = _build_deform_field_from_cfg(def_cfg)

        # --- Render options & lighting ---
        render_cfg = cfg.get("render", {})
        ds = int(render_cfg.get("grid_downsample", 1))
        save_n = bool(render_cfg.get("save_normal_map", False))
        save_z = bool(render_cfg.get("save_depth_map", False))
        save_flow = bool(render_cfg.get("save_flow_from_markers", True))

        base_gray = float(cfg.get("background_gray", 20.0))

        L = cfg.get("lighting", {})
        dir_r = np.asarray(L.get("dir_r", [0.577, 0.0, 0.816]), dtype=np.float32)
        dir_g = np.asarray(L.get("dir_g", [-0.289, 0.5, 0.816]), dtype=np.float32)
        dir_b = np.asarray(L.get("dir_b", [-0.289, -0.5, 0.816]), dtype=np.float32)
        gains = L.get("gains_rgb", None)
        gains_rgb = tuple(float(x) for x in gains) if gains is not None else None

        # --- Build per-pixel mm grid (possibly downsampled for speed) ---
        Hs = img_h // ds
        Ws = img_w // ds
        Xs_mm, Ys_mm, dx_mm_s, dy_mm_s = _mm_grid_for_image(Hs, Ws, Lx_mm, Ly_mm)
        Zs_mm = np.full((Hs * Ws,), z_surf_mm, dtype=np.float32)
        

        # sample deformation at Z = z_surf_mm, get absolute height Z' = z_surf_mm + u_z
        pts0 = np.stack([Xs_mm.ravel(), Ys_mm.ravel(), Zs_mm.ravel()], axis=-1)
        print(f"field.mode={field.mode}")
        if field.mode != "none":
            disp = field.sample(pts0).astype(np.float32)  # (M,3)
            uz = disp[:, 2].reshape(Hs, Ws)
        else:
            uz = np.zeros((Hs, Ws), dtype=np.float32)

        z_abs = z_surf_mm + uz  # absolute height field at the membrane

        # normals from height map (downsampled grid)
        n_small = _finite_diff_normals(z_abs, dx_mm_s, dy_mm_s)

        # upsample normals to full res if needed, then renormalize
        if ds > 1:
            n_full = np.empty((img_h, img_w, 3), dtype=np.float32)
            for c in range(3):
                n_full[..., c] = cv2.resize(n_small[..., c], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            n_full = _normalize(n_full, axis=-1)
            z_full = cv2.resize(z_abs, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        else:
            n_full = n_small
            z_full = z_abs

        # Lambert shading (BGR)
        img_bgr = _lambert_image_bgr(n_full, dir_r, dir_g, dir_b, base_gray, gains_rgb=gains_rgb)

        # --- Markers (regular grid in membrane coords) ---
        M = cfg.get("markers", {})
        spacing_mm = float(M.get("spacing_mm", 1.0))
        radius_px = int(M.get("radius_px", 2))
        color_bgr = tuple(int(c) for c in M.get("color_bgr", [0, 0, 0]))
        jitter_px = int(M.get("jitter_px", 0))  # kept but unused by default (we keep it deterministic)

        Xg, Yg = _regular_marker_grid(Lx_mm, Ly_mm, spacing_mm)  # (Ny,Nx)
        Ny, Nx = Xg.shape
        Zg = np.full_like(Xg, z_surf_mm, dtype=np.float32)  # <-- use surface z value

        # rest → def in 3D (mm)
        rest_xyz = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=-1).astype(np.float32)
        if field.mode != "none":
            d_mark = field.sample(rest_xyz).astype(np.float32)
        else:
            d_mark = np.zeros_like(rest_xyz, dtype=np.float32)
        def_xyz = rest_xyz + d_mark

        # project markers to pixels
        rest_px, _ = cam.project_mm(rest_xyz)   # (M,2), (M,)
        def_px,  _ = cam.project_mm(def_xyz)

        # overlay markers (draw at deformed locations)
        img_bgr_draw = img_bgr.copy()
        for uv in def_px:
            u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= u < img_w and 0 <= v < img_h:
                cv2.circle(img_bgr_draw, (u, v), radius_px, color_bgr, thickness=-1, lineType=cv2.LINE_AA)

        # --- Dense flow from marker grid (optional) ---
        modalities: Dict[str, Any] = {
            "image_bgr": img_bgr_draw
        }

        # Save marker arrays as modalities for downstream algorithms
        modalities["markers_rest_px"] = rest_px.copy()
        modalities["markers_def_px"] = def_px.copy()
        modalities["markers_rest_xyz"] = rest_xyz.copy()
        modalities["markers_def_xyz"] = def_xyz.copy()

        if save_flow and Nx >= 2 and Ny >= 2:
            # Flow on the marker grid in pixels (def - rest)
            rest_px_grid = rest_px.reshape(Ny, Nx, 2)
            def_px_grid = def_px.reshape(Ny, Nx, 2)
            flow_g = (def_px_grid - rest_px_grid).astype(np.float32)
            flow_gx = flow_g[..., 0]
            flow_gy = flow_g[..., 1]

            # Interpolate to full image
            X_mm, Y_mm, _, _ = _mm_grid_for_image(img_h, img_w, Lx_mm, Ly_mm)
            flow_uv, mask = _bilinear_flow_from_marker_grid(flow_gx, flow_gy, Xg, Yg, X_mm, Y_mm)
            modalities["flow_from_markers_dense"] = flow_uv.astype(np.float32)
            modalities["flow_from_markers_mask"] = mask

        # Optional GT convenience
        if save_n:
            modalities["normal_map"] = n_full.astype(np.float32)
        if save_z:
            modalities["depth_map"] = z_full.astype(np.float32)

        meta = {
            "renderer": self.name(),
            "version": self.version(),
            "image_size": (img_w, img_h),
            "marker_grid_shape": (Ny, Nx),
            "marker_spacing_mm": spacing_mm,
        }

        return FrameBundle(modalities=modalities, metadata=meta, aux={})
