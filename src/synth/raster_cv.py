"""
Rasterizer using OpenCV:
- Depth-sort particles (far → near) and draw disks or random convex polygons.
- Supersampling AA: render at scale S (e.g., 2x), then downsample with INTER_AREA.
- Simple per-layer depth brightness attenuation (no alpha blending; near overwrites far).

Currently is fully CPU based.
"""

from __future__ import annotations
import numpy as np
import cv2

from .camera import PinholeCamera
from .particles import Particles, LayerSpec, depth_brightness, _random_convex_polygon


def render_frame_cv(cam: PinholeCamera,
                    parts: Particles,
                    layers: list[LayerSpec],
                    Lx_mm: float, Ly_mm: float,
                    supersample: int = 2,
                    bg_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Render Camera view
    RGB frame (OpenCV BGR), of size (H,W,3) uint8.
    """

    ss = max(int(supersample), 1)
    W, H = cam.img_w, cam.img_h
    Wss, Hss = W * ss, H * ss

    # Project particles to pixel coords and compute pixel radii
    uv, Z = cam.project_mm(parts.xyz_mm)
    r_px = cam.radius_mm_to_px(parts.radius_mm, Z) * ss

    # Discard anything completely off-screen (with a small margin)
    margin = 2.0 * ss
    mask = (
        (uv[:, 0] >= -margin) & (uv[:, 0] < W + margin) &
        (uv[:, 1] >= -margin) & (uv[:, 1] < H + margin) &
        (r_px > 0.2)
    )
    if not np.any(mask):
        img = np.full((H, W, 3), bg_color, dtype=np.uint8)
        return img

    uv = uv[mask] * ss
    Z = Z[mask]
    r_px = r_px[mask]
    color = parts.color_bgr[mask]
    layer_id = parts.layer_id[mask]
    is_poly = parts.is_polygon[mask]
    poly_verts = parts.poly_verts[mask]
    xyz = parts.xyz_mm[mask]

    # Depth brightness per layer
    brightness = np.ones(uv.shape[0], dtype=float)
    for lid, L in enumerate(layers):
        sel = (layer_id == lid)
        if not np.any(sel):
            continue
        brightness[sel] = depth_brightness(brightness[sel], xyz[sel, 2], L.depth_atten, L.depth_beta)

    # Depth sort: far → near (so nearer overwrites farther)
    order = np.argsort(Z)[::-1]
    uv = uv[order]
    r_px = r_px[order]
    color = color[order]
    is_poly = is_poly[order]
    poly_verts = poly_verts[order]
    layer_id = layer_id[order]
    brightness = brightness[order]

    # Supersampled canvas
    canvas = np.full((Hss, Wss, 3), bg_color, dtype=np.uint8)

    # Draw
    for i in range(uv.shape[0]):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        rad = float(r_px[i])
        b = float(brightness[i])
        col = np.clip((color[i].astype(np.float32) * b), 0, 255).astype(np.uint8)

        if is_poly[i] and poly_verts[i] >= 3:
            poly = _random_convex_polygon((u, v), rad, int(poly_verts[i]))
            cv2.fillPoly(canvas, [poly], color=tuple(int(c) for c in col))
        else:
            cv2.circle(canvas, (int(round(u)), int(round(v))), int(max(1, round(rad))),
                       color=tuple(int(c) for c in col), thickness=-1, lineType=cv2.LINE_AA)

    if ss > 1:
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return canvas
