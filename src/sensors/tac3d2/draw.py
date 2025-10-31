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
Drawing utilities for Tac3D dots/ellipses

using OpenCV.
"""


from __future__ import annotations
import numpy as np
import cv2


def draw_ellipses(img: np.ndarray, centers_uv: np.ndarray, axes_ab: np.ndarray,
                  angles_deg: np.ndarray, color_bgr=(255, 255, 255),
                  in_fov_mask: np.ndarray | None = None, thickness: int = -1):
    """
    Draw N ellipses (filled) at given centers.
    - centers_uv: (N,2) float (u,v) pixels
    - axes_ab: (N,2) float semi-axes (a,b) in pixels
    - angles_deg: (N,) float orientation angles in degrees
    - in_fov_mask: skip dots of out of FOV
    - thickness: -1 means filled
    """

    H, W = img.shape[:2]
    N = centers_uv.shape[0]
    if in_fov_mask is None:
        in_fov = np.ones((N,), dtype=bool)
    else:
        in_fov = in_fov_mask.astype(bool)

    for i in range(N):
        if not in_fov[i]:
            continue
        u, v = float(centers_uv[i, 0]), float(centers_uv[i, 1])
        a, b = float(axes_ab[i, 0]), float(axes_ab[i, 1])
        ang = float(angles_deg[i])

        # Skip invalid and too small ones
        if not (np.isfinite(u) and np.isfinite(v) and a > 0.1 and b > 0.1):
            continue

        center = (int(round(u)), int(round(v)))
        axes = (int(round(a)), int(round(b)))  # cv2 expects integer semi-axes
        # cv2.ellipse uses angle in degrees (rotation from x-axis), start/end are 0..360
        cv2.ellipse(img, center, axes, ang, 0.0, 360.0, color_bgr, thickness=thickness)


def draw_circles(img: np.ndarray, centers_uv: np.ndarray, radii_px: np.ndarray,
                 color_bgr=(255, 255, 255),
                 in_fov_mask: np.ndarray | None = None, thickness: int = -1):
    """
    Draw filled circles.
    radii_px: (N,) float radius in pixels
    """
    H, W = img.shape[:2]
    N = centers_uv.shape[0]
    if in_fov_mask is None:
        in_fov = np.ones((N,), dtype=bool)
    else:
        in_fov = in_fov_mask.astype(bool)

    for i in range(N):
        if not in_fov[i]:
            continue
        u, v = float(centers_uv[i, 0]), float(centers_uv[i, 1])
        r = float(radii_px[i])
        if not (np.isfinite(u) and np.isfinite(v) and r > 0.1):
            continue
        center = (int(round(u)), int(round(v)))
        cv2.circle(img, center, int(round(r)), color_bgr, thickness=thickness)

