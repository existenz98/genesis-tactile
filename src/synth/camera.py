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
Simple pinhole camera
From physical position millimetres to pixels in the image.
Assume the camera center is at z = -z_cam_mm, looking along +z.
Points in the gel have z in [0, Lz_mm], so their camera depth is Z = z + z_cam_mm.

Calibrate (fx, fy) so that result is an 'edge-to-edge' image: a rectangle of size (Lx_mm, Ly_mm) placed at
depth Z_ref_mm maps exactly to (img_w, img_h) pixels.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PinholeCamera:
    img_w: int
    img_h: int
    fx: float
    fy: float
    cx: float
    cy: float
    z_cam_mm: float   # camera center z; must > 0 (camera is at -z_cam_mm, looking up at +z direction)
    view_Lx_mm: float
    view_Ly_mm: float

    @staticmethod
    def from_viewbox(img_w: int, img_h: int,
                     Lx_mm: float, Ly_mm: float,
                     Z_ref_mm: float,
                     z_cam_mm: float) -> "PinholeCamera":
        """
        Choose fx, fy so that a plane at depth Z_ref_mm covers the full image.
        For points on that plane: u = fx * X / Z_ref + cx maps X ∈ [0, Lx] to u ∈ [0, img_w].
        """
        assert Z_ref_mm > 0, "Z_ref_mm must be positive (distance from camera)"
        fx = img_w * Z_ref_mm / Lx_mm
        fy = img_h * Z_ref_mm / Ly_mm
        cx = img_w * 0.5
        cy = img_h * 0.5
        return PinholeCamera(img_w, img_h, fx, fy, cx, cy, z_cam_mm, Lx_mm, Ly_mm)

    def project_mm(self, xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points (N,3) in mm position to pixel coords (N,2) and depths Z (N,).
        Returns (uv_px, Z_mm). No clipping/culling here.
        """
        X = xyz_mm[:, 0]
        Y = xyz_mm[:, 1]
        Z = xyz_mm[:, 2] + self.z_cam_mm  # camera at z = -z_cam_mm
        # Avoid divide-by-zero
        Z = np.maximum(Z, 1e-9)
        #u = self.fx * (X / Z) + self.cx
        #v = self.fy * (Y / Z) + self.cy
        Xc = X - 0.5 * self.view_Lx_mm
        Yc = Y - 0.5 * self.view_Ly_mm
        u  = self.fx * (Xc / Z) + self.cx
        v  = self.fy * (Yc / Z) + self.cy
        uv = np.stack([u, v], axis=1)
        return uv, Z

    def radius_mm_to_px(self, r_mm: np.ndarray, Z_mm: np.ndarray) -> np.ndarray:
        """Convert a physical radius (mm) to pixel radius at depth Z using fx (assumes roughly square pixels)."""
        Z = np.maximum(Z_mm, 1e-9)
        return (self.fx * (r_mm / Z)).astype(float)
