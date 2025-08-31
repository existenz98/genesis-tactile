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
Particle generation utilities:
- Layer config -> random particles (x,y,z,r,color)
- Optional random convex polygon shape
"""


from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class LayerSpec:
    name: str
    z_min_mm: float
    z_max_mm: float
    color_bgr: tuple[int, int, int]  # particle color, BGR
    density_per_mm2: float           # expected particles per mm^2
    radius_mm: float                 # single size per layer
    depth_atten: str = "exp"         # "none" | "exp" | "linear"
    depth_beta: float = 0.2          # attenuation strength (for exp/linear)
    shape: str = "disk"              # "disk" | "poly",  circular or polygon
    poly_verts: int = 0              # if >0, convex polygon number of vertices


@dataclass
class Particles:
    xyz_mm: np.ndarray      # (N,3)
    radius_mm: np.ndarray   # (N,)
    color_bgr: np.ndarray   # (N,3) uint8
    layer_id: np.ndarray    # (N,) int
    is_polygon: np.ndarray  # (N,) bool
    poly_verts: np.ndarray  # (N,) int (0 means disk)


def _random_convex_polygon(center_px: tuple[float, float],
                           radius_px: float,
                           n_verts: int,
                           jitter: float = 0.35) -> np.ndarray:
    """
    Generate a random convex polygon (2D) around a center with approx radius.
    Returns int32 array of shape (n_verts, 1, 2), suitable for drawing using cv2.fillPoly.
    """
    angles = np.linspace(0, 2*np.pi, n_verts, endpoint=False)
    angles += np.random.uniform(0, 2*np.pi/n_verts)  # random rotation
    r = radius_px * (1.0 + jitter * (2*np.random.rand(n_verts) - 1.0))
    pts = np.stack([np.cos(angles) * r, np.sin(angles) * r], axis=1)
    pts[:, 0] += center_px[0]
    pts[:, 1] += center_px[1]
    return pts.astype(np.int32).reshape(-1, 1, 2)


def generate_particles(layers: list[LayerSpec],
                       Lx_mm: float, Ly_mm: float,
                       rng: np.random.Generator) -> Particles:
    """
    Sample particles independently per layer
    uniform within [0,Lx]x[0,Ly]x[zmin,zmax].
    """
    xyz_list, r_list, col_list, lid_list, isp_list, pv_list = [], [], [], [], [], []
    area = Lx_mm * Ly_mm
    for lid, L in enumerate(layers):
        expN = max(int(round(L.density_per_mm2 * area)), 0)
        if expN <= 0:
            continue
        x = rng.uniform(0.0, Lx_mm, size=expN)
        y = rng.uniform(0.0, Ly_mm, size=expN)
        z = rng.uniform(L.z_min_mm, L.z_max_mm, size=expN)
        xyz = np.stack([x, y, z], axis=1)
        xyz_list.append(xyz)
        r_list.append(np.full(expN, L.radius_mm, dtype=float))
        col_list.append(np.tile(np.array(L.color_bgr, dtype=np.uint8), (expN, 1)))
        lid_list.append(np.full(expN, lid, dtype=np.int32))
        is_poly = (L.shape == "poly" and L.poly_verts and L.poly_verts > 2)
        isp_list.append(np.full(expN, is_poly, dtype=bool))
        pv_list.append(np.full(expN, int(L.poly_verts or 0), dtype=np.int32))

    if not xyz_list:
        return Particles(
            xyz_mm=np.zeros((0,3), float),
            radius_mm=np.zeros((0,), float),
            color_bgr=np.zeros((0,3), np.uint8),
            layer_id=np.zeros((0,), np.int32),
            is_polygon=np.zeros((0,), bool),
            poly_verts=np.zeros((0,), np.int32),
        )

    xyz_mm = np.concatenate(xyz_list, axis=0)
    radius_mm = np.concatenate(r_list, axis=0)
    color_bgr = np.concatenate(col_list, axis=0)
    layer_id = np.concatenate(lid_list, axis=0)
    is_polygon = np.concatenate(isp_list, axis=0)
    poly_verts = np.concatenate(pv_list, axis=0)

    return Particles(xyz_mm, radius_mm, color_bgr, layer_id, is_polygon, poly_verts)


def depth_brightness(weights: np.ndarray, z_mm: np.ndarray, mode: str, beta: float) -> np.ndarray:
    """
    Apply per-particle brightness attenuation based on depth (mm).
    """

    if mode == "none":
        return weights
    if mode == "exp":
        return weights * np.exp(-beta * z_mm)
    if mode == "linear":
        return weights * np.clip(1.0 - beta * z_mm, 0.0, 1.0)
    return weights

