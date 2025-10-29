
"""
Grid utilities for Tac3D (panel-space lattice)
"""


from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class GridSpec:
    rows: int
    cols: int
    spacing_mm: float
    origin_center: bool = True  # origin at panel center by default


def make_grid_uv(spec: GridSpec):
    """
    Build grid of (u,v) coordinates in panel space.

    Return (uv_mm, idx_rc)
    - uv_mm: (R*C, 2) array of (u, v) in mm, 
        grid is centered at (0,0) if origin_center, otherwise grid starts from (0,0)
    - idx_rc: (R*C, 2) int indices (r, c) for each entry in uv_mm
    """

    R, C, s = spec.rows, spec.cols, spec.spacing_mm

    # Coordinate values along each axis in mm
    if spec.origin_center:
        u_vals = (np.arange(C) - (C - 1) / 2.0) * s
        v_vals = (np.arange(R) - (R - 1) / 2.0) * s
    else:
        u_vals = np.arange(C) * s
        v_vals = np.arange(R) * s

    # Meshgrid with rows along v (down) and cols along u (right)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(float)

    # r,c indices in the same flattening order (row-major)
    rc = np.stack(np.meshgrid(np.arange(R), np.arange(C), indexing='ij'), axis=-1).reshape(-1, 2)

    return uv, rc
