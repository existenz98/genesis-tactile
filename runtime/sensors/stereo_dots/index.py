# -----------------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later WITH LicenseRef-YF-Device-Interface-Exception
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# This file is part of the Runtime of the tactile vision platform.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE-RUNTIME-AGPL for details.
#
# Special Exception (Device Interface Exception):
#   Proprietary or separately-licensed device drivers or hardware interface
#   modules that communicate with the Runtime solely through the documented
#   TSI/plugin/IPC interfaces are not considered derivative works of the
#   Runtime by this project, and thus are not subject to the copyleft
#   obligations of the AGPL, provided they do not include or modify Runtime code.
#   See LICENSE-EXCEPTIONS for the full text.
#
# Patent Notice:
#   Except for any rights granted under the applicable open-source license,
#   no patent license is granted or implied. Users are responsible for ensuring
#   their use does not infringe third-party patents (e.g., tactile sensor
#   hardware or methods).
#
# Citation:
#   If you use this software in academic work, please cite the associated
#   publications when available.
# -----------------------------------------------------------------------------


"""
Indexing utilities for grid of dots

"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, Tuple, Optional


#----- Tracking based -----

def match_to_predictions(pts: np.ndarray, preds: np.ndarray, gate_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-neighbor with gating.
    Strategy:
    - during prepare() use geometry to predict pixel positions for each (r,c),
    - then match detections to nearest predictions with gating. For subsequent frames,
    - track by proximity to previous positions.

    pts: (N,2), preds: (M,2) where M = R*C and rows are in fixed order of (r,c).

    Returns:
      - idx_pred_for_pt: (N,) index to preds or -1 if no match
      - idx_pt_for_pred: (M,) index to pts or -1 if no match
    """
    N = pts.shape[0]; M = preds.shape[0]
    if N == 0 or M == 0:
        print("[match_to_predictions] Warning: no points or no predictions")
        return -np.ones(N, dtype=int), -np.ones(M, dtype=int)

    # compute pairwise distances
    D = np.linalg.norm(pts[:,None,:] - preds[None,:,:], axis=2)  # (N,M)
    idx_pred_for_pt = -np.ones(N, dtype=int)
    idx_pt_for_pred = -np.ones(M, dtype=int)

    # greedy: repeatedly pick smallest distance pair under gate
    used_pts = np.zeros(N, dtype=bool)
    used_pred = np.zeros(M, dtype=bool)

    # Flatten sorted indices by distance
    order = np.argsort(D, axis=None)
    gate2 = gate_px
    for k in order:
        i = k // M; j = k % M
        if used_pts[i] or used_pred[j]: 
            continue
        if D[i,j] <= gate2:
            idx_pred_for_pt[i] = j
            idx_pt_for_pred[j] = i
            used_pts[i] = True; used_pred[j] = True
    return idx_pred_for_pt, idx_pt_for_pred


#----- Robust single-frame indexing based on directional cones -----


def index_grid_from_points(
    pts, cols, rows,
    cone_half_angle_deg=40.0,
    par_window=(0.5, 1.8),
    perp_max=0.35,
    lambda_perp=0.7,
    anchor="top-left",   # choose one of {"top-left","top-right","bottom-left","bottom-right"}
):
    """
    Parameters
    ----------
    pts : (N,2) array-like
        Detected point (u,v) pixel coordinates. OpenCV convention (u=x right, v=y down).
    cols, rows : int
        Expected grid dimensions. Some cells may be missing.
    cone_half_angle_deg : float
        Half-angle of directional cones around +x, -x, +y, -y in degrees (in normalized space).
    par_window : (float, float)
        Allowed parallel step range in normalized units.
    perp_max : float
        Allowed perpendicular deviation in normalized units.
    lambda_perp : float
        Weight for perpendicular deviation in neighbor selection cost.
    anchor : str
        Anchoring of indices to a particular image corner:
        "top-left" (default) | "top-right" | "bottom-left" | "bottom-right".

    Returns
    -------
    grid : (rows, cols, 2) float64
        u,v pixel coords per (row, col). Missing cells are np.nan.
    mask : (rows, cols) bool
        True where a detection was assigned to (row, col).
    """
    if pts is None or len(pts) == 0:
        grid = np.full((rows, cols, 2), np.nan, dtype=float)
        mask = np.zeros((rows, cols), dtype=bool)
        return grid, mask

    pts = np.asarray(pts, dtype=float)
    x = pts[:, 0]  # u
    y = pts[:, 1]  # v
    N = pts.shape[0]

    # -----------------------------
    # 0) Robust global step sizes s_x, s_y (normalization)
    # -----------------------------
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    np.fill_diagonal(dx, np.nan)
    np.fill_diagonal(dy, np.nan)

    tan0 = np.tan(np.deg2rad(cone_half_angle_deg))

    east_mask = (dx > 0) & (np.abs(dy) <= tan0 * dx)
    west_mask = (dx < 0) & (np.abs(dy) <= tan0 * (-dx))
    south_mask = (dy > 0) & (np.abs(dx) <= tan0 * dy)
    north_mask = (dy < 0) & (np.abs(dx) <= tan0 * (-dy))

    with np.errstate(all='ignore'):
        east_min = np.nanmin(np.where(east_mask, dx, np.nan), axis=1)
        west_min = np.nanmin(np.where(west_mask, -dx, np.nan), axis=1)
        south_min = np.nanmin(np.where(south_mask, dy, np.nan), axis=1)
        north_min = np.nanmin(np.where(north_mask, -dy, np.nan), axis=1)

    sx_candidates = np.concatenate([east_min[~np.isnan(east_min)],
                                    west_min[~np.isnan(west_min)]], axis=0)
    sy_candidates = np.concatenate([south_min[~np.isnan(south_min)],
                                    north_min[~np.isnan(north_min)]], axis=0)

    if sx_candidates.size == 0 or sy_candidates.size == 0:
        d2 = dx**2 + dy**2
        d2[np.isnan(d2)] = np.inf
        nn = np.min(d2, axis=1) ** 0.5
        m = np.median(nn[nn < np.inf]) if np.any(nn < np.inf) else 1.0
        sx = np.median(sx_candidates) if sx_candidates.size else m
        sy = np.median(sy_candidates) if sy_candidates.size else m
    else:
        sx = np.median(sx_candidates)
        sy = np.median(sy_candidates)

    x_n = x / (sx if sx > 0 else 1.0)
    y_n = y / (sy if sy > 0 else 1.0)

    dxn = x_n[None, :] - x_n[:, None]
    dyn = y_n[None, :] - y_n[:, None]
    np.fill_diagonal(dxn, np.nan)
    np.fill_diagonal(dyn, np.nan)

    tan = tan0
    par_lo, par_hi = par_window
    lam = float(lambda_perp)

    # -----------------------------
    # 1) Pick at most one neighbor in each cone
    # -----------------------------
    def pick_neighbor(direction):
        if direction == 'E':
            par = dxn; perp = dyn
            keep = (par > 0) & (np.abs(perp) <= tan * par)
            par_eff = par
        elif direction == 'W':
            par = -dxn; perp = dyn
            keep = (par > 0) & (np.abs(perp) <= tan * par)
            par_eff = par
        elif direction == 'S':
            par = dyn; perp = dxn
            keep = (par > 0) & (np.abs(perp) <= tan * par)
            par_eff = par
        elif direction == 'N':
            par = -dyn; perp = dxn
            keep = (par > 0) & (np.abs(perp) <= tan * par)
            par_eff = par
        else:
            raise ValueError("direction must be one of 'E','W','N','S'")

        keep &= (par_eff >= par_lo) & (par_eff <= par_hi) & (np.abs(perp) <= perp_max)
        dist = np.sqrt(dxn**2 + dyn**2)
        cost = np.where(keep, np.abs(par_eff - 1.0) + lam * np.abs(perp) + 1e-3 * dist, np.inf)

        j_idx = np.argmin(cost, axis=1)
        j_cost = cost[np.arange(N), j_idx]
        j_idx[~np.isfinite(j_cost)] = -1
        return j_idx

    right_cand = pick_neighbor('E')
    left_cand  = pick_neighbor('W')
    down_cand  = pick_neighbor('S')
    up_cand    = pick_neighbor('N')

    # -----------------------------
    # 2) Mutual confirmation
    # -----------------------------
    right = np.full(N, -1, dtype=int)
    left  = np.full(N, -1, dtype=int)
    down  = np.full(N, -1, dtype=int)
    up    = np.full(N, -1, dtype=int)

    for i, j in enumerate(right_cand):
        if j >= 0 and left_cand[j] == i:
            right[i] = j
            left[j] = i
    for i, j in enumerate(down_cand):
        if j >= 0 and up_cand[j] == i:
            down[i] = j
            up[j] = i

    # degree is the number of confirmed neighbors per node
    degree = (right >= 0).astype(int) + (left >= 0).astype(int) + \
             (up >= 0).astype(int) + (down >= 0).astype(int)

    # -----------------------------
    # 3) Seed selection (central, well-connected)
    # -----------------------------
    centroid = np.array([x_n.mean(), y_n.mean()])
    d2c = (x_n - centroid[0])**2 + (y_n - centroid[1])**2
    order = np.lexsort((d2c, -degree))  # sorting by primary key: -degree, secondary key: distance to centroid
    seed = order[0]

    # -----------------------------
    # 4) BFS indexing
    # -----------------------------
    col_idx = np.full(N, np.nan)
    row_idx = np.full(N, np.nan)
    col_idx[seed] = 0.0
    row_idx[seed] = 0.0

    from collections import deque
    q = deque([seed])

    while q:
        i = q.popleft()
        ci = int(col_idx[i]); ri = int(row_idx[i])
        for j, dc, dr in ((right[i], +1, 0),
                          (left[i],  -1, 0),
                          (down[i],  0, +1),
                          (up[i],    0, -1)):
            if j < 0: 
                continue
            if np.isnan(col_idx[j]) or np.isnan(row_idx[j]):
                col_idx[j] = ci + dc
                row_idx[j] = ri + dr
                q.append(j)
            else:
                # keep first assignment (conservative)
                pass

    labeled = ~np.isnan(col_idx) & ~np.isnan(row_idx)
    if not np.any(labeled):
        grid = np.full((rows, cols, 2), np.nan, dtype=float)
        mask = np.zeros((rows, cols), dtype=bool)
        return grid, mask

    col_idx = col_idx[labeled].astype(int)
    row_idx = row_idx[labeled].astype(int)
    pts_lab = pts[labeled]
    degree_lab = degree[labeled]

    # -----------------------------
    # 5) Anchoring (small change)
    # -----------------------------
    cmin, cmax = int(col_idx.min()), int(col_idx.max())
    rmin, rmax = int(row_idx.min()), int(row_idx.max())

    if anchor == "top-left":
        col_idx = col_idx - cmin
        row_idx = row_idx - rmin
    elif anchor == "top-right":
        # Rightmost visible -> cols-1; topmost visible -> 0
        col_idx = col_idx + (cols - 1 - cmax)
        row_idx = row_idx - rmin
    elif anchor == "bottom-left":
        col_idx = col_idx - cmin
        row_idx = row_idx + (rows - 1 - rmax)
    elif anchor == "bottom-right":
        col_idx = col_idx + (cols - 1 - cmax)
        row_idx = row_idx + (rows - 1 - rmax)
    else:
        raise ValueError("anchor must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")

    # -----------------------------
    # 6) Write out grid & mask (resolve duplicates by degree)
    # -----------------------------
    grid = np.full((rows, cols, 2), np.nan, dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    conf = np.full((rows, cols), -1, dtype=int)

    for (c, r, p, deg) in zip(col_idx, row_idx, pts_lab, degree_lab):
        if 0 <= r < rows and 0 <= c < cols:
            if not mask[r, c] or deg > conf[r, c]:
                grid[r, c, :] = p
                mask[r, c] = True
                conf[r, c] = deg

    return grid, mask

