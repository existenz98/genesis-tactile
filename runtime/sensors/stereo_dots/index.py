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
import numpy as np
from typing import Tuple, Optional

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



def _fit_anchor_column(pts: np.ndarray,
                       rows: int,
                       side: str,
                       anchor_bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an 'anchor column' as a smooth curve:
      - bin points along v (image row) direction
      - per bin choose extreme-u point: left view -> max u, right view -> min u
      - smooth by low-degree polyfit u(v)
      - resample to exactly rows anchor points at uniformly spaced v
    Returns (anchor_rc[rows,2], mask[rows]) in pixel coords.
    """
    if pts.size == 0:
        return np.full((rows, 2), np.nan, float), np.zeros((rows,), bool)

    vmin, vmax = np.min(pts[:, 1]), np.max(pts[:, 1])
    B = int(anchor_bins or rows)
    edges = np.linspace(vmin, vmax + 1e-9, B + 1)
    picks = []
    for b in range(B):
        sel = (pts[:, 1] >= edges[b]) & (pts[:, 1] < edges[b+1])
        if not np.any(sel):
            continue
        subset = pts[sel]
        k = np.argmax(subset[:, 0]) if side == "left" else np.argmin(subset[:, 0])
        picks.append(subset[k])
    if len(picks) < 2:
        return np.full((rows, 2), np.nan, float), np.zeros((rows,), bool)
    picks = np.array(sorted(picks, key=lambda p: p[1]), dtype=float)  # sort by v
    deg = int(min(3, len(picks) - 1))
    coef = np.polyfit(picks[:, 1], picks[:, 0], deg=deg)  # u(v)
    v_targets = np.linspace(picks[0, 1], picks[-1, 1], rows)
    u_smooth = np.polyval(coef, v_targets)
    anchor = np.stack([u_smooth, v_targets], axis=1)
    # valid where inside original v-range by small margin
    mask = (v_targets >= vmin) & (v_targets <= vmax)
    return anchor, mask


def _estimate_step_and_normals(anchor: np.ndarray, pts: np.ndarray, side: str) -> Tuple[np.ndarray, float]:
    """
    From the anchor curve (rows x 2), compute per-row inward normals and a
    robust inter-column step size (in pixels near the anchor).
    """
    R = anchor.shape[0]
    # tangent via central differences in (u,v) pixel space
    T = np.zeros_like(anchor)
    T[1:-1] = anchor[2:] - anchor[:-2]
    T[0] = anchor[1] - anchor[0]
    T[-1] = anchor[-1] - anchor[-2]
    # normal (rotate tangent by +90°)
    N = np.empty_like(T)
    N[:, 0] = -T[:, 1]
    N[:, 1] = T[:, 0]
    # normalize
    nrm = np.linalg.norm(N, axis=1, keepdims=True)
    nrm = np.where(nrm > 1e-9, nrm, 1.0)
    N = N / nrm
    # Ensure normals point inward: left view -> decreasing u; right view -> increasing u
    sgn = -1.0 if side == "left" else +1.0
    N[:, 0] = np.where(np.sign(N[:, 0]) == sgn, N[:, 0], -N[:, 0])
    N[:, 1] = np.where(np.sign(N[:, 0]) == sgn, N[:, 1], -N[:, 1])  # flip both when needed

    # Estimate one-step distance d by projecting all points onto normals from anchor
    proj_dist = []
    for r in range(R):
        a = anchor[r]
        n = N[r]
        d = (pts - a[None, :]) @ n  # signed along normal
        dpos = d[d > 0.0]
        if dpos.size:
            proj_dist.append(np.min(dpos))
    if len(proj_dist) == 0:
        d_pix = 8.0  # fallback
    else:
        d_pix = float(np.median(proj_dist))
    return N, d_pix

def robust_index_half(pts: np.ndarray,
                      rows: int,
                      cols: int,
                      side: str,
                      anchor_bins: Optional[int] = None,
                      gate_px: float = 12.0,
                      max_step_mult: float = 1.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust single-frame indexing (without tracking)
      1) build anchor column as a smooth extreme-u curve (rightmost for left view, leftmost for right view)
      2) propagate inward along per-row normals with estimated step size
      3) per column, pick nearest detections within a gate
    Returns (uv_grid[R,C,2], mask[R,C]).
    """
    R, C = int(rows), int(cols)
    grid = np.full((R, C, 2), np.nan, dtype=float)
    mask = np.zeros((R, C), dtype=bool)
    if pts.size == 0:
        return grid, mask

    anchor, amask = _fit_anchor_column(pts, R, side=side, anchor_bins=anchor_bins)
    if not np.any(amask):
        return grid, mask

    Nrm, d_pix = _estimate_step_and_normals(anchor, pts, side=side)
    # Column 0 = anchor (extreme side)
    c0 = 0
    grid[:, c0] = anchor
    mask[:, c0] = amask

    # Greedy per-column growth
    used = np.zeros((pts.shape[0],), dtype=bool)
    # mark used: anything snapped to anchor within gate
    for r in range(R):
        if not mask[r, c0]:
            continue
        d = np.linalg.norm(pts - grid[r, c0][None, :], axis=1)
        j = np.argmin(d)
        if d[j] <= gate_px:
            used[j] = True

    for c in range(1, C):
        for r in range(R):
            if not mask[r, c-1]:
                continue
            pred = grid[r, c-1] + Nrm[r] * d_pix
            d = np.linalg.norm(pts - pred[None, :], axis=1)
            j = np.argmin(d)
            if used[j]:
                continue
            if d[j] <= max_step_mult * gate_px:
                grid[r, c] = pts[j]
                mask[r, c] = True
                used[j] = True
            else:
                # leave missing; next columns may still succeed for other rows
                pass
    return grid, mask

def assign_grid_to_predictions(grid_uv: np.ndarray,
                               grid_mask: np.ndarray,
                               preds_flat: np.ndarray,
                               gate_px: float) -> np.ndarray:
    """
    Map robust grid (R,C,2) to predicted index order (N,2) by nearest neighbor with gating.
    Returns (N,2) pixels with NaNs for unmatched.
    """
    R, C = grid_uv.shape[:2]
    N = R * C
    uv_flat = grid_uv.reshape(N, 2)
    m_flat = grid_mask.reshape(N)
    out = np.full((N, 2), np.nan, dtype=float)
    if preds_flat.size == 0:
        return out
    # For each predicted slot, pick nearest valid robust uv if within gate
    for j in range(N):
        if not np.isfinite(preds_flat[j, 0]):
            continue
        d = np.linalg.norm(uv_flat[m_flat] - preds_flat[j][None, :], axis=1)
        if d.size == 0:
            continue
        k_local = np.argmin(d)
        # map back to absolute index
        k_abs = np.flatnonzero(m_flat)[k_local]
        if d[k_local] <= gate_px:
            out[j] = uv_flat[k_abs]
    return out

