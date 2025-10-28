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
Mirror-Stereo Dot Grid sensor (e.g. Tac3D)

- Per-dot lattice output, with (ux, uy, uz) in millimeters in the gel surface frame.
- Handles wide aspect and partial FoV, handling missing far edge dots.
- No pixel rectification; uses point-space geometry (mirror plane to virtual camera) and anchor-guided lattice indexing.
- Returns NaN where a dot is not available; includes valid_mask and confidence in debug.
- Provides lattice_to_dense() to produce a dense field with [du(px), dv(px), uz(mm)] for the sensor surface.

Config fields expected sensor.params:
- nx, ny, dot_thresh, min_area, max_area, subpixel ("ellipse"|"moments")
- fx, fy, cx, cy, cam_center_mm[3], mirror_normal[3], mirror_point_mm[3]
- panel_w, panel_h, left_offset[2], right_offset[2]
- output_kind ("3d"|"2d"), return_confidence (bool)
- Optional: spacing_mm, diameter_mm, gel_tilt_deg, gel_center_mm — used only for sanity checks.

Implementation details:
- Anchor rule implemented:
  - Right panel (real view): lattice starts from left
  - Left panel (mirror): lattice starts from right
- Partial FoV: Rows are built by 1D kmeans on the PCA row coordinate; columns are ordered within each row from the anchor side.
  Missing far edge columns are simply unassigned; those lattice entries remain NaN.
- Pairing: align the left lattice to the right lattice via a small search over sign flips and integer offsets, then triangulate only over the intersection.
  then align the current right lattice to the rest right lattice (integer offset only) to get global (i,j) indices for output.
- Surface frame: computed once from the rest 3D lattice
- Dense util: lattice_to_dense() using Delaunay + barycentric interpolation (OpenCV Subdiv2D). 
  Pixels outside the convex hull are NaN.



"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
import math
import time

from ...config.settings import RuntimeConfig
from ...output.display import DebugDisplay
from ...output.visualizer import VideoWriters
from ...tir.types import Deformation
from ..base import SensorAdapter



# ========================= small numeric helpers ==============================

def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def _householder_reflection(vec: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    """Reflect a vector (or array of vectors) across a plane with unit normal n."""
    # vec shape (..., 3), n_unit shape (3,)
    d = np.sum(vec * n_unit, axis=-1, keepdims=True)
    return vec - 2.0 * d * n_unit

def _triangulate_two_rays(C1: np.ndarray, r1: np.ndarray,
                          C2: np.ndarray, r2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Midpoint of the shortest segment between two skew rays:
      X1 = C1 + t r1,  X2 = C2 + s r2
    r1, r2 assumed unit length.
    Returns (X, residual_mm).
    """
    a = float(np.dot(r1, r2))  # r1·r2
    b = C2 - C1                # vector from C1 to C2
    denom = 1.0 - a * a
    if denom < 1e-10:
        # nearly parallel – pick midpoint between closest projections along r1
        t = float(np.dot(b, r1))
        X1 = C1 + t * r1
        X2 = C2
    else:
        t = float(np.dot(b, (r1 - a * r2)) / denom)
        s = float(np.dot(b, (a * r1 - r2)) / denom)
        X1 = C1 + t * r1
        X2 = C2 + s * r2
    X = 0.5 * (X1 + X2)
    resid = float(np.linalg.norm(X1 - X2))
    return X, resid

def _build_K_and_inv(fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    Kinv = np.linalg.inv(K)
    return K, Kinv

def _to_ray(u: float, v: float, Kinv: np.ndarray) -> np.ndarray:
    x = Kinv @ np.array([u, v, 1.0], dtype=np.float32)
    return _norm(x)[..., :3]

def _pca_axes(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, and two unit PCA axes (e1 ~ columns, e2 ~ rows)."""
    m = xy.mean(axis=0, keepdims=True)
    X = xy - m
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0, :].astype(np.float32)
    e2 = Vt[1, :].astype(np.float32)
    # make right-handed and image-friendly (prefer e1 increasing to +x)
    if e1[0] < 0:
        e1 = -e1
    if np.cross(np.append(e1, 0), np.append(e2, 0))[2] < 0:
        e2 = -e2
    return m[0], _norm(e1), _norm(e2)

def _kmeans_1d(vals: np.ndarray, K: int, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means: returns (centroids[K], labels[N])."""
    v = vals.reshape(-1, 1).astype(np.float32)
    # init by quantiles
    qs = np.linspace(0.0, 1.0, K + 2)[1:-1]
    c = np.quantile(v, qs, axis=0).astype(np.float32)  # (K,1)
    if K == 1:
        return c[:, 0], np.zeros((v.shape[0],), np.int32)
    for _ in range(max(1, iters)):
        # assign
        d = (v - c.T) ** 2  # (N,K)
        lbl = np.argmin(d, axis=1)
        # update
        for k in range(K):
            m = (lbl == k)
            if np.any(m):
                c[k, 0] = float(v[m].mean())
    return c[:, 0], lbl.astype(np.int32)

def _detect_bright_dots(gray: np.ndarray, thr: int, min_area: int, max_area: int,
                        subpixel: str = "ellipse") -> Tuple[np.ndarray, np.ndarray]:
    """
    Bright-blob detector: threshold -> CC.
    Returns centers (N,2) float32 in pixel coords (u,v) and areas (N,).
    """
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    centers = cents[1:, :]  # (N,2) float64
    keep = (areas >= min_area) & (areas <= max_area)
    centers = centers[keep, :].astype(np.float32)
    areas = areas[keep].astype(np.float32)

    if centers.shape[0] == 0:
        return centers, areas

    if subpixel == "moments":
        # Use CC-provided centroids (already subpixel)
        return centers, areas

    # Optional ellipse refine (center only)
    # Build contours by label; this is more expensive, so only for small N.
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    refined = []
    for cnt in contours:
        a = float(cv2.contourArea(cnt))
        if a < min_area or a > max_area or len(cnt) < 5:
            continue
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        refined.append([cx, cy, a])
    if len(refined) >= 0:
        ref = np.array(refined, dtype=np.float32)
        # Re-match by nearest area to avoid reordering
        out = []
        for c0, a0 in zip(centers, areas):
            d = np.abs(ref[:, 2] - a0)
            k = int(np.argmin(d))
            out.append(ref[k, :2])
        centers = np.array(out, dtype=np.float32)
    return centers, areas





# ====================== lattice indexing & pairing ============================

def _index_panel_lattice(centers_uv: np.ndarray, nx: int, ny: int,
                         anchor: str) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int,int], int]]:
    """
    Index an unordered set of dot centers into a (partial) lattice.
    - centers_uv: (N,2) [u,v] in *panel* coordinates (not global).
    - nx, ny: full grid size (we accept partial FoV).
    - anchor: "left" (start from min column) or "right" (start from max column).
    Returns:
      ij_for_pt: (N,2) int32 local indices (i,j) counted from the anchor side.
      rows_sorted: list of arrays of point indices per row (for debug).
      map_ij_to_pt: dict {(i,j)->pt_index} of visible nodes.
    """
    N = centers_uv.shape[0]
    if N == 0:
        return (np.zeros((0, 2), np.int32),
                [np.array([], np.int32) for _ in range(ny)],
                {})

    # PCA axes
    m, e1, e2 = _pca_axes(centers_uv)
    C = centers_uv - m[None, :]
    c1 = C @ e1  # columns
    c2 = C @ e2  # rows

    # 1D K-means on c2 to form rows (robust to missing columns)
    cents, lbl = _kmeans_1d(c2, ny, iters=40)  # (ny,), (N,)
    order_rows = np.argsort(cents)             # top->bottom
    # build per-row lists
    rows_sorted: List[np.ndarray] = []
    ij_for_pt = np.empty((N, 2), np.int32)
    map_ij_to_pt: Dict[Tuple[int, int], int] = {}

    for j_ord, rid in enumerate(order_rows):
        pts = np.where(lbl == rid)[0]
        if pts.size == 0:
            rows_sorted.append(np.array([], np.int32))
            continue
        # sort this row by c1 (left->right in rectified space)
        s = pts[np.argsort(c1[pts])]
        if anchor == "right":
            s = s[::-1]  # start from right edge if mirrored panel is fully visible on right
        rows_sorted.append(s)

        # assign i index starting from anchor side
        for i_local, k in enumerate(s):
            ij_for_pt[k, 0] = i_local
            ij_for_pt[k, 1] = j_ord
            map_ij_to_pt[(i_local, j_ord)] = int(k)

    return ij_for_pt, rows_sorted, map_ij_to_pt

def _best_overlap_transform(ijA: np.ndarray, ijB: np.ndarray,
                            allow_flip: bool = True) -> Tuple[int, int, int, int, int]:
    """
    Find integer transform (with optional sign flips) that maximizes the overlap
    between two index sets A and B.
    Returns: (s_i, s_j, di, dj, overlap_count)
      such that (iA', jA') = (s_i*iA + di, s_j*jA + dj) best matches B.
    """
    if ijA.size == 0 or ijB.size == 0:
        return 1, 1, 0, 0, 0

    setB = set((int(i), int(j)) for i, j in ijB)
    signs = [(1, 1)]
    if allow_flip:
        signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    best = (1, 1, 0, 0, -1)
    for si, sj in signs:
        # compute histogram of (iB - si*iA, jB - sj*jA) for all pairs by mode search
        # We'll approximate by using medians of differences of nearest neighbors in L1.
        # Faster: use a coarse hash on differences.
        diffs: Dict[Tuple[int, int], int] = {}
        for (i, j) in ijA:
            # try to align to each B by assuming same (i,j); estimate di,dj from a single sample:
            # using nearest integer assumption is fine because lattice indexing is discrete.
            # We'll vote di,dj that would align this A sample to some B with same (i,j).
            # In practice, take di,dj from aligning to the *closest* (i,j) in B (by rounding).
            di_cand = None
            dj_cand = None
            # snap to closest B in L1 (coarse)
            # build neighbors around (si*i, sj*j)
            for di_try in (-2, -1, 0, 1, 2):
                for dj_try in (-2, -1, 0, 1, 2):
                    key = (si * int(i) + di_try, sj * int(j) + dj_try)
                    if key in setB:
                        di_cand = di_try
                        dj_cand = dj_try
                        break
                if di_cand is not None:
                    break
            if di_cand is None:
                continue
            key = (si, sj, di_cand, dj_cand)
            diffs[(di_cand, dj_cand)] = diffs.get((di_cand, dj_cand), 0) + 1

        if not diffs:
            continue
        (di, dj), cnt = max(diffs.items(), key=lambda kv: kv[1])
        # compute exact overlap with this candidate
        overlap = 0
        for (i, j) in ijA:
            if (si * int(i) + di, sj * int(j) + dj) in setB:
                overlap += 1
        if overlap > best[4]:
            best = (si, sj, int(di), int(dj), int(overlap))
    return best



# ============================ main adapter ====================================

class Tac3DAdapter(SensorAdapter):
    """
    Tac3D-style mirror stereo adapter.
    - Outputs per-dot lattice deformation in the gel-surface frame (mm).
    - No pixel rectification; operates in coordinate space.
    - Robust to partial FoV; missing nodes return NaN.
    """

    def __init__(self,
                 cfg: RuntimeConfig,
                 dbg_disp: Optional[DebugDisplay] = None,
                 dbg_writers: Optional[VideoWriters] = None) -> None:
        super().__init__(cfg, dbg_disp, dbg_writers)

        s = getattr(cfg, "sensor", None)
        # lattice & detection
        self.nx: int = int(getattr(s, "nx", 18))
        self.ny: int = int(getattr(s, "ny", 18))
        self.dot_thresh: int = int(getattr(s, "dot_thresh", 180))
        self.min_area: int = int(getattr(s, "min_area", 20))
        self.max_area: int = int(getattr(s, "max_area", 2000))
        self.subpixel: str = str(getattr(s, "subpixel", "ellipse")).lower()

        # intrinsics (pixels)
        self.fx: float = float(getattr(s, "fx", 600.0))
        self.fy: float = float(getattr(s, "fy", 600.0))
        self.cx: float = float(getattr(s, "cx", 0.0))
        self.cy: float = float(getattr(s, "cy", 0.0))

        # camera center (mm) and mirror plane (mm)
        cam_center = getattr(s, "cam_center_mm", [0.0, 0.0, 0.0])
        self.C = np.array([float(cam_center[0]), float(cam_center[1]), float(cam_center[2])], np.float32)
        mp = getattr(s, "mirror_point_mm", [0.0, 0.0, 0.0])
        mn = getattr(s, "mirror_normal", [1.0, 0.0, 0.0])
        self.mirror_p = np.array(mp, dtype=np.float32)
        self.mirror_n = _norm(np.array(mn, dtype=np.float32))

        # panel ROIs in the composite image
        self.panel_w: int = int(getattr(s, "panel_w", 640))
        self.panel_h: int = int(getattr(s, "panel_h", 480))
        self.left_off = tuple(int(x) for x in getattr(s, "left_offset", [0, 0]))
        self.right_off = tuple(int(x) for x in getattr(s, "right_offset", [self.panel_w, 0]))

        # outputs
        self.output_kind: str = str(getattr(s, "output_kind", "3d")).lower()
        self.return_conf: bool = bool(getattr(s, "return_confidence", True))

        # intrinsics matrices
        self.K, self.Kinv = _build_K_and_inv(self.fx, self.fy, self.cx, self.cy)

        # --- rest state storages ---
        self.rest_right_uv: Optional[np.ndarray] = None    # (ny, nx, 2) global pixels
        self.rest_left_uv: Optional[np.ndarray] = None     # (ny, nx, 2) global pixels
        self.rest_P: Optional[np.ndarray] = None           # (ny, nx, 3) world mm (only where both panels available)
        self.valid_mask0: Optional[np.ndarray] = None      # (ny, nx) bool for rest triangulated nodes

        # surface frame
        self.Tx: Optional[np.ndarray] = None  # (3,) unit tangent x
        self.Ty: Optional[np.ndarray] = None  # (3,) unit tangent y
        self.Nz: Optional[np.ndarray] = None  # (3,) unit normal
        self.center_mm: Optional[np.ndarray] = None  # (3,)

        # mapping meta (debug)
        self.lr_si_sj_di_dj = None
        self.rr_di_dj = None
        self.baseline_mm: Optional[float] = None

    # ---------------------------- core helpers --------------------------------

    def _split_panels(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xL, yL = self.left_off
        xR, yR = self.right_off
        w, h = self.panel_w, self.panel_h
        left = bgr[yL:yL+h, xL:xL+w, :]
        right = bgr[yR:yR+h, xR:xR+w, :]
        return left, right

    def _detect_panel(self, bgr_panel: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr_panel, cv2.COLOR_BGR2GRAY)
        centers, _ = _detect_bright_dots(g, self.dot_thresh, self.min_area, self.max_area, self.subpixel)
        return centers  # (N,2) panel coords

    def _globalize_uv(self, uv_panel: np.ndarray, side: str) -> np.ndarray:
        """Convert panel-local (u,v) to composite-image (global) pixels."""
        off = self.left_off if side == "left" else self.right_off
        if uv_panel.size == 0:
            return uv_panel.astype(np.float32)
        uv = uv_panel.copy().astype(np.float32)
        uv[:, 0] += float(off[0])
        uv[:, 1] += float(off[1])
        return uv

    def _pair_indices(self, ijL: np.ndarray, ijR: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int,int]]:
        """Find best sign+offset to align L indices to R indices."""
        si, sj, di, dj, ov = _best_overlap_transform(ijL, ijR, allow_flip=True)
        self.lr_si_sj_di_dj = (si, sj, di, dj, ov)
        # compute intersection pairs
        setR = {(int(i), int(j)) for (i, j) in ijR}
        pairs = []
        for (i, j) in ijL:
            iR = si * int(i) + di
            jR = sj * int(j) + dj
            if (iR, jR) in setR:
                pairs.append((iR, jR, int(i), int(j)))
        return np.array(pairs, np.int32), (si, sj, di, dj, ov)

    def _align_right_to_rest(self, ijR_now: np.ndarray, ijR_rest: np.ndarray) -> Tuple[int, int, int]:
        si, sj, di, dj, ov = _best_overlap_transform(ijR_now, ijR_rest, allow_flip=False)
        self.rr_di_dj = (di, dj, ov)
        return di, dj, ov

    def _compute_surface_frame(self, P: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        From rest P (ny,nx,3) with valid mask -> (Tx, Ty, Nz, center).
        Tx ~ median of (i+1,j)-(i,j); Ty ~ median of (i,j+1)-(i,j).
        """
        ny, nx, _ = P.shape
        # center
        center = np.nanmean(P[valid], axis=0).astype(np.float32)

        # Tx
        Vx = []
        for j in range(ny):
            for i in range(nx - 1):
                if valid[j, i] and valid[j, i + 1]:
                    Vx.append(P[j, i + 1] - P[j, i])
        Vy = []
        for j in range(ny - 1):
            for i in range(nx):
                if valid[j, i] and valid[j + 1, i]:
                    Vy.append(P[j + 1, i] - P[j, i])
        if len(Vx) == 0 or len(Vy) == 0:
            # fall back to PCA of all (less robust)
            Q = P[valid] - center[None, :]
            U, S, Vt = np.linalg.svd(Q, full_matrices=False)
            tx = _norm(Vt[0, :])
            ty = _norm(Vt[1, :])
        else:
            tx = _norm(np.median(np.stack(Vx, axis=0), axis=0))
            ty = _norm(np.median(np.stack(Vy, axis=0), axis=0))
        nz = _norm(np.cross(tx, ty))
        # re-orthonormalize Ty to be orthogonal to Tx
        ty = _norm(np.cross(nz, tx))
        return tx.astype(np.float32), ty.astype(np.float32), nz.astype(np.float32), center.astype(np.float32)

    # ------------------------------ prepare -----------------------------------

    def prepare(self, first_bgr: np.ndarray) -> None:
        """
        Detect, index, pair, triangulate rest state; build surface frame.
        """
        # 1) detect per panel
        left, right = self._split_panels(first_bgr)
        uvL_local = self._detect_panel(left)
        uvR_local = self._detect_panel(right)

        uvL = self._globalize_uv(uvL_local, "left")
        uvR = self._globalize_uv(uvR_local, "right")

        # 2) lattice indexing (anchor: right panel 'left', left panel 'right')
        ijL, rowsL, mapL = _index_panel_lattice(uvL_local, self.nx, self.ny, anchor="right")
        ijR, rowsR, mapR = _index_panel_lattice(uvR_local, self.nx, self.ny, anchor="left")

        # 3) align L to R and assemble matched pairs
        pairs, (si, sj, di, dj, ov) = self._pair_indices(ijL, ijR)

        # 4) triangulate matched dots -> P0
        C = self.C.astype(np.float32)
        Cvirt = _householder_reflection(C[None, :], self.mirror_n)[0] + (self.mirror_p - _householder_reflection(self.mirror_p[None, :], self.mirror_n)[0])
        # Above simpler: virtual center is reflection of C across plane
        Cvirt = C - 2.0 * np.dot((C - self.mirror_p), self.mirror_n) * self.mirror_n
        self.baseline_mm = float(np.linalg.norm(Cvirt - C))

        ny, nx = self.ny, self.nx
        P0 = np.full((ny, nx, 3), np.nan, np.float32)
        mask0 = np.zeros((ny, nx), bool)

        for iR, jR, iLloc, jLloc in pairs:
            # indices in right panel local lattice
            if iR < 0 or jR < 0 or iR >= nx or jR >= ny:
                continue
            # right uv
            kR = mapR.get((iR, jR), None)
            # left uv (local indices anchored from right side)
            kL = mapL.get((iLloc, jLloc), None)
            if kR is None or kL is None:
                continue

            uR, vR = uvR[kR]
            uL, vL = uvL[kL]

            # rays
            rR = _to_ray(uR, vR, self.Kinv)
            rL = _to_ray(uL, vL, self.Kinv)
            rV = _householder_reflection(rL[None, :], self.mirror_n)[0]

            X, resid = _triangulate_two_rays(C, rR, Cvirt, rV)
            P0[jR, iR, :] = X
            mask0[jR, iR] = True

        if not np.any(mask0):
            print("[Tac3DAdapter] ERROR: no matched dots after pairing in prepare().")
            return

        # 5) surface frame from P0
        Tx, Ty, Nz, center = self._compute_surface_frame(P0, mask0)

        self.rest_P = P0
        self.valid_mask0 = mask0
        # save rest UVs (global) per panel as dense lattice (NaN where missing)
        UVR = np.full((ny, nx, 2), np.nan, np.float32)
        UVL = np.full((ny, nx, 2), np.nan, np.float32)
        for (i, j), k in mapR.items():
            if 0 <= i < nx and 0 <= j < ny:
                UVR[j, i, :] = uvR[k]
        for (i, j), k in mapL.items():
            # bring left indices into right frame: (iR, jR) = (si*i + di, sj*j + dj)
            iR = si * int(i) + di
            jR = sj * int(j) + dj
            if 0 <= iR < nx and 0 <= jR < ny:
                UVL[jR, iR, :] = uvL[k]

        self.rest_right_uv = UVR
        self.rest_left_uv = UVL
        self.Tx, self.Ty, self.Nz, self.center_mm = Tx, Ty, Nz, center

    # ------------------------------- process ----------------------------------

    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        self.frame_id += 1
        if self.rest_P is None or self.valid_mask0 is None:
            print("[Tac3DAdapter] Error: prepare() must be called before process().")
            return None

        ny, nx = self.ny, self.nx

        # detect now
        left, right = self._split_panels(bgr)
        uvL_local = self._detect_panel(left)
        uvR_local = self._detect_panel(right)
        uvL = self._globalize_uv(uvL_local, "left")
        uvR = self._globalize_uv(uvR_local, "right")

        # index per panel
        ijL, rowsL, mapL = _index_panel_lattice(uvL_local, nx, ny, anchor="right")
        ijR, rowsR, mapR = _index_panel_lattice(uvR_local, nx, ny, anchor="left")

        # align L->R for this frame
        pairsLR, (si, sj, di, dj, ov) = self._pair_indices(ijL, ijR)

        # align right-now to right-rest (global lattice)
        # Build right-rest index set from rest_right_uv (valid where not NaN)
        ijR_rest = []
        for j in range(ny):
            for i in range(nx):
                if not np.any(np.isnan(self.rest_right_uv[j, i, :])):
                    ijR_rest.append((i, j))
        ijR_rest = np.array(ijR_rest, np.int32)
        diR, djR, ovR = self._align_right_to_rest(ijR, ijR_rest)

        # Triangulate and compute deformation in surface frame
        C = self.C.astype(np.float32)
        Cvirt = C - 2.0 * np.dot((C - self.mirror_p), self.mirror_n) * self.mirror_n

        data = np.full((ny, nx, 3), np.nan, np.float32)
        conf = np.full((ny, nx), np.nan, np.float32)

        for iR_now, jR_now, iLloc, jLloc in pairsLR:
            # map right-now index to global lattice (rest alignment)
            iG = iR_now - diR
            jG = jR_now - djR
            if iG < 0 or jG < 0 or iG >= nx or jG >= ny:
                continue
            # require that rest had a triangulated point at this global index
            if not self.valid_mask0[jG, iG]:
                continue

            kR = mapR.get((iR_now, jR_now), None)
            kL = mapL.get((iLloc, jLloc), None)
            if kR is None or kL is None:
                continue

            uR, vR = uvR[kR]
            uL, vL = uvL[kL]
            rR = _to_ray(uR, vR, self.Kinv)
            rL = _to_ray(uL, vL, self.Kinv)
            rV = _householder_reflection(rL[None, :], self.mirror_n)[0]

            X, resid = _triangulate_two_rays(C, rR, Cvirt, rV)

            # deformation in surface frame (mm)
            X0 = self.rest_P[jG, iG, :]
            dP = (X - X0).astype(np.float32)
            ux = float(np.dot(dP, self.Tx))
            uy = float(np.dot(dP, self.Ty))
            uz = float(np.dot(dP, self.Nz))
            data[jG, iG, 0] = ux
            data[jG, iG, 1] = uy
            data[jG, iG, 2] = uz
            conf[jG, iG] = 1.0 / (1.0 + resid)  # simple residual-based confidence

        # Package TIR
        if self.output_kind == "2d":
            # return ux, uy only (mm); channels order [uy, ux] or [ux, uy]? We'll use [ux, uy].
            arr = data[..., :2]
        else:
            arr = data  # (ny,nx,3)

        # replace NaN with 0 for TIR
        arr = np.nan_to_num(arr, nan=0.0)

        # shape -> (1, H, W, C)
        out = Deformation(
            data=arr[None, ...].astype(np.float32),
            kind='3d' if self.output_kind != "2d" else '2d',
            meta=self._mk_meta(ny, nx),
            z_of_layer=None,
            debug={
                "valid_mask": np.isfinite(data[..., 0]) & np.isfinite(data[..., 1]),
                "confidence": conf if self.return_conf else None,
                "surface_frame": {
                    "Tx": self.Tx, "Ty": self.Ty, "Nz": self.Nz, "center_mm": self.center_mm
                },
                "panel_roi": {
                    "left": {"offset": self.left_off, "wh": (self.panel_w, self.panel_h)},
                    "right": {"offset": self.right_off, "wh": (self.panel_w, self.panel_h)}
                },
                "pairing": {
                    "LtoR": (si, sj, di, dj, ov),
                    "RtoRest": (diR, djR, ovR),
                    "baseline_mm": self.baseline_mm
                }
            }
        )
        return out


# =================== lattice -> dense util (right panel) ======================

def lattice_to_dense(right_uv_rest: np.ndarray,
                     right_uv_now: np.ndarray,
                     uz_mm_lattice: np.ndarray,
                     panel_wh: Tuple[int, int],
                     fill_outside_with_nan: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate per-dot deformations to a dense (H,W) field for the *right* panel ROI.

    Inputs:
      right_uv_rest : (ny,nx,2) global pixel coords at rest (NaN where missing)
      right_uv_now  : (ny,nx,2) global pixel coords now (NaN where missing)
      uz_mm_lattice : (ny,nx) normal deformation in mm (NaN where missing)
      panel_wh      : (W,H) of the right ROI
    Returns:
      Vx, Vy, Z : each (H,W) float32,
                  Vx,Vy in *pixel* units (Δu, Δv), Z in mm.
                  Outside convex hull -> NaN (or zeros if fill_outside_with_nan=False).
    """
    W, H = int(panel_wh[0]), int(panel_wh[1])

    # Gather valid vertices
    mask = np.isfinite(right_uv_rest[..., 0]) & np.isfinite(right_uv_rest[..., 1]) \
           & np.isfinite(right_uv_now[..., 0]) & np.isfinite(right_uv_now[..., 1]) \
           & np.isfinite(uz_mm_lattice)
    js, is_ = np.where(mask)
    if js.size < 3:
        # not enough points to triangulate
        Vx = np.full((H, W), np.nan, np.float32)
        Vy = np.full((H, W), np.nan, np.float32)
        Z = np.full((H, W), np.nan, np.float32)
        if not fill_outside_with_nan:
            Vx.fill(0.0); Vy.fill(0.0); Z.fill(0.0)
        return Vx, Vy, Z

    P = right_uv_rest[mask].astype(np.float32)   # (N,2)
    Q = right_uv_now[mask].astype(np.float32)    # (N,2)
    d = (Q - P).astype(np.float32)               # (N,2) Δu,Δv
    z = uz_mm_lattice[mask].astype(np.float32)   # (N,)

    # Delaunay via OpenCV
    rect = (0, 0, W, H)
    subdiv = cv2.Subdiv2D(rect)
    for (u, v) in P:
        # clamp insert positions to ROI; Subdiv2D expects int-ish coords within rect
        uu = float(np.clip(u, 0, W - 1))
        vv = float(np.clip(v, 0, H - 1))
        subdiv.insert((uu, vv))

    tris = subdiv.getTriangleList()  # (M,6): x1,y1,x2,y2,x3,y3 in float
    if tris is None or len(tris) == 0:
        Vx = np.full((H, W), np.nan, np.float32)
        Vy = np.full((H, W), np.nan, np.float32)
        Z = np.full((H, W), np.nan, np.float32)
        if not fill_outside_with_nan:
            Vx.fill(0.0); Vy.fill(0.0); Z.fill(0.0)
        return Vx, Vy, Z

    # map triangle vertices back to nearest input points (small N -> brute force ok)
    def nearest_idx(pt: np.ndarray) -> int:
        diff = P - pt[None, :]
        k = int(np.argmin(np.einsum('ij,ij->i', diff, diff)))
        return k

    Vx = np.full((H, W), np.nan, np.float32)
    Vy = np.full((H, W), np.nan, np.float32)
    Z  = np.full((H, W), np.nan, np.float32)

    for t in tris:
        x1, y1, x2, y2, x3, y3 = t.tolist()
        tri = np.array([[x1, y1], [x2, y2], [x3, y3]], np.float32)

        # clip triangles outside rect
        if (tri[:, 0].max() < 0) or (tri[:, 1].max() < 0) or \
           (tri[:, 0].min() > W - 1) or (tri[:, 1].min() > H - 1):
            continue

        k1 = nearest_idx(tri[0]); k2 = nearest_idx(tri[1]); k3 = nearest_idx(tri[2])
        # three vertices (rest positions) and per-vertex values
        A = np.array([P[k1], P[k2], P[k3]], np.float32)  # (3,2)
        vals_vx = np.array([d[k1, 0], d[k2, 0], d[k3, 0]], np.float32)
        vals_vy = np.array([d[k1, 1], d[k2, 1], d[k3, 1]], np.float32)
        vals_z  = np.array([z[k1],    z[k2],    z[k3]   ], np.float32)

        # barycentric transform precompute
        M = np.array([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]],
            [1.0,     1.0,     1.0    ]], np.float32)
        Minv = np.linalg.pinv(M)

        xmin = int(max(0, math.floor(A[:, 0].min())))
        xmax = int(min(W - 1, math.ceil(A[:, 0].max())))
        ymin = int(max(0, math.floor(A[:, 1].min())))
        ymax = int(min(H - 1, math.ceil(A[:, 1].max())))

        # rasterize bbox
        uu = np.arange(xmin, xmax + 1, dtype=np.float32)
        vv = np.arange(ymin, ymax + 1, dtype=np.float32)
        U, V = np.meshgrid(uu, vv, indexing="xy")
        ones = np.ones_like(U)
        B = np.stack([U, V, ones], axis=-1)  # (Hbb, Wbb, 3)
        Wb = B @ Minv.T                      # barycentric weights (Hbb, Wbb, 3)

        inside = (Wb[..., 0] >= -1e-4) & (Wb[..., 1] >= -1e-4) & (Wb[..., 2] >= -1e-4) \
                 & (np.abs(np.sum(Wb, axis=-1) - 1.0) <= 1e-3)

        if not np.any(inside):
            continue

        vx_patch = (Wb[..., 0] * vals_vx[0] + Wb[..., 1] * vals_vx[1] + Wb[..., 2] * vals_vx[2]).astype(np.float32)
        vy_patch = (Wb[..., 0] * vals_vy[0] + Wb[..., 1] * vals_vy[1] + Wb[..., 2] * vals_vy[2]).astype(np.float32)
        z_patch  = (Wb[..., 0] * vals_z [0] + Wb[..., 1] * vals_z [1] + Wb[..., 2] * vals_z [2]).astype(np.float32)

        Vx[ymin:ymax+1, xmin:xmax+1][inside] = vx_patch[inside]
        Vy[ymin:ymax+1, xmin:xmax+1][inside] = vy_patch[inside]
        Z [ymin:ymax+1, xmin:xmax+1][inside] = z_patch [inside]

    if not fill_outside_with_nan:
        Vx = np.nan_to_num(Vx, nan=0.0)
        Vy = np.nan_to_num(Vy, nan=0.0)
        Z  = np.nan_to_num(Z,  nan=0.0)

    return Vx, Vy, Z

