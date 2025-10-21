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
Adapter for photometric gel tactile sensors with a regular printed marker grid.
E.g. GelSight-style.
- Produces a dense 2D deformation field (vy, vx) from a regular marker grid printed on the reflective layer.
- API-ready stub for 3D deformation (ux, uy, uz). Depth from photometric 3D will be added later.

Design goals:
- Frame-independent: no temporal tracking; robust to big/sudden motion and dropped frames.
- Real-time: only thresholding, connected components, PCA, sorting, and bilinear interpolation.

Assumptions:
- All dots are visible in every frame.
- Marker dots form a regular rectangular grid.
- First frame is rest (or near-rest). We index the lattice there and store a linear pixel<->grid map.

Config usage (read from RuntimeConfig at runtime):
    cfg.sensor.type               == "gelsight" (or "gelsight_style")
    cfg.sensor.nx, cfg.sensor.ny  marker grid size
    cfg.sensor.dot_thresh         gray threshold for black dot detection
    cfg.sensor.min_area, max_area blob area range
    cfg.physics.mm_per_px

Outputs:
- Deformation(kind='2d'): data shape (1, H, W, 2) with channels [vy, vx] in pixel units.
- Deformation(kind='3d') with channels [ux, uy, uz] in mm.


Usage Example:
python -m runtime.run_demo \
    --source video   --input dataset/sequences/000001/video/sequence.mp4   --loop \
    --config runtime/config/runtime_config_gelsight.yaml \
    --vis2d --vis3d
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from synth import deform

from ...config.settings import RuntimeConfig
from ...output.display import DebugDisplay
from ...output.visualizer import (
    flow_to_color_bgr, draw_quiver_bgr, draw_quiver_grid_bgr,
    scalar_to_color_bgr, VideoWriters
)
from ...preprocessing.optical_flow import to_gray_f32_bgr
from ...tir.types import Deformation
from ..base import SensorAdapter

# ---------------------------- small helpers ----------------------------------

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.uint8)

def _detect_dot_centers(gray_u8: np.ndarray,
                        thresh: int,
                        min_area: int,
                        max_area: int) -> np.ndarray:
    """
    black-dot detector: threshold -> CC -> centroid.
    Returns (N,2) float32 array of [x,y] in pixel coordinates (x=col=u, y=row=v).
    """
    # Threshold for "black"
    _, bw = cv2.threshold(gray_u8, thresh, 255, cv2.THRESH_BINARY_INV)

    # Connected components
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # num is count, labels is (H,W) int, stats is (num,5), cents is (num,2) float
    if num <= 1:
        return np.empty((0, 2), dtype=np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    centers = cents[1:, :]  # exclude background
    keep = (areas >= min_area) & (areas <= max_area)
    centers = centers[keep, :]  # (K,2): [x, y] = [u, v]
    return centers.astype(np.float32)

def _pca_axes(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA axes for 2D points.  For aligning the marker grid.

    Returns unit vectors (e1, e2) as rows (shape (2,)).
    - e1 ~ major axis (columns)
    - e2 ~ minor axis (rows).
    """

    X = xy - xy.mean(axis=0, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    e1 = Vt[0, :]  # major
    e2 = Vt[1, :]  # minor

    # Make axes right-handed and image-aligned
    if e1[0] < 0:  # prefer e1 pointing to +x
        e1 = -e1
    if np.cross(np.append(e1, 0), np.append(e2, 0))[2] < 0:
        e2 = -e2
    return e1.astype(np.float32), e2.astype(np.float32)

def _index_grid_from_points(xy: np.ndarray, nx: int, ny: int) -> Optional[np.ndarray]:
    """
    Assign lattice indices (i,j) to unordered points using PCA & sorting.

    Steps:
    - Project onto PCA axes (e1 ~ columns, e2 ~ rows).
    - Sort by e2, and split into ny contiguous groups of equal size nx.
    - Within each group (row), sort by e1 to assign i=0..nx-1.
    - Finally, order rows by ascending mean e2 to assign j=0..ny-1.

    Inputs:
        xy (N,2): detected dot centers in pixel coordinates.
        nx, ny: expected grid size.

    Returns:
        idx (N,2) array of integer indices (i,j) for each xy[k].
    """
    assert xy.ndim == 2 and xy.shape[1] == 2
    N = xy.shape[0]
    if N != nx * ny:
        # best-effort: round to closest rectangular factors
        print(f"[GelSightAdapter] Error: detected {N} dots but expected nx*ny={nx*ny}")
        return None

    # Find grid's principal axes
    e1, e2 = _pca_axes(xy)
    c1 = xy @ e1  # projection ~ columns
    c2 = xy @ e2  # projection ~ rows

    order = np.argsort(c2)  # sort by row coordinate
    c1_sorted = c1[order]
    c2_sorted = c2[order]

    idx = np.empty((N, 2), dtype=np.int32)

    # Partition into ny bands of size nx
    bands = np.array_split(np.arange(N, dtype=np.int32), ny)
    # Ensure equal sizes; in normal operation N==nx*ny so every band has nx entries.
    for j, band in enumerate(bands):
        # within a band, sort by c1 to order columns
        band_sorted = band[np.argsort(c1_sorted[band])]
        # assign i=0..nx-1
        if band_sorted.size != nx:
            # fall back: pick the closest nx samples
            band_sorted = band_sorted[:nx]
        idx_rows = order[band_sorted]  # indices into xy
        # Fill idx for those points
        for ii, k in enumerate(idx_rows):
            idx[k, 0] = ii  # i
            idx[k, 1] = j   # j

    # Now rows are in ascending image y (top->bottom). 
    # And already used ascending c2, which correlates with rows
    # So now the grid order should be correct.
    return idx

def _fit_affine_grid_to_pixels(grid_ij: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit an affine map:
        [u, v, 1]^T ~= T @ [i, j, 1]^T
    where T is R^{3x3}, last row is [0,0,1]
    Return (T, T_inv).
    """
    N = grid_ij.shape[0]
    A = np.concatenate([grid_ij, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,3) = [i, j, 1]
    u = uv[:, 0]; v = uv[:, 1]

    # Least squares for u = a*i + b*j + c ; v = d*i + e*j + f
    coef_u, _, _, _ = np.linalg.lstsq(A, u, rcond=None)
    coef_v, _, _, _ = np.linalg.lstsq(A, v, rcond=None)

    T = np.array([[coef_u[0], coef_u[1], coef_u[2]],
                  [coef_v[0], coef_v[1], coef_v[2]],
                  [0.0,       0.0,       1.0     ]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    return T, T_inv

def _bilinear_dense_from_grid(flow_gx: np.ndarray, flow_gy: np.ndarray,
                              T_inv: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a dense deformation field on the image pixel coordinate from a grid (i,j).

    Inputs:
    - flow_gx, flow_gy: (ny, nx) deformation in pixels at grid nodes (columns x, rows y).
    - T_inv: inverse affine mapping from pixels -> grid indices.
    Returns:
        vy (H,W), vx (H,W) in pixel units.
    """
    ny, nx = flow_gx.shape
    # Build per-pixel grid coordinates (i,j) via inverse affine
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32), indexing="xy")
    ones = np.ones_like(uu)
    uv1 = np.stack([uu, vv, ones], axis=-1)  # (H, W, 3)
    ij1 = uv1 @ T_inv.T                      # (H, W, 3)
    i_f = ij1[..., 0]; j_f = ij1[..., 1]

    # Bilinear weights
    i0 = np.floor(i_f).astype(np.int32)
    j0 = np.floor(j_f).astype(np.int32)
    i1 = i0 + 1; j1 = j0 + 1

    valid = (i0 >= 0) & (i1 < nx) & (j0 >= 0) & (j1 < ny)

    i0c = np.clip(i0, 0, nx - 2)
    j0c = np.clip(j0, 0, ny - 2)
    i1c = i0c + 1; j1c = j0c + 1

    wx = np.clip(i_f - i0c.astype(np.float32), 0.0, 1.0)
    wy = np.clip(j_f - j0c.astype(np.float32), 0.0, 1.0)

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    # Gather node flows
    f00x = flow_gx[j0c, i0c]; f10x = flow_gx[j0c, i1c]
    f01x = flow_gx[j1c, i0c]; f11x = flow_gx[j1c, i1c]
    f00y = flow_gy[j0c, i0c]; f10y = flow_gy[j0c, i1c]
    f01y = flow_gy[j1c, i0c]; f11y = flow_gy[j1c, i1c]

    vx = w00 * f00x + w10 * f10x + w01 * f01x + w11 * f11x
    vy = w00 * f00y + w10 * f10y + w01 * f01y + w11 * f11y

    # Mask invalid pixels (outside convex hull of the grid) with zeros (caller may ignore or visualize)
    vx = np.where(valid, vx, 0.0).astype(np.float32)
    vy = np.where(valid, vy, 0.0).astype(np.float32)
    return vy, vx


# ------------------------------ Adapter --------------------------------------

class PhotometricAdapter(SensorAdapter):
    """
    Photometric tactile sensor (e.g. GelSight-style)'s pre-processing:
      - Detect and index regular dot grid from scratch per frame (no temporal dependence).
      - Build sparse grid displacement (pixels) vs. rest state.
      - Interpolate to a dense (H,W) field and pack as Deformation(kind='2d') for the solver.
      - TODO: add 3D deformation (ux,uy,uz), depth to be implemented via PS.

    Notes:
      - Expects input frames already downscaled by cfg.downscale.
      - Outputs per-pixel deformation value is in *original camera pixel units* (divides by cfg.downscale).
    """

    def __init__(self,
                 cfg: RuntimeConfig,
                 dbg_disp: Optional[DebugDisplay] = None,
                 dbg_writers: Optional[VideoWriters] = None) -> None:
        super().__init__(cfg, dbg_disp, dbg_writers)

        s = getattr(cfg, "sensor", None)
        print("[GelSightAdapter] Initializing with sensor params:", s)
        
        self.nx: int = int(getattr(s, "nx", 40))
        self.ny: int = int(getattr(s, "ny", 30))
        self.dot_thresh: int = int(getattr(s, "dot_thresh", 40))
        self.min_area: int = int(getattr(s, "min_area", 5))
        self.max_area: int = int(getattr(s, "max_area", 60))

        # Precomputed rest-state data on first frame
        self.rest_grid_uv: Optional[np.ndarray] = None   # (ny, nx, 2) float32
        self.T: Optional[np.ndarray] = None              # pixels <- grid  (3x3)
        self.T_inv: Optional[np.ndarray] = None          # grid   <- pixels (3x3)

        # 3D deformation
        self.deformation3d: Optional[Deformation] = None

    def prepare(self, first_bgr: np.ndarray) -> None:
        """
        Build rest-state grid
        - detect dots
        - index (i,j)
        - fit affine map pixels<->grid.
        """
        print("[GelSightAdapter] prepare()")

        # Detect dots in first frame
        gray = _to_gray(first_bgr)
        centers = _detect_dot_centers(gray, self.dot_thresh, self.min_area, self.max_area)
        if centers.shape[0] != self.nx * self.ny:
            print(f"[GelSightAdapter] WARNING: detected {centers.shape[0]} dots; expected {self.nx*self.ny}. "
                  f"Trying anyway with PCA indexer.")
        idx_ij = _index_grid_from_points(centers, self.nx, self.ny)  # (N,2) int

        if idx_ij is None:
            return  # cannot proceed

        # Assemble grid order (ny, nx, 2) in pixel coords
        grid_uv = np.empty((self.ny, self.nx, 2), dtype=np.float32)
        for k in range(centers.shape[0]):
            i, j = int(idx_ij[k, 0]), int(idx_ij[k, 1])
            if 0 <= i < self.nx and 0 <= j < self.ny:
                grid_uv[j, i, :] = centers[k, :]

        # Fit affine pixels <-> grid
        grid_ij = np.stack(np.meshgrid(np.arange(self.nx, dtype=np.float32),
                                       np.arange(self.ny, dtype=np.float32), indexing="xy"), axis=-1)  # (ny,nx,2)
        T, T_inv = _fit_affine_grid_to_pixels(grid_ij.reshape(-1, 2), grid_uv.reshape(-1, 2))
        self.rest_grid_uv = grid_uv
        self.T, self.T_inv = T, T_inv

        # Debug display
        if self.cfg.display.show_input:
            overlay = first_bgr.copy()
            for j in range(self.ny):
                for i in range(self.nx):
                    u, v = int(round(grid_uv[j, i, 0])), int(round(grid_uv[j, i, 1]))
                    cv2.circle(overlay, (u, v), 1, (0, 0, 255), -1)
            self.dbg_disp.show("gs_rest_markers", overlay)

    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        """
        Per-frame:
          - Detect dots & re-index (no temporal dependence).
          - Compute grid flow (def - rest) at nodes in pixel units.
          - Interpolate to dense vy, vx.
          - Return Deformation(kind='2d').
          - TODO: compute deformation3d.
        """
        self.frame_id += 1

        if self.rest_grid_uv is None or self.T_inv is None:
            print("[GelSightAdapter] Error: rest grid is None.  prepare() must be called before process().")
            return None

        # 1) Detect dots
        gray = _to_gray(bgr)
        centers = _detect_dot_centers(gray, self.dot_thresh, self.min_area, self.max_area)
        if centers.shape[0] != self.nx * self.ny:
            print(f"[GelSightAdapter] Error: detected {centers.shape[0]} dots; expected {self.nx*self.ny}.")
            return None

        # 2) Build grid order (ny, nx, 2)
        idx_ij = _index_grid_from_points(centers, self.nx, self.ny)
        cur_grid_uv = np.empty((self.ny, self.nx, 2), dtype=np.float32)
        for k in range(centers.shape[0]):
            i, j = int(idx_ij[k, 0]), int(idx_ij[k, 1])
            if 0 <= i < self.nx and 0 <= j < self.ny:
                cur_grid_uv[j, i, :] = centers[k, :]

        # 3) Grid deformation (pixels) at nodes: +x right, +y down
        flow_g = (cur_grid_uv - self.rest_grid_uv).astype(np.float32)
        flow_gx = flow_g[..., 0]  # along image x (u)
        flow_gy = flow_g[..., 1]  # along image y (v)

        # 4) Interpolate to dense per-pixel flow
        H, W = gray.shape[:2]
        vy, vx = _bilinear_dense_from_grid(flow_gx, flow_gy, self.T_inv, H, W)

        # Convert to original camera pixel scale
        scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
        if scale not in (0.0, 1.0):
            vx = vx / scale
            vy = vy / scale

        # Debug visualization
        if self.cfg.display.show_flow_color_raw:
            color_bgr = flow_to_color_bgr(vy, vx, self.cfg.flow.vis_flow_max)
            self.dbg_disp.show("gs_flow_color_raw", color_bgr)
            if self.dbg_writers is not None:
                self.dbg_writers.write("gs_flow_color_raw", color_bgr)

        if self.cfg.display.show_flow_quiver_raw:
            quiv_bgr = draw_quiver_bgr(
                vy, vx,
                block=self.cfg.display.quiver_block,
                pool=self.cfg.display.quiver_pool,
                scale=self.cfg.display.quiver_scale,
                thickness=self.cfg.display.quiver_thickness,
                color=self.cfg.display.quiver_color,
                bg=self.cfg.display.quiver_bg,
                min_px=self.cfg.display.quiver_min_px,
                draw_centers=self.cfg.display.quiver_draw_centers,
                center_color=self.cfg.display.quiver_color,
            )
            self.dbg_disp.show("gs_flow_quiver_raw", quiv_bgr)

        # Pack 2D TIR (L=1, C=2 -> [vy, vx])
        data = np.stack([vy, vx], axis=-1).astype(np.float32)[None, ...]
        deform2d = Deformation(
            data=data, kind='2d',
            meta=self._mk_meta(H, W),
            z_of_layer=None, debug=None
        )

        # 3D deformation
        # to be filled by photometric stereo,
        # for just zero field (mm).
        if False:  # flip to True once PS is implemented
            H3, W3 = H, W
            uxyz = np.zeros((1, H3, W3, 3), dtype=np.float32)
            self.deformation3d = Deformation(
                data=uxyz, kind='3d',
                meta=self._mk_meta(H3, W3),
                z_of_layer=None, debug=None
            )
        else:
            self.deformation3d = None
        
        return deform2d
