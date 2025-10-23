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
    flow_to_color_bgr, draw_quiver_bgr, draw_quiver_grid_bgr, VideoWriters
)
from ...preprocessing.optical_flow import to_gray_f32_bgr
from ...tir.types import Deformation
from ..base import SensorAdapter

# ---------------------------- small helpers ----------------------------------

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.uint8)


# ---------------------------- Marker Grid ----------------------------------

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


# ---------------------- photometric stereo helper: dot mask + inpainting ---------------------------

def _dot_mask(gray_u8: np.ndarray, thresh: int, dilate: int = 2) -> np.ndarray:
    """Binary mask (uint8 in {0,255}) of marker dots (dilated)."""
    _, bw = cv2.threshold(gray_u8, thresh, 255, cv2.THRESH_BINARY_INV)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
        bw = cv2.dilate(bw, k, iterations=1)
    return bw

def _inpaint_bgr(img_bgr: np.ndarray, mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Per-channel Telea inpainting for masked pixels (fast, good quality)."""
    # OpenCV expects mask 0/255
    return cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# ---------------------- photometric stereo core -------------------------

def _estimate_channel_gains_from_rest(rest_bgr: np.ndarray,
                                      mask: np.ndarray,
                                      dir_r: np.ndarray,
                                      dir_g: np.ndarray,
                                      dir_b: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate channel gains g_r, g_g, g_b
    Should run it on the initial rest (flat) frame.
    Using median intensity outside the dot mask and the known l_c dot n0 (n0=[0,0,1]).

    dir_r, dir_g, dir_b: light directions for each channel (unit vectors).
    """
    # inpaint first to avoid bias near dots
    rest_inp = _inpaint_bgr(rest_bgr, mask, radius=2)

    # use linear 0..1 units, not 0..255
    Ir = (rest_inp[..., 2].astype(np.float32)) / 255.0
    Ig = (rest_inp[..., 1].astype(np.float32)) / 255.0
    Ib = (rest_inp[..., 0].astype(np.float32)) / 255.0

    m_r = float(np.median(Ir[mask == 0]))
    m_g = float(np.median(Ig[mask == 0]))
    m_b = float(np.median(Ib[mask == 0]))

    n0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eps = 1e-6
    g_r = m_r / max(eps, float(np.dot(dir_r, n0)))
    g_g = m_g / max(eps, float(np.dot(dir_g, n0)))
    g_b = m_b / max(eps, float(np.dot(dir_b, n0)))
    return g_r, g_g, g_b


def _solve_normals_from_rgb(img_bgr: np.ndarray,
                            gains_rgb: Tuple[float, float, float],
                            dir_r: np.ndarray,
                            dir_g: np.ndarray,
                            dir_b: np.ndarray,
                            mask: Optional[np.ndarray] = None,
                            pre_smooth_sigma: float = 0.8,
                            post_smooth_sigma: float = 1.5) -> np.ndarray:
    """
    Analytic RGB photometric stereo with dot-aware inpainting.
    - Inputs and gains in linear [0,1] radiometry.
    - Handles N_z<0 by flipping the whole vector.
    - Optional tiny smoothing to remove pixelation.
    """
    H, W = img_bgr.shape[:2]

    # Inpaint to remove dot contamination
    if mask is not None:
        img = _inpaint_bgr(img_bgr, mask, radius=2)
    else:
        img = img_bgr
    
    # ---- linearize to 0..1
    img_lin = img.astype(np.float32) / 255.0

    # ---- pre-smoothing (removes quantization noise)
    if pre_smooth_sigma > 0:
        img_lin = cv2.GaussianBlur(img_lin, ksize=(0, 0),
                                   sigmaX=pre_smooth_sigma,
                                   sigmaY=pre_smooth_sigma,
                                   borderType=cv2.BORDER_REFLECT)


    # Normalize by pre-calibrated gains (e.g. from rest frame, also in 0..1 units)
    Ir = img_lin[..., 2] / float(gains_rgb[0])
    Ig = img_lin[..., 1] / float(gains_rgb[1])
    Ib = img_lin[..., 0] / float(gains_rgb[2])
    I3 = np.stack([Ir, Ig, Ib], axis=-1)  # (H,W,3)

    # Light matrix L. rows are l_r, l_g, l_b
    L = np.stack([dir_r, dir_g, dir_b], axis=0).astype(np.float32)  # (3,3)
    L_inv = np.linalg.inv(L)

    # Solve for unnormalized "normals * albedo" : G = L^{-1} I' ,   given L dot G = I
    # method 1. Explicit matrix multiplication
    G = I3 @ L_inv.T    # (H,W,3)
    # method 2. Using linear solver
    #G = np.linalg.solve(L, I3.reshape(-1, 3).T).T.reshape(I3.shape)

    # Normalize to unit normals
    eps = 1e-8
    N = G / np.maximum(np.linalg.norm(G, axis=-1, keepdims=True), eps)  # N = G / ||G||
    
    # flip entire vector where Nz<0 (do not clamping Nz)
    neg = N[..., 2] < 0.0
    N[neg] = -N[neg]

    # Optional small post-smoothing in normal space, then renormalize
    if post_smooth_sigma > 0:
        for c in range(3):
            N[..., c] = cv2.GaussianBlur(N[..., c], (0, 0), post_smooth_sigma)
        N = N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), eps)

    return N.astype(np.float32)


def _frankot_chellappa(p: np.ndarray, q: np.ndarray, dx_mm: float, dy_mm: float) -> np.ndarray:
    """
    Frankot-Chellappa integration
    Basic idea is to integrate gradients in frequency domain.
    Inputs:
    - p = dz/dx
    - q = dz/dy
    - anisotropic grid spacing (dx_mm, dy_mm).
    Returns z in millimeters (up to an additive constant).
    """
    H, W = p.shape

    # Frequency domain grids with physical spacing, not pixel spacing.
    wx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx_mm).astype(np.float32)  # (W,)
    wy = 2.0 * np.pi * np.fft.fftfreq(H, d=dy_mm).astype(np.float32)  # (H,)
    WX, WY = np.meshgrid(wx, wy, indexing="xy")

    # FFT of p, q
    P = np.fft.fft2(p); Q = np.fft.fft2(q)

    # denominator is |w|^2
    denom = (WX**2 + WY**2).astype(np.float32)
    denom[0, 0] = 1.0

    # Integrate in frequency domain
    Z = (-1j * WX * P - 1j * WY * Q) / denom

    # remove DC component
    Z[0, 0] = 0.0

    # Inverse FFT to get z
    z = np.real(np.fft.ifft2(Z)).astype(np.float32)

    return z


#---- Visualization helpers ----
def normals_to_rgb_bgr(normals: np.ndarray,
                       gain=(10.0, 10.0, 1.0),   # (gx, gy, gz)
                       gamma: float = 0.8) -> np.ndarray:
    """
    Color visualization of unit normals as RGB.
    """
    n = normals.astype(np.float32)
    # ensure unit length
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    # apply per-axis gain
    n[..., 0] *= float(gain[0])   # boosts R (Nx)
    n[..., 1] *= float(gain[1])   # boosts G (Ny)
    n[..., 2] *= float(gain[2])   # boosts B (Nz)
    # clamp back to valid range
    n = np.clip(n, -1.0, 1.0)
    # map [-1,1] -> [0,1]
    rgb = 0.5 * (n + 1.0)
    # gamma for punchier midtones
    if gamma != 1.0:
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / float(gamma))
    # to uint8 BGR for OpenCV
    rgb8 = (rgb * 255.0).astype(np.uint8)
    bgr8 = rgb8[..., [2, 1, 0]]
    return bgr8

def hillshade_from_normals_bgr(normals: np.ndarray,
                               light_dir=(0.4, 0.4, 0.82),
                               ambient=0.35,
                               diffuse=0.85,
                               spec=0.0,
                               shininess=16,
                               clip_percent=0.5,
                               mask: np.ndarray | None = None) -> np.ndarray:
    """
    Grayscale shading image from unit normals.
    normals: (H,W,3) float32 in [-1,1], assumed unit (will re-normalize just in case)
    Returns BGR uint8 image for OpenCV display.
    """
    n = normals.astype(np.float32)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)

    l = np.asarray(light_dir, np.float32)
    l /= (np.linalg.norm(l) + 1e-8)

    ndotl = np.clip((n * l).sum(-1), 0.0, 1.0)  # (H,W)
    I = ambient + diffuse * ndotl

    if spec > 0.0:
        v = np.array([0, 0, 1], np.float32)              # view toward +z
        r = 2.0 * ndotl[..., None] * n - l               # reflect(l,n)
        spec_term = np.clip((r * v).sum(-1), 0.0, 1.0) ** shininess
        I = I + spec * spec_term

    if mask is not None:
        # Optional: suppress shading where normals are invalid (e.g., masked dots)
        I = np.where(mask.astype(bool), I, I)

    # gentle contrast normalization
    lo, hi = np.percentile(I, [clip_percent, 100.0 - clip_percent])
    if hi > lo:
        I = (I - lo) / (hi - lo)
    I = np.clip(I, 0, 1)
    gray = (I * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def depth_to_gray_bgr(z_mm: np.ndarray,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      clip_percent: float = 0.5,
                      invert: bool = False) -> np.ndarray:
    """
    Convert a depth map (mm) to a grayscale BGR image for display.

    Args:
      z_mm: (H,W) float depth, mm.
      vmin, vmax: fixed display range in mm. If None, use robust percentiles.
      clip_percent: when vmin/vmax are None, trim that % from each tail.
      invert: if True, deeper (more positive) is darker.

    Returns:
      BGR uint8 image suitable for DebugDisplay.show().
    """
    z = z_mm.astype(np.float32)

    # Pick display range
    if (vmin is None) or (vmax is None):
        # auto range from robust percentiles
        lo, hi = np.percentile(z, [clip_percent, 100.0 - clip_percent])
        if hi <= lo:
            lo, hi = float(z.min()), float(z.max())
    else:
        lo, hi = float(vmin), float(vmax)
        if hi <= lo:
            hi = lo + 1e-6

    # Normalize to [0,1]
    g = (z - lo) / (hi - lo)
    g = np.clip(g, 0.0, 1.0)

    # Optional invert
    if invert:
        g = 1.0 - g

    gray = (g * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ------------------------------ Adapter --------------------------------------

class PhotometricAdapter(SensorAdapter):
    """
    Photometric tactile sensor (e.g. GelSight-style)'s pre-processing:
      - Detect and index regular dot grid from scratch per frame (no temporal dependence).
      - Build sparse grid displacement (pixels) vs. rest state.
      - Interpolate to a dense (H,W) field and pack as Deformation(kind='2d') for the solver.
      - Photometric stereo normals and depth (mm) with dot-aware inpainting.
      - 3D deformation: [ux(px), uy(px), uz(mm)].   (first two channels is 2D deformation in pixel units)

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
        print("[PhotometricAdapter] Initializing with sensor params:", s)

        # Marker grid params:        
        self.nx: int = int(getattr(s, "nx", 40))
        self.ny: int = int(getattr(s, "ny", 30))
        self.dot_thresh: int = int(getattr(s, "dot_thresh", 40))
        self.min_area: int = int(getattr(s, "min_area", 5))
        self.max_area: int = int(getattr(s, "max_area", 60))

        # Photometric stereo params:
        L = getattr(s, "lighting", None)
        # Light direction
        self.dir_r = np.array(getattr(L, "dir_r", [ 0.577,  0.000, 0.816]), dtype=np.float32)
        self.dir_g = np.array(getattr(L, "dir_g", [-0.289,  0.500, 0.816]), dtype=np.float32)
        self.dir_b = np.array(getattr(L, "dir_b", [-0.289, -0.500, 0.816]), dtype=np.float32)
        # note: for generating synthetic data, setting is in renderer_gelsight.yaml

        self.inpaint_radius: int = int(getattr(s, "inpaint_radius", 2))
        self.enable_depth: bool = bool(getattr(s, "enable_depth", True))

        # Rest-state of marker grid (computed on first frame)
        self.rest_grid_uv: Optional[np.ndarray] = None   # (ny, nx, 2) float32
        self.T: Optional[np.ndarray] = None              # pixels <- grid  (3x3)
        self.T_inv: Optional[np.ndarray] = None          # grid   <- pixels (3x3)

        # Rest-state photometric gains (computed on first frame)
        self.ps_gains_rgb: Optional[Tuple[float, float, float]] = None

        self.rest_z_mm: Optional[np.ndarray] = None

        # 3D deformation
        self.deformation3d: Optional[Deformation] = None

    def prepare(self, first_bgr: np.ndarray) -> None:
        """
        Build rest-state grid
        - Estimate marker grid rest state
          - detect dots
          - index (i,j)
          - fit affine map pixels<->grid.
        - Estimate photometric gains from rest frame.
          - dot mask + inpaint
          - median intensities outside dots
          - compute gains.
        """
        print("[PhotometricAdapter] prepare()")

        # --- 1) Marker grid at rest (for 2D flow) ---

        # Detect dots in first frame
        gray = _to_gray(first_bgr)
        centers = _detect_dot_centers(gray, self.dot_thresh, self.min_area, self.max_area)
        if centers.shape[0] != self.nx * self.ny:
            print(f"[PhotometricAdapter] WARNING: detected {centers.shape[0]} dots; expected {self.nx*self.ny}. "
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
            self.dbg_disp.show("gs rest markers", overlay)

        # --- 2) Photometric stereo calibration (gains) + rest depth ---
        if self.enable_depth:
            # calibrate per color gains
            scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
            inpaint_r = max(1, int(round(self.inpaint_radius / max(scale, 1e-6))))
            dotmask = _dot_mask(gray, self.dot_thresh, dilate=inpaint_r)
            self.ps_gains_rgb = _estimate_channel_gains_from_rest(
                first_bgr, dotmask, self.dir_r, self.dir_g, self.dir_b
            )

            # compute normals at rest
            normals0 = _solve_normals_from_rgb(first_bgr, self.ps_gains_rgb,
                                               self.dir_r, self.dir_g, self.dir_b, mask=dotmask)
            # slopes from normals
            nz = np.maximum(normals0[..., 2], 1e-6)
            p0 = -normals0[..., 0] / nz
            q0 = -normals0[..., 1] / nz

            # integrate to get rest depth
            mm_per_px = float(self.cfg.physics.mm_per_px)
            scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
            mm_per_px_eff = float(self.cfg.physics.mm_per_px) / (scale if scale > 0 else 1.0)
            z0 = _frankot_chellappa(p0, q0, dx_mm=mm_per_px_eff, dy_mm=mm_per_px_eff)

            # Anchor depth: subtract median of 10px border to ~0 outside contact
            border = 10
            border_vals = np.concatenate([
                z0[:border, :].ravel(), z0[-border:, :].ravel(),
                z0[:, :border].ravel(), z0[:, -border:].ravel()
            ])
            z0 = z0 - np.median(border_vals)
            self.rest_z_mm = z0

            if self.cfg.display.show_normal_map:
                Nvis = normals_to_rgb_bgr(normals0)
                self.dbg_disp.show("gs normals rest", Nvis)
            if self.cfg.display.show_depth_map:
                zvis = hillshade_from_normals_bgr(normals0, light_dir=(0.4, 0.4, 0.82))
                self.dbg_disp.show("gs depth rest", zvis)



    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        """
        Per-frame:
          - Marker grid flow (2D deformation in pixel units).
            - Detect dots & re-index (no temporal dependence).
            - Compute grid flow (def - rest) at nodes in pixel units.
            - Interpolate to dense vy, vx.  get Deformation(kind='2d').
          - Photometric stereo depth (3D deformation in mm).
            - dot mask + inpaint
            - solve normals
            - integrate to depth
            - get Deformation(kind='3d') with (ux, uy, uz).
        """
        self.frame_id += 1

        deform2d = None
        deform3d = None


        if self.rest_grid_uv is None or self.T_inv is None:
            print("[PhotometricAdapter] Error: rest grid is None.  prepare() must be called before process().")
            return None

        # 1) Detect dots
        gray = _to_gray(bgr)
        centers = _detect_dot_centers(gray, self.dot_thresh, self.min_area, self.max_area)
        if centers.shape[0] != self.nx * self.ny:
            print(f"[PhotometricAdapter] Error: detected {centers.shape[0]} dots; expected {self.nx*self.ny}.")
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

        # ---- 3D deformation via photometric stereo (uz in mm) ----
        uxyz = None
        if self.enable_depth and self.ps_gains_rgb is not None and self.rest_z_mm is not None:

            # Solve normals
            scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
            inpaint_r = max(1, int(round(self.inpaint_radius / max(scale, 1e-6))))
            dotmask = _dot_mask(gray, self.dot_thresh, dilate=inpaint_r)
            normals = _solve_normals_from_rgb(bgr, self.ps_gains_rgb,
                                              self.dir_r, self.dir_g, self.dir_b, mask=dotmask)
            # Slopes
            nz = np.maximum(normals[..., 2], 1e-6)
            p = -normals[..., 0] / nz
            q = -normals[..., 1] / nz

            # Integrate to get depth
            mm_per_px = float(self.cfg.physics.mm_per_px)
            z_mm = _frankot_chellappa(p, q, dx_mm=mm_per_px, dy_mm=mm_per_px)

            # Anchor z to z estimated from rest frame
            z_mm = z_mm - np.median(z_mm[:10, :].ravel().tolist()
                                    + z_mm[-10:, :].ravel().tolist()
                                    + z_mm[:, :10].ravel().tolist()
                                    + z_mm[:, -10:].ravel().tolist())
            uz_mm = z_mm - self.rest_z_mm  # deformation in mm

            # Debug visualization
            if self.cfg.display.show_depth_map:
                zvis = hillshade_from_normals_bgr(normals, light_dir=(0.4, 0.4, 0.82),
                    ambient=0.35, diffuse=0.85, spec=0.0, shininess=16, clip_percent=0.5)
                self.dbg_disp.show("gs depth (shaded)", zvis)
                gray_bgr = depth_to_gray_bgr(uz_mm, vmin=-0.5, vmax=0.5, invert=True)
                #print(f"[PhotometricAdapter] uz_mm min: {np.min(uz_mm):.4f}, max: {np.max(uz_mm):.4f}, median: {np.median(uz_mm):.4f} mm, rest_z = {np.median(self.rest_z_mm):.4f} mm")
                #gray_bgr = depth_to_gray_bgr(uz_mm, vmin=None, vmax=None, clip_percent=0.5, invert=True)
                self.dbg_disp.show("gs depth (gray)", gray_bgr)

            if self.cfg.display.show_normal_map:
                Nvis = normals_to_rgb_bgr(normals)
                self.dbg_disp.show("gs normals", Nvis)
            
            # Build 3D deformation data [ux(px), uy(px), uz(mm)]
            uxyz = np.stack([vx, vy, uz_mm], axis=-1).astype(np.float32)[None, ...]
            deform3d = Deformation(
                data=uxyz, kind='3d',
                meta=self._mk_meta(H, W),
                z_of_layer=None, debug=None
            )

        if deform3d is not None:
            return deform3d
        else:
            return deform2d

