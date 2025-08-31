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


import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image

from ..config.settings import PreprocConfig

def bgr_to_rgb(img_bgr_uint8: np.ndarray) -> np.ndarray:
    return img_bgr_uint8[..., ::-1].copy()

def rgb_to_bgr(img_rgb_uint8: np.ndarray) -> np.ndarray:
    return img_rgb_uint8[..., ::-1].copy()

def srgb_to_linear_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    img = img_rgb_uint8.astype(np.float32) / 255.0
    a = 0.055
    return np.where(img <= 0.04045, img/12.92, ((img + a)/(1+a))**2.4)

def linear_to_srgb_uint8(img_lin: np.ndarray) -> np.ndarray:
    a = 0.055
    out = np.where(img_lin <= 0.0031308, 12.92*img_lin, (1+a)*np.power(img_lin, 1/2.4) - a)
    return (np.clip(out, 0, 1)*255.0).astype(np.uint8)

def compute_highlight_mask_rgb(img_rgb_uint8: np.ndarray, sat_thr: float, val_thr: float) -> np.ndarray:
    img = img_rgb_uint8.astype(np.float32) / 255.0
    r,g,b = img[...,0], img[...,1], img[...,2]
    maxc = np.max(img, axis=-1); minc = np.min(img, axis=-1)
    S = np.zeros_like(maxc, dtype=np.float32)
    nz = maxc > 1e-6
    S[nz] = (maxc[nz]-minc[nz]) / maxc[nz]
    V = maxc
    near_sat = (r>0.985) | (g>0.985) | (b>0.985)
    return ((S < sat_thr) & (V > val_thr)) | near_sat

def _robust_patch_median(patch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = patch[valid] if valid is not None else patch.reshape(-1,3)
    if vals.size == 0: vals = patch.reshape(-1,3)
    return np.median(vals, axis=0)

def _smooth2d(arr: np.ndarray, k: int, iters: int) -> np.ndarray:
    ker = np.ones((k,k), dtype=np.float32) / (k*k)
    out = arr.copy()
    pad = k//2
    for _ in range(iters):
        padded = np.pad(out, ((pad,pad),(pad,pad)), mode='edge')
        H,W = out.shape
        acc = np.zeros_like(out, dtype=np.float64)
        for dy in range(k):
            for dx in range(k):
                acc += ker[dy,dx] * padded[dy:dy+H, dx:dx+W]
        out = acc.astype(np.float32)
    return out

def _upsample(field: np.ndarray, H: int, W: int) -> np.ndarray:
    im = Image.fromarray(field.astype(np.float32), mode='F')
    im2 = im.resize((W,H), resample=Image.Resampling.BICUBIC)
    return np.array(im2, dtype=np.float32)

@dataclass
class CompensationFields:
    s: np.ndarray
    wR: np.ndarray
    wG: np.ndarray
    wB: np.ndarray

class BaselineCompensator:
    """Estimate spatial brightness and per-channel WB fields from first frame; reuse for subsequent frames."""
    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg
        self.fields: Optional[CompensationFields] = None

    def fit(self, first_frame_bgr: np.ndarray) -> CompensationFields:
        cfg = self.cfg
        rgb = bgr_to_rgb(first_frame_bgr)
        H,W,_ = rgb.shape
        hl = compute_highlight_mask_rgb(rgb, cfg.sat_thr, cfg.val_thr)
        lin = srgb_to_linear_uint8(rgb)
        patch = cfg.patch
        gh = (H + patch - 1)//patch
        gw = (W + patch - 1)//patch
        mu = np.zeros((gh,gw,3), dtype=np.float64)
        valid_ratio = np.zeros((gh,gw), dtype=np.float32)
        for iy in range(gh):
            for ix in range(gw):
                y0=iy*patch; y1=min((iy+1)*patch,H)
                x0=ix*patch; x1=min((ix+1)*patch,W)
                p = lin[y0:y1, x0:x1, :]
                v = ~hl[y0:y1, x0:x1]
                valid_ratio[iy,ix] = float(v.mean())
                mu[iy,ix] = _robust_patch_median(p, v)
        L = mu.sum(axis=-1) + 1e-8
        c = mu / L[...,None]
        reliable = valid_ratio > 0.40
        if not np.any(reliable):
            reliable = np.ones_like(reliable, dtype=bool)
        L_ref = np.median(L[reliable])
        c_ref = np.median(c[reliable], axis=0)
        s_grid = np.clip((L_ref / L).astype(np.float32), cfg.clip_min, cfg.clip_max)
        w_grid = np.clip((c_ref[None,None,:] / (c + 1e-8)).astype(np.float32), cfg.clip_min, cfg.clip_max)
        s_grid = _smooth2d(s_grid, cfg.smooth_kernel, cfg.smooth_iters)
        wR_grid = _smooth2d(w_grid[...,0], cfg.smooth_kernel, cfg.smooth_iters)
        wG_grid = _smooth2d(w_grid[...,1], cfg.smooth_kernel, cfg.smooth_iters)
        wB_grid = _smooth2d(w_grid[...,2], cfg.smooth_kernel, cfg.smooth_iters)
        s = _upsample(s_grid, H, W)
        wR = _upsample(wR_grid, H, W)
        wG = _upsample(wG_grid, H, W)
        wB = _upsample(wB_grid, H, W)
        self.fields = CompensationFields(s=s, wR=wR, wG=wG, wB=wB)
        return self.fields

    def apply(self, frame_bgr: np.ndarray, keep_highlights=True) -> np.ndarray:
        assert self.fields is not None, "call fit() first"
        cfg = self.cfg
        rgb = bgr_to_rgb(frame_bgr)
        hl = compute_highlight_mask_rgb(rgb, cfg.sat_thr, cfg.val_thr)
        lin = srgb_to_linear_uint8(rgb)
        f = self.fields
        comp = np.stack([f.s*f.wR*lin[...,0],
                         f.s*f.wG*lin[...,1],
                         f.s*f.wB*lin[...,2]], axis=-1)
        if keep_highlights:
            comp[hl] = lin[hl]
        rgb_corr = linear_to_srgb_uint8(np.clip(comp,0,1))
        return rgb_to_bgr(rgb_corr)

class PerFrameCompensator:
    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg
    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        base = BaselineCompensator(self.cfg)
        base.fit(frame_bgr)
        return base.apply(frame_bgr, keep_highlights=True)
