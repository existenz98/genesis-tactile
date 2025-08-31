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
Flow/force visualization utils
"""

from __future__ import annotations
import numpy as np
import cv2

def flow_to_color_rgb(vy: np.ndarray, vx: np.ndarray, max_flow: float | None = None) -> np.ndarray:
    mag = np.sqrt(vy**2 + vx**2)
    ang = np.arctan2(vy, vx)
    if max_flow is None:
        max_flow = max(1e-6, float(np.percentile(mag, 95.0)))
    H = (ang + np.pi) / (2*np.pi)
    S = np.ones_like(H, dtype=np.float32)
    V = np.clip(mag / max_flow, 0.0, 1.0)
    hsv = np.stack([H*179.0, S*255.0, V*255.0], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def downsample_vec(vy: np.ndarray, vx: np.ndarray, block: int = 16):
    H, W = vy.shape
    Hc = H // block; Wc = W // block
    vyc = vy[:Hc*block, :Wc*block].reshape(Hc, block, Wc, block).mean(axis=(1,3))
    vxc = vx[:Hc*block, :Wc*block].reshape(Hc, block, Wc, block).mean(axis=(1,3))
    ys = np.linspace(block/2, H - block/2, Hc)
    xs = np.linspace(block/2, W - block/2, Wc)
    X, Y = np.meshgrid(xs, ys)  # X cols, Y rows
    return Y, X, vyc, vxc

def draw_quiver_bgr(img_bgr: np.ndarray, Y: np.ndarray, X: np.ndarray,
                    vy: np.ndarray, vx: np.ndarray,
                    scale: float = 8.0, min_len: float = 0.2, color=(0,0,0)) -> np.ndarray:
    out = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    mag = np.sqrt(vy**2 + vx**2)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mag[i, j] < min_len:
                continue
            y0 = int(round(Y[i, j])); x0 = int(round(X[i, j]))
            x1 = int(round(x0 + scale * vx[i, j]))
            y1 = int(round(y0 + scale * vy[i, j]))
            if 0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H:
                cv2.arrowedLine(out, (x0, y0), (x1, y1), color, 1, tipLength=0.3)
    return out

def heatmap_rgb(p: np.ndarray, vmin: float | None = None, vmax: float | None = None, cmap: str = "turbo") -> np.ndarray:
    if vmin is None or vmax is None:
        vmin = float(np.percentile(p, 2.0))
        vmax = float(np.percentile(p, 98.0))
        if vmax <= vmin + 1e-6: vmax = vmin + 1.0
    pn = np.clip((p - vmin) / (vmax - vmin), 0.0, 1.0)
    cm = cv2.applyColorMap((pn*255.0).astype(np.uint8), getattr(cv2, f'COLORMAP_{cmap.upper()}', cv2.COLORMAP_TURBO))
    rgb = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return rgb

