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


import numpy as np
import cv2
from typing import Iterator

def flow_to_color_bgr(vy: np.ndarray, vx: np.ndarray, max_flow: float = None) -> np.ndarray:
    mag = np.sqrt(vy**2 + vx**2)
    ang = np.arctan2(vy, vx)            # image coords: +vy is down
    if max_flow is None:
        max_flow = np.percentile(mag, 95)
        if max_flow < 1e-6: max_flow = 1e-6
    H = (ang + np.pi) / (2*np.pi)       # [0,1)
    S = np.ones_like(H, dtype=np.float32)
    V = np.clip(mag / max_flow, 0, 1)
    hsv = np.stack([H*179.0, S*255.0, V*255.0], axis=-1).astype(np.uint8)  # OpenCV HSV scales
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def scalar_to_color_bgr(
    field: np.ndarray,
    vmin: float = None,
    vmax: float = None,
    colormap: int = cv2.COLORMAP_TURBO
    ) -> np.ndarray:
    """
    Normalize a scalar field to [0,255] and apply a colormap.
    """
    f = field.astype(np.float32)
    if vmin is None:
        vmin = np.percentile(f, 2.0)
    if vmax is None:
        vmax = np.percentile(f, 98.0)
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1.0
    g = np.clip((f - vmin) / (vmax - vmin), 0.0, 1.0)
    u8 = (g * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(u8, colormap)

def draw_quiver_grid_bgr(
    vy_grid: np.ndarray,
    vx_grid: np.ndarray,
    cell_px: int,
    out_H: int,
    out_W: int,
    scale: float = 1.0,
    thickness: int = 1,
    color=(255,255,255),
    bg: str = "black",
    min_len: float = 0.6
    ) -> np.ndarray:
    """
    Draw a quiver given a coarse grid (no extra pooling).
    vy_grid, vx_grid shape: (Hc, Wc).
    cell_px: stride in pixels between neighboring grid nodes on the full-res canvas.
    """
    img = np.full((out_H, out_W, 3), 255, np.uint8) if bg == "white" else np.zeros((out_H, out_W, 3), np.uint8)
    Hc, Wc = vy_grid.shape
    ys = (np.arange(Hc) * cell_px + 0.5 * cell_px).astype(np.float32)
    xs = (np.arange(Wc) * cell_px + 0.5 * cell_px).astype(np.float32)
    for iy, cy in enumerate(ys):
        if cy >= out_H: continue
        for ix, cx in enumerate(xs):
            if cx >= out_W: continue
            dy = float(vy_grid[iy, ix]) * scale
            dx = float(vx_grid[iy, ix]) * scale
            if (dx*dx + dy*dy) < (min_len * min_len):
                continue
            x0, y0 = int(round(cx)), int(round(cy))
            x1 = int(round(cx + dx))
            y1 = int(round(cy + dy))  # image coords (down = +)
            cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.35)
    return img

def flow_block_reduce(vy: np.ndarray, vx: np.ndarray, block: int = 16, pool: str = "median"):
    """
    Spatial pooling of flow vectors on block tiles; returns vy_ds, vx_ds, block.
    """

    H, W = vy.shape
    h = H // block
    w = W // block
    if h == 0 or w == 0:
        return vy, vx, block
    vy_c = vy[:h*block, :w*block].reshape(h, block, w, block).transpose(0, 2, 1, 3)
    vx_c = vx[:h*block, :w*block].reshape(h, block, w, block).transpose(0, 2, 1, 3)
    if pool == "mean":
        vy_ds = vy_c.mean(axis=(2, 3))
        vx_ds = vx_c.mean(axis=(2, 3))
    else:
        vy_ds = np.median(vy_c, axis=(2, 3))
        vx_ds = np.median(vx_c, axis=(2, 3))
    return vy_ds.astype(np.float32), vx_ds.astype(np.float32), block

def draw_quiver_bgr(
    vy: np.ndarray, vx: np.ndarray, block: int = 16, pool: str = "median",
    scale: float = 1.0, thickness: int = 1, color=(255, 255, 255), bg: str = "black",
    min_px: float = 0.6,
    draw_centers: bool = False,
    center_color=(80, 80, 80),
    center_radius: int = 1,
    ) -> np.ndarray:
    """
    Draw quiver using block-pooled vectors.
    - Skips arrows whose magnitude < min_px (prevents dot lattice).
    """
    H, W = vy.shape
    vy_ds, vx_ds, blk = flow_block_reduce(vy, vx, block=block, pool=pool)
    img = np.full((H, W, 3), 255, np.uint8) if bg == "white" else np.zeros((H, W, 3), np.uint8)

    ys = (np.arange(vy_ds.shape[0]) * blk + blk * 0.5).astype(np.float32)
    xs = (np.arange(vx_ds.shape[1]) * blk + blk * 0.5).astype(np.float32)

    if draw_centers:
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                cv2.circle(img, (int(round(cx)), int(round(cy))), center_radius, center_color, -1, lineType=cv2.LINE_AA)

    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            dy = float(vy_ds[iy, ix]) * scale
            dx = float(vx_ds[iy, ix]) * scale
            if (dx*dx + dy*dy) < (min_px * min_px):
                continue  # skip tiny vectors to avoid "dot" artifacts
            x0, y0 = int(round(cx)), int(round(cy))
            x1 = int(round(cx + dx))
            y1 = int(round(cy + dy))            # image coords: +dy draws downward
            cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.35)
    return img


class VideoWriters:
    def __init__(self, out_dir: str, fps: float):
        self.out_dir = out_dir
        self.fps = fps
        self.writers = {}

    def _open(self, key: str, frame_shape):
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        h,w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.out_dir, f"{key}.mp4")
        wr = cv2.VideoWriter(path, fourcc, self.fps, (w,h))
        self.writers[key] = (wr, path)
        return wr

    def write(self, key: str, frame_bgr: np.ndarray):
        if key not in self.writers:
            self._open(key, frame_bgr.shape)
        self.writers[key][0].write(frame_bgr)

    def close(self):
        for wr, _ in self.writers.values():
            wr.release()

    def paths(self):
        return {k: p for k, (_, p) in self.writers.items()}
