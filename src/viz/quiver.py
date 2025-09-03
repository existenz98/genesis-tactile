# -----------------------------------------------------------------------------
# SPDX-License-Identifier: LicenseRef-YF-Research-NC-1.0
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# Licensed for academic research and non-commercial use only.
# Any Commercial Use (including production use or any use for commercial
# advantage) requires a separate written license from the copyright holder.
# See LICENSE-SRC-RESEARCH-NC for definitions and terms.
#
# Patent Notice:
#   No patent license is granted or implied. Users are responsible for
#   third-party patent clearance.
#
# Citation:
#   Please cite associated publications when available.
# -----------------------------------------------------------------------------


from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional, Union

def draw_quiver_bgr(
    vy: np.ndarray,
    vx: np.ndarray,
    block: int = 16,
    pool: int = 1,
    scale: float = 4.0,
    thickness: int = 1,
    color: Tuple[int, int, int] = (0, 255, 255),  # BGR (yellow)
    bg: Optional[Union[Tuple[int, int, int], np.ndarray]] = (0, 0, 0),
    min_px: float = 1.5,
    draw_centers: bool = False,
    center_color: Tuple[int, int, int] = (255, 255, 255),
    use_arrow: bool = True,
    tip_length: float = 0.35,  # fraction of the vector length (OpenCV semantics)
) -> np.ndarray:
    """
    Render a dense (vx, vy) flow as a BGR quiver image with arrow heads.

    vy, vx: float32 arrays of shape (H, W) (OpenCV convention: flow[...,1], flow[...,0]).
    block:  stride in pixels between arrow centers.
    pool:   local-average window radius; if >1, average flow in (pool x pool).
    scale:  multiply flow vector length (pixels) for drawing.
    min_px: do not draw arrows whose scaled magnitude is < min_px.
    bg:     tuple(B,G,R) for solid background, or an (H,W,3) uint8 array to draw over.
    """
    assert vy.shape == vx.shape, "vy and vx must have same shape"
    h, w = vy.shape[:2]

    if isinstance(bg, np.ndarray):
        assert bg.shape[:2] == (h, w), "bg image size must match flow"
        canvas = bg.copy()
    elif bg is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        canvas = np.full((h, w, 3), bg, dtype=np.uint8)

    vx = vx.astype(np.float32, copy=False)
    vy = vy.astype(np.float32, copy=False)

    half_blk  = max(1, block // 2)
    half_pool = max(0, (pool // 2))

    for cy in range(half_blk, h, block):
        y0 = max(0, cy - half_pool); y1 = min(h, cy + half_pool + 1)
        for cx in range(half_blk, w, block):
            x0 = max(0, cx - half_pool); x1 = min(w, cx + half_pool + 1)

            vx_loc = float(vx[y0:y1, x0:x1].mean())
            vy_loc = float(vy[y0:y1, x0:x1].mean())

            dx = vx_loc * scale
            dy = vy_loc * scale
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < float(min_px):
                continue

            start = (int(round(cx)), int(round(cy)))
            end   = (int(round(cx + dx)), int(round(cy + dy)))

            if use_arrow:
                # Arrow head length is tip_length * |vector|
                cv2.arrowedLine(canvas, start, end, color,
                                thickness=thickness, line_type=cv2.LINE_AA,
                                tipLength=float(tip_length))
            else:
                cv2.line(canvas, start, end, color,
                         thickness=thickness, lineType=cv2.LINE_AA)

            if draw_centers:
                cv2.circle(canvas, start, 1, center_color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas
