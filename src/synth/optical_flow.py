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
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np
import cv2

class FlowMethod(str, Enum):
    FARNEBACK = "farneback"
    DIS       = "dis"
    TVL1      = "tvl1"

@dataclass
class FlowConfig:
    method: FlowMethod = FlowMethod.FARNEBACK

# ---- helpers ----
def to_gray_f32_bgr(img_bgr_uint8: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY)
    return (g.astype(np.float32) / 255.0)

def _as_u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        return (np.clip(gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return gray

def _as_f32(gray: np.ndarray) -> np.ndarray:
    return gray.astype(np.float32, copy=False)

def compute_flow(ref_gray: np.ndarray, cur_gray: np.ndarray, cfg: FlowConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow from ref_gray (I0) -> cur_gray (I1).
    Returns (vx, vy) in float32 (OpenCV convention: flow[...,0], flow[...,1]).
    """
    m = cfg.method

    if m == FlowMethod.FARNEBACK:
        ref = _as_f32(ref_gray); cur = _as_f32(cur_gray)
        flow = cv2.calcOpticalFlowFarneback(ref, cur, None,
                                            pyr_scale=0.5, levels=4, winsize=15, iterations=5,
                                            poly_n=7, poly_sigma=1.5, flags=0)
        vx = flow[..., 0].astype(np.float32); vy = flow[..., 1].astype(np.float32)
        return vx, vy

    if m == FlowMethod.DIS:
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DIS"):
            dis = cv2.optflow.createOptFlow_DIS(getattr(cv2.optflow, "DISOPTICAL_FLOW_PRESET_MEDIUM", 2))
        elif hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", 2))
        else:
            raise RuntimeError("DIS not available. Install opencv-contrib-python.")
        ref = _as_u8(ref_gray); cur = _as_u8(cur_gray)
        flow = dis.calc(ref, cur, None)
        vx = flow[..., 0].astype(np.float32); vy = flow[..., 1].astype(np.float32)
        return vx, vy

    if m == FlowMethod.TVL1:
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        elif hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        else:
            raise RuntimeError("TV-L1 not available. Install opencv-contrib-python.")
        ref = _as_u8(ref_gray); cur = _as_u8(cur_gray)
        flow = tvl1.calc(ref, cur, None)
        vx = flow[..., 0].astype(np.float32); vy = flow[..., 1].astype(np.float32)
        return vx, vy

    raise ValueError(f"Unknown flow method: {cfg.method}")

# ---- visualization (HSV color wheel) ----
def flow_to_bgr(vx: np.ndarray, vy: np.ndarray, clip_mag: float | None = None) -> np.ndarray:
    """
    Convert flow (vx, vy) to BGR image using the classic color wheel.
    clip_mag: if set, clamp magnitude to this for visualization.
    """
    mag, ang = cv2.cartToPolar(vx, vy, angleInDegrees=False)
    if clip_mag is not None:
        mag = np.clip(mag, 0, float(clip_mag))
    # HSV: H=angle, S=1, V=normalized magnitude
    h = (ang / (2*np.pi)) * 179.0  # OpenCV hue in [0,179]
    s = np.ones_like(h) * 255.0
    v = (mag / (mag.max() + 1e-12) * 255.0)
    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
