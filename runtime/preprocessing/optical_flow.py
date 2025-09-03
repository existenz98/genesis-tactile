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
from typing import Tuple
from ..config.settings import FlowConfig, FlowMethod

def to_gray_f32_bgr(img_bgr_uint8: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY)
    return (g.astype(np.float32) / 255.0)

def to_gray_f32_linear_component(component_lin: np.ndarray) -> np.ndarray:
    # component_lin is already linear [0,inf); normalize robustly
    p = np.percentile(component_lin, 99.0)
    p = max(p, 1e-6)
    return np.clip(component_lin / p, 0, 1).astype(np.float32)

def _as_u8(gray):
    # gray can be float [0,1] or anything else; clamp & scale to 0..255 uint8
    if gray.dtype != np.uint8:
        return (np.clip(gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return gray

def _as_f32(gray):
    return gray.astype(np.float32, copy=False)


# module-global
_OPTICAL_FLOW_DIS_CACHE = {}

def _get_dis(H: int, W: int, fast_preset=False):
    """
    Create the DIS instance once (reuse it every frame)
    because construction + parameter setup costs several ms and dominates at small sizes.
    Keep a cached instance keyed by image size + config.
    """

    key = (H, W, fast_preset)
    dis = _OPTICAL_FLOW_DIS_CACHE.get(key)
    if dis is None:
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DIS"):
            preset = cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST if fast_preset else cv2.optflow.DISOPTICAL_FLOW_PRESET_FAST
            dis = cv2.optflow.createOptFlow_DIS(preset)
        elif hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST if fast_preset else cv2.DISOPTICAL_FLOW_PRESET_FAST
            )
        else:
            raise RuntimeError("DIS not available in this OpenCV build. Install opencv-contrib-python "
                               "or switch flow.method to 'farneback' or 'tvl1'.")

        # Further TUNING for speed:
        #dis.setFinestScale(2)                    # skip full-res stage; 0 = full-res; 1=1/2; 2=1/4
        #dis.setPatchSize(8)
        #dis.setPatchStride(4)
        #dis.setGradientDescentIterations(12)
        #dis.setUseMeanNormalization(False)
        #dis.setUseSpatialPropagation(True)
        #dis.setVariationalRefinementIterations(0)  # disable slow refinement

        _OPTICAL_FLOW_DIS_CACHE[key] = dis
    return dis


def compute_flow(ref_gray: np.ndarray, cur_gray: np.ndarray, cfg: FlowConfig) -> Tuple[np.ndarray, np.ndarray]:
    method = cfg.method

    if method == FlowMethod.FARNEBACK:
        # need float
        ref = _as_f32(ref_gray)
        cur = _as_f32(cur_gray)
        flow = cv2.calcOpticalFlowFarneback(ref, cur, None,
                                            pyr_scale=0.5, levels=4, winsize=15, iterations=5,
                                            poly_n=7, poly_sigma=1.5, flags=0)
        vx = flow[...,0].astype(np.float32); vy = flow[...,1].astype(np.float32)
        return vy, vx

    elif method == FlowMethod.DIS:
        # need u8
        ref = _as_u8(ref_gray)
        cur = _as_u8(cur_gray)

        if False:   # create a new DIS instance every frame (slow)
            if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DIS"):
                dis = cv2.optflow.createOptFlow_DIS(
                    #getattr(cv2.optflow, "DISOPTICAL_FLOW_PRESET_MEDIUM", 2)
                    getattr(cv2.optflow, "DISOPTICAL_FLOW_PRESET_FAST", 2)
                )
            elif hasattr(cv2, "DISOpticalFlow_create"):
                dis = cv2.DISOpticalFlow_create(
                    #getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", 2)
                    getattr(cv2, "DISOPTICAL_FLOW_PRESET_FAST", 2)
                )
            else:
                raise RuntimeError("DIS not available in this OpenCV build. Install opencv-contrib-python "
                                "or switch flow.method to 'farneback' or 'tvl1'.")
        else:       # reuse existing dis object (faster)
            H, W = ref.shape[:2]
            dis = _get_dis(H, W, fast_preset=False)      # speed up: reusing existing dis object

        flow = dis.calc(ref, cur, None)
        vx = flow[...,0].astype(np.float32); vy = flow[...,1].astype(np.float32)
        return vy, vx

    elif method == FlowMethod.TVL1:
        # TV-L1 also has two spellings
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        elif hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        else:
            raise RuntimeError("TV-L1 not available. Install opencv-contrib-python or choose 'farneback'/'dis'.")
        # need u8
        ref = _as_u8(ref_gray); cur = _as_u8(cur_gray)
        flow = tvl1.calc(ref, cur, None)
        vx = flow[...,0].astype(np.float32); vy = flow[...,1].astype(np.float32)
        return vy, vx

    else:
        raise ValueError(f"Unknown flow method: {method}")

