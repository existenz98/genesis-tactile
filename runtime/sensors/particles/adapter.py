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
ParticlesAdapter:
Single dense 2D deformation field from observation of random particles in gel layer.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np

from ...config.settings import RuntimeConfig, CompensationMode
from ...output.display import DebugDisplay
from ...output.visualizer import flow_to_color_bgr, draw_quiver_bgr, flow_block_reduce, scalar_to_color_bgr, draw_quiver_grid_bgr, VideoWriters
from ...preprocessing.photo_compensator import BaselineCompensator, PerFrameCompensator
from ...preprocessing.optical_flow import compute_flow, to_gray_f32_bgr
from ...tir.types import Deformation
from ..base import SensorAdapter


class ParticlesAdapter(SensorAdapter):
    """
    single-layer gel random particle's CV pre-processing
      - optional photometric compensation (baseline/per-frame/skip)
      - gray conversion
      - dense optical flow vs. first reference frame
      - returns a single-layer Deformation
    Notes:
      - Expects input frames already downscaled by cfg.downscale.
      - Scales the resulting flow back to original camera pixel units by dividing by cfg.downscale.
    """

    def __init__(self, cfg: RuntimeConfig, dbg_disp: Optional[DebugDisplay] = None, dbg_writers: Optional[VideoWriters] = None):
        super().__init__(cfg, dbg_disp)
        self.compensator: Optional[object] = None
        self.ref_bal_bgr: Optional[np.ndarray] = None
        self.ref_raw_gray: Optional[np.ndarray] = None

        # Build image local brightness compensator
        if cfg.compensation_mode == CompensationMode.BASELINE:
            print("[ParticlesAdapter] Initializing BaselineCompensator.")
            self.compensator = BaselineCompensator(cfg.preproc)
        elif cfg.compensation_mode == CompensationMode.PER_FRAME:
            print("[ParticlesAdapter] Initializing PerFrameCompensator.")
            self.compensator = PerFrameCompensator(cfg.preproc)
        else:
            print("[ParticlesAdapter] No compensation will be applied.")
            self.compensator = None

    def prepare(self, first_bgr: np.ndarray) -> None:
        print("[ParticlesAdapter] prepare() called.")

        # Fit baseline once if needed
        bal_bgr = first_bgr
        if isinstance(self.compensator, BaselineCompensator):
            print("[ParticlesAdapter] Fitting baseline compensator.")
            self.compensator.fit(first_bgr)
            bal_bgr = self.compensator.apply(first_bgr)
        elif isinstance(self.compensator, PerFrameCompensator):
            print("[ParticlesAdapter] Applying per-frame compensator on first frame.")
            bal_bgr = self.compensator.apply(first_bgr)

        print("[ParticlesAdapter] Storing reference balanced BGR and grayscale images.")
        self.ref_bal_bgr = bal_bgr.copy()
        self.ref_raw_gray = to_gray_f32_bgr(self.ref_bal_bgr)

    def process(self, bgr: np.ndarray) -> Deformation:
        self.frame_id += 1

        print("[ParticlesAdapter] process() called for frame id", self.frame_id)

        # 1) Compensation
        if self.compensator is not None:
            bal_bgr = self.compensator.apply(bgr)
            # Debug display
            if self.cfg.display.show_compensated:
                self.dbg_disp.show("compensated", bal_bgr)
        else:
            bal_bgr = bgr

        # 2) Dense flow on downscaled grid (pixels @ downscaled)

        # Gray images
        cur_gray = to_gray_f32_bgr(bal_bgr)
        ref_gray = self.ref_raw_gray

        # Compute dense flow (pixels in current downscaled grid)
        vy, vx = compute_flow(ref_gray, cur_gray, self.cfg.flow)

        # Debug display
        print(f"[ParticlesAdapter] show_flow_color_raw: {self.cfg.display.show_flow_color_raw}")
        if self.cfg.display.show_flow_color_raw:
            color_bgr = flow_to_color_bgr(vy, vx, self.cfg.flow.vis_flow_max)
            self.dbg_disp.show("flow_color_raw", color_bgr)
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
            self.dbg_disp.show("flow_quiver_raw", quiv_bgr)


        # 3) Convert to original camera pixel scale to match solvers and SHM's need.
        scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
        if scale != 0 and scale != 1.0:
            vx = vx / scale
            vy = vy / scale

        # 4) Pack into Deformation (L=1, C=2 -> [vy, vx] in pixel units)
        H, W = vy.shape
        data = np.stack([vy, vx], axis=-1).astype(np.float32)       # (H,W,2)
        data = data[None, ...]                                      # (1,H,W,2)

        deform = Deformation(
            data=data,
            kind='2d',
            meta=self._mk_meta(H, W),
            z_of_layer=None,
            debug=None
        )
        return deform
