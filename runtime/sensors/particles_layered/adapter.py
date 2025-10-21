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
ParticlesLayeredAdapter: 

For multilayer gels with different (colored) particles per layer.
performs color unmix + 3 per-color flows (R,G,B)
"""


from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from ...config.settings import RuntimeConfig, CompensationMode, UnmixMode
from ...output.display import DebugDisplay
from ...output.visualizer import flow_to_color_bgr, draw_quiver_bgr, flow_block_reduce, scalar_to_color_bgr, draw_quiver_grid_bgr, VideoWriters
from ...preprocessing.photo_compensator import BaselineCompensator, PerFrameCompensator
from ...preprocessing.feature_extraction import UnmixModel
from ...preprocessing.optical_flow import compute_flow, to_gray_f32_bgr, to_gray_f32_linear_component
from ...tir.types import Deformation
from ..base import SensorAdapter

class ParticlesLayeredAdapter(SensorAdapter):
    """
    Multilayer particles:
      - optional photometric compensation (baseline/per-frame/skip)
      - unmix r, g, b particles.
      - per-layer optical flows (dense 2D deformation) vs reference frame
      - outputs 3-layer Deformation
    """

    def __init__(self, cfg: RuntimeConfig, dbg_disp: Optional[DebugDisplay] = None, dbg_writers: Optional[VideoWriters] = None):
        super().__init__(cfg, dbg_disp, dbg_writers)
        self.compensator: Optional[object] = None
        self.unmix: Optional[UnmixModel] = None

        self.ref_bal_bgr: Optional[np.ndarray] = None
        self.ref_comp_lin: Optional[np.ndarray] = None  # HxWx3 linear components at reference

        # Compensation
        if cfg.compensation_mode == CompensationMode.BASELINE:
            self.compensator = BaselineCompensator(cfg.preproc)
        elif cfg.compensation_mode == CompensationMode.PER_FRAME:
            self.compensator = PerFrameCompensator(cfg.preproc)
        else:
            self.compensator = None

        # Unmix r, g, b particles
        print(f"[ParticlesLayeredAdapter] cfg.unmix.mode = '{cfg.unmix.mode}'")
        if cfg.unmix.mode != UnmixMode.SKIP:
            self.unmix = UnmixModel(cfg.unmix)
        else:
            self.unmix = None

    def prepare(self, first_bgr: np.ndarray) -> None:
        # Compensation fit/apply
        bal_bgr = first_bgr
        if isinstance(self.compensator, BaselineCompensator):
            self.compensator.fit(first_bgr)
            bal_bgr = self.compensator.apply(first_bgr)
        elif isinstance(self.compensator, PerFrameCompensator):
            bal_bgr = self.compensator.apply(first_bgr)

        self.ref_bal_bgr = bal_bgr.copy()

        if self.unmix is None:
            raise ValueError("[ParticlesLayeredAdapter] Error: requires unmixing; got unmix_mode=SKIP.")
        
        # Fit unmix on reference
        self.unmix.fit(self.ref_bal_bgr)
        self.ref_comp_lin = self.unmix.transform(self.ref_bal_bgr)  # HxWx3 (linear components)

    def _compute_layer_flow(self, ref_comp_lin: np.ndarray, cur_comp_lin: np.ndarray
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute per-layer flows for R/G/B linear components.
        """
        vy_layers, vx_layers = [], []
        for k in range(3):  # R,G,B order as in your existing code
            refk = to_gray_f32_linear_component(ref_comp_lin[..., k])
            curk = to_gray_f32_linear_component(cur_comp_lin[..., k])
            vy, vx = compute_flow(refk, curk, self.cfg.flow)
            vy_layers.append(vy)
            vx_layers.append(vx)
        return vy_layers, vx_layers

    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        self.frame_id += 1

        # 1) Apply Compensation, get balanced BGR
        if self.compensator is not None:
            bal_bgr = self.compensator.apply(bgr)
            # Debug display
            if self.cfg.display.show_compensated:
                self.dbg_disp.show("compensated", bal_bgr)
        else:
            bal_bgr = bgr

        # 2) Unmix current frame to linear components of each color (HxWx3)
        if self.unmix is None or self.ref_comp_lin is None:
            raise ValueError("[ParticlesLayeredAdapter] Error: requires unmixing; got unmix_mode=SKIP.")

        cur_comp_lin = self.unmix.transform(bal_bgr)

        # Visualization: segmentation color & component maps
        if self.cfg.display.show_seg_color or self.cfg.display.show_seg_R or self.cfg.display.show_seg_G or self.cfg.display.show_seg_B:
            seg_idx = np.argmax(cur_comp_lin, axis=-1)
            seg_bgr = np.zeros_like(bal_bgr)
            # R,G,B colors in BGR
            palette = np.array([[0,0,255],[0,255,0],[255,0,0]], dtype=np.uint8)
            for k in range(3):
                seg_bgr[seg_idx==k] = palette[k]
            if self.cfg.display.show_seg_color:
                self.dbg_disp.show("seg_rgb", seg_bgr)
            def to_vis(comp):
                g = to_gray_f32_linear_component(comp)
                return (np.stack([g,g,g], axis=-1)*255).astype(np.uint8)
            if self.cfg.display.show_seg_R: self.dbg_disp.show("comp_R", to_vis(cur_comp_lin[...,0]))
            if self.cfg.display.show_seg_G: self.dbg_disp.show("comp_G", to_vis(cur_comp_lin[...,1]))
            if self.cfg.display.show_seg_B: self.dbg_disp.show("comp_B", to_vis(cur_comp_lin[...,2]))

        # 3) Per-layer flows in the downscaled grid
        vy_layers, vx_layers = self._compute_layer_flow(self.ref_comp_lin, cur_comp_lin)

        # 4) Scale flows back to original camera pixel units
        scale = float(self.cfg.downscale) if self.cfg.downscale else 1.0
        if scale != 0 and scale != 1.0:
            vy_layers = [vy / scale for vy in vy_layers]        # flow strength as original pixel scale
            vx_layers = [vx / scale for vx in vx_layers]        # flow strength as original pixel scale

        # debug display per-layer flows
        for k, name in enumerate(["R","G","B"]):
            vx = vx_layers[k]
            vy = vy_layers[k]
            # COLOR window + (optional) writing
            if getattr(self.cfg.display, f"show_flow_color_{name}"):
                color_bgr = flow_to_color_bgr(vy, vx, self.cfg.flow.vis_flow_max)
                self.dbg_disp.show(f"flow_color_{name}", color_bgr)
                if self.dbg_writers is not None:
                    self.dbg_writers.write(f"flow_color_{name}", color_bgr)
            # QUIVER window
            if getattr(self.cfg.display, f"show_flow_quiver_{name}"):
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
                self.dbg_disp.show(f"flow_quiver_{name}", quiv_bgr)

        # 5) Pack layers into (L,H,W,2)
        H, W = vy_layers[0].shape
        layers = [np.stack([vy_layers[i], vx_layers[i]], axis=-1) for i in range(3)]  # each (H,W,2)
        data = np.stack(layers, axis=0).astype(np.float32)                             # (3,H,W,2)

        deform = Deformation(
            data=data,
            kind='2d',
            meta=self._mk_meta(H, W),
            z_of_layer=None,   # TODO: fill layer depths
            debug=None
        )
        return deform
