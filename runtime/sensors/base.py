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
SensorAdapter base interface
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import time

from ..tir.types import TirMeta, Deformation

class SensorAdapter:
    """
    abstract interface for sensor-specific processing of raw images into TIR (Tactile Intermediate Representation) data for solver to use.
    
    prepare() will be called once with the first (downscaled) frame, then process() will be called per-frame.
    """

    def __init__(self, cfg, dbg_disp=None, dbg_writers=None):
        self.cfg = cfg
        self.dbg_disp = dbg_disp
        self.dbg_writers = dbg_writers
        self.frame_id = 0

    def _mk_meta(self, H: int, W: int) -> TirMeta:
        return TirMeta(
            H=H, W=W,
            mm_per_px=float(self.cfg.physics.mm_per_px),
            downscale=float(self.cfg.downscale),
            frame_id=self.frame_id,
            timestamp_usec=int(time.time() * 1e6),
        )

    def prepare(self, first_bgr: np.ndarray) -> None:
        """
        perform one-time fitting
        e.g. compensation baseline, unmix fitting, reference frames.
        """
        raise NotImplementedError

    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        """
        Return TIR Deformation for the given frame.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional: reset internal references (e.g., perform one time fitting again).
        """
        self.frame_id = 0
