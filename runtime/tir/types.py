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
TIR (Tactile Intermediate Representation) types

One struct, one array shape: (L, H, W, C).
Layering is just the first dimension (L); 2D vs 3D is just C=2 vs C=3.
Units: pixels for 2D, millimeters for 3D.
Helpers provide averaging and typed views without forcing the pipeline to branch all over.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

@dataclass
class TirMeta:
    """
    Metadata for Tactile Intermediate Representation,
    e.g., image dimensions and scaling.
    """
    H: int  # image height in pixels
    W: int  # image width in pixels
    mm_per_px: float  # millimeters per pixel
    downscale: float  # downscaling factor
    frame_id: int  # frame ID
    timestamp_usec: int  # timestamp in microseconds    
    valid_mask: Optional[np.ndarray] = None # (H,W) boolean, optional
    sensor_frame: str = "S"  # image-right x, image-down y, out-of-sensor z.

@dataclass
class Deformation:
    """
    Generic deformation field.
    Intermediate Representation for dense optical flows / displacements.

    data: (L, H, W, C)
      - C=2  -> 2D image-plane deformation [vy, vx] in *pixels*
      - C=3  -> 3D surface displacement    [ux, uy, uz] in *millimeters*
    L: number of layers (1 for single-layer sensors; >1 for multilayer).
    z_of_layer: optional (L,) depths in millimeters (top surface z=0 by convention).
    kind:
      - '2d' deformation info is only 2D displacements.  Including single layer 2D, or multi-layer 2D
     -  '3d' deformation info is 3D
    """
    data: np.ndarray              # shape (L, H, W, C) float32
    kind: str                     # '2d' | '3d'
    meta: TirMeta
    z_of_layer: Optional[np.ndarray] = None  # (L,)
    debug: Optional[Dict[str, Any]] = None   # optional images/overlays

    # Convenience accessors to data
    def L(self) -> int: return int(self.data.shape[0])  # number of layers
    def H(self) -> int: return int(self.data.shape[1])  # height
    def W(self) -> int: return int(self.data.shape[2])  # width
    def C(self) -> int: return int(self.data.shape[3])  # channels, value should be 2 (2D displacement) or 3 (3D displacement)

    def _average_to_single_layer(self) -> "Deformation":
        """
        Collapse multi layer to 1, by simple mean over layers.
        """
        if self.L() == 1:
            return self
        d = np.nanmean(self.data, axis=0, keepdims=True)
        zl = None if self.z_of_layer is None else np.array([np.nanmean(self.z_of_layer)], dtype=np.float32)
        return Deformation(d, self.kind, self.meta, zl, self.debug)


    # ---------------------------
    # Layer-averaged getters
    # ---------------------------

    def vyvx(self) -> tuple:
        """
        Return (vy, vx) if kind='2d' (L,H,W) each.
        """
        if self.kind != '2d' or self.C() != 2:      
            raise ValueError("[TirDeformation] Error: vyvx() called but kind is not '2d' with C=2.")
        # If multi-layered, return averaged for compatibility
        d = self._average_to_single_layer().data[0]   # (H,W,2)
        return d[...,0], d[...,1]


    def uxuyuz(self) -> tuple:
        """
        Return (ux, uy, uz) if kind='3d', (L,H,W) each.
        """
        if self.kind != '3d' or self.C() != 3: 
            raise ValueError("[TirDeformation] Error: uxuyuz() called but kind is not '3d' with C=3.")
        d = self._average_to_single_layer().data[0]   # (H,W,3)
        return d[...,0], d[...,1], d[...,2]

    # ---------------------------
    # Per-layer typed getters
    # ---------------------------

    def vyvx_layer(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        For kind='2d': return (vy_k, vx_k) views of shape (H,W).
        """
        if self.kind != '2d' or self.C() != 2:
            raise ValueError("[TirDeformation] Error: vyvx_layer() requires kind='2d' with C=2.")
        d = self.data[k, ...]  # (H,W,2) view
        return d[..., 0], d[..., 1]     # (H,W), (H,W)

    def uxuyuz_layer(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For kind='3d': return (ux_k, uy_k, uz_k) views of shape (H,W)."""
        if self.kind != '3d' or self.C() != 3:
            raise ValueError("[TirDeformation] Error: uxuyuz_layer() requires kind='3d' with C=3.")
        d = self.data[k, ...]  # (H,W,3) view
        return d[..., 0], d[..., 1], d[..., 2]      # (H,W), (H,W), (H,W)



def to_flow2d_pixels(deform: Deformation) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Deformation to 2D flow in pixels (vy, vx).
      - If deform.kind == '2d': return averaged (vy, vx) in pixels.
      - If deform.kind == '3d': return (uy, ux) converted to pixels, ignore uz.
        (Note: image coords are y-down, x-right; units mm -> pixels via mm_per_px.)
    """
    if deform.kind == '2d':
        return deform.vyvx()
    elif deform.kind == '3d':
        ux, uy, uz = deform.uxuyuz()
        mpp = deform.meta.mm_per_px
        vy = uy / mpp
        vx = ux / mpp
        return vy.astype(np.float32), vx.astype(np.float32)
    else:
        raise ValueError(f"Unsupported kind: {deform.kind}")
