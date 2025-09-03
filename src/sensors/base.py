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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


__all__ = [
    "FEMOutputs",
    "Scene",
    "FrameBundle",
    "TargetBundle",
    "SensorRenderer",
]



@dataclass
class FEMOutputs:
    """
    Container for FEM results needed by sensor renderers.

    Attributes
    ----------
    u_dofs : Any
        FEM nodal displacement DOFs (format is project-specific).
    force_top : Any
        Ground-truth traction/force on the top surface (format is project-specific).
    surface_mesh : Any, optional
        The (possibly deformed) surface mesh or a handle to reconstruct it.
        Provide what your synth pipeline expects (tri mesh, grid, etc.).
    contact_patch : Any, optional
        Optional description of the contact region; leave None if not used.
    meta : Dict[str, Any], optional
        Extra metadata (units, scales, etc.).
    """
    u_dofs: Any
    force_top: Any
    surface_mesh: Optional[Any] = None
    contact_patch: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)



@dataclass
class Scene:
    """
    Scene description needed for rendering.

    Attributes
    ----------
    camera : Any
        Camera model object expected by your synth.camera (e.g., has project()).
    calibration : Dict[str, Any], optional
        Calibration blob (if needed by a sensor).
    rng : Any, optional
        Random number generator for stochastic effects.
    meta : Dict[str, Any], optional
        Extra scene metadata (units, transforms, etc.).
    """
    camera: Any
    calibration: Optional[Dict[str, Any]] = None
    rng: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)



@dataclass
class FrameBundle:
    """
    Outputs produced by a sensor renderer for one frame.

    Attributes
    ----------
    modalities : Dict[str, Any]
        Dict of named observable modalities, e.g.:
        - "image_rgb": np.ndarray HxWx3 uint8
        - "flow_dense": np.ndarray HxWx2 float32
        - "flow_mask": np.ndarray HxW uint8 (optional)
    metadata : Dict[str, Any]
        Free-form metadata (image_size, counts, configs, etc.).
    aux : Dict[str, Any]
        Optional aux artifacts (projected points, intermediate buffers, etc.).
        Not intended for training; useful for debugging/inspection.
    """
    modalities: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)



@dataclass
class TargetBundle:
    """
    Physical ground-truth targets aligned with this frame.

    Attributes
    ----------
    targets : Dict[str, Any]
        e.g., {"u_dofs": ..., "force_top": ...}
    metadata : Dict[str, Any]
        Free-form metadata (units, hashes, etc.).
    """
    targets: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)



class SensorRenderer(ABC):
    """
    Abstract base class for all tactile sensor renderers.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        # Make a shallow copy to avoid external mutation
        self.cfg: Dict[str, Any] = dict(cfg or {})

    # ---- Identification / schema ----
    @abstractmethod
    def name(self) -> str:
        """Stable, lowercase sensor type name (e.g., 'particle_vts')."""
        raise NotImplementedError

    def version(self) -> str:
        """Semantic version for this renderer implementation."""
        return "0.1.0"

    def modalities(self) -> Dict[str, str]:
        """
        Human-readable schema of produced modalities.
        Keys should match FrameBundle.modalities keys.
        """
        return {}

    # ---- Core rendering API ----
    @abstractmethod
    def render_frame(self, fem: FEMOutputs, scene: Scene) -> FrameBundle:
        """
        Produce observable modalities for one frame.
        """
        raise NotImplementedError

    # ---- Targets export (default passthrough) ----
    def export_targets(self, fem: FEMOutputs) -> TargetBundle:
        """
        Default target exporter: passes through u_dofs and force_top if present.
        Sensor-specific subclasses may override this if they define additional targets.
        """
        t: Dict[str, Any] = {}
        if getattr(fem, "u_dofs", None) is not None:
            t["u_dofs"] = fem.u_dofs
        if getattr(fem, "force_top", None) is not None:
            t["force_top"] = fem.force_top
        return TargetBundle(targets=t)
