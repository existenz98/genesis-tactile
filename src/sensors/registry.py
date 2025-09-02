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


from __future__ import annotations
from typing import Any, Dict, List, Mapping, Type
from .base import SensorRenderer


# Registry of all sensor renderers
_REGISTRY: Dict[str, Type[SensorRenderer]] = {}


def register_sensor(name: str):
    """
    Class decorator,  to register a SensorRenderer with a string key.
    Usage:
        @register_sensor("particle_vts")
        class ParticleVTSRenderer(SensorRenderer):
            ...
            ...
    """
    def _decorator(cls: Type[SensorRenderer]) -> Type[SensorRenderer]:
        if not issubclass(cls, SensorRenderer):
            raise TypeError(f"{cls.__name__} must subclass SensorRenderer")
        key = str(name).strip().lower()
        if not key:
            raise ValueError("sensor name must be a non-empty string")
        if key in _REGISTRY:
            raise KeyError(f"sensor '{key}' already registered with {_REGISTRY[key]}")
        _REGISTRY[key] = cls
        return cls
    return _decorator


def make_sensor(name: str, **cfg: Any) -> SensorRenderer:
    """
    Instantiate a registered SensorRenderer by name.
    Example:
        renderer = make_sensor("particle_vts", particles={...}, rasterizer={...})
    """
    key = str(name).strip().lower()
    try:
        cls = _REGISTRY[key]
    except KeyError as e:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown sensor '{key}'. Available: {available}") from e
    return cls(cfg)  # type: ignore[arg-type]


def list_sensors() -> List[str]:
    """Return sorted of all registered sensor names."""
    return sorted(_REGISTRY.keys())
