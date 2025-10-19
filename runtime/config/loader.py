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


from dataclasses import is_dataclass, fields
from enum import Enum
from typing import Any, Dict
import yaml

from .settings import (
    RuntimeConfig, CameraConfig, VirtualSourceConfig, PreprocConfig, UnmixConfig,
    FlowConfig, OutputConfig, DisplayConfig,
    SourceMode, CompensationMode, UnmixMode, FlowMethod
)


def _coerce_field_value(field_def, current_value, new_value):
    """Coerce new_value to the type of field_def."""
    ftype = field_def.type

    # Try Optional[...] / Union[...] wrappers
    origin = getattr(ftype, "__origin__", None)
    args = getattr(ftype, "__args__", ())

    def _enum_from(t, v):
        return t(v) if isinstance(t, type) and issubclass(t, Enum) and isinstance(v, str) else None

    # Try to coerce enum first
    if origin is not None and args:
        for t in args:
            coerced = _enum_from(t, new_value)
            if coerced is not None:
                return coerced

    # Direct enum coercion
    if isinstance(ftype, type) and issubclass(ftype, Enum) and isinstance(new_value, str):
        return ftype(new_value)

    # If the target field is a Tuple[...] and YAML gave a list, coerce to tuple
    # NOTE: if current_value is None, we can still coerce by checking the annotation
    ftype_str = str(ftype)
    if (isinstance(current_value, tuple) or "typing.Tuple" in ftype_str or "tuple" == ftype_str.lower()) and isinstance(new_value, list):
        return tuple(new_value)
    
    return new_value


def _merge_dict_into_dataclass(dc_obj, data: Dict):
    """
    Generic deep update into a dataclass instance from a dict.
    """
    if not is_dataclass(dc_obj):
        raise TypeError("dc_obj must be a dataclass instance")

    known = {f.name: f for f in fields(dc_obj)}
    for name, v in data.items():
        if name not in known:
            # Unknown key; ignore or log here if desired
            print(f"[config] Error: Unknown key ignored: '{name}'")
            continue
        f = known[name]
        cur = getattr(dc_obj, name)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge_dict_into_dataclass(cur, v)
        else:
            v = _coerce_field_value(f, cur, v)
            old_v = getattr(dc_obj, name)
            setattr(dc_obj, name, v)
            print(f"[config yaml loader] Set '{dc_obj.__class__.__name__}'.'{name}' from '{old_v}' to '{v}'")


def load_yaml_config(path: str) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = RuntimeConfig()                 # start from defaults
    print(f"[config yaml loader] Applying config from '{path}'")
    _merge_dict_into_dataclass(cfg, data) # merge YAML on top
    return cfg



