
from dataclasses import is_dataclass, fields
from typing import Any, Dict, Type
import yaml

from .settings import (
    RuntimeConfig, CameraConfig, VirtualSourceConfig, PreprocConfig, UnmixConfig,
    FlowConfig, OutputConfig, DisplayConfig,
    SourceMode, CompensationMode, UnmixMode, FlowMethod
)

# Map field names to their enum types for coercion
_ENUM_FIELDS = {
    ("source_mode",): SourceMode,
    ("compensation_mode",): CompensationMode,
    ("unmix_mode",): UnmixMode,
    ("flow", "method"): FlowMethod,
}

def _enum_coerce(path: tuple, value: Any):
    for key_path, enum_type in _ENUM_FIELDS.items():
        if path == key_path and isinstance(value, str):
            return enum_type(value)
    return value

# Generic deep update into a dataclass instance
def _merge_dict_into_dataclass(dc_obj, d: Dict, path_prefix: tuple = ()):
    if not is_dataclass(dc_obj):
        raise TypeError("dc_obj must be a dataclass instance")
    for f in fields(dc_obj):
        name = f.name
        if name not in d:
            continue
        v = d[name]
        subpath = path_prefix + (name,)
        # enum coercion
        v = _enum_coerce(subpath, v)
        cur = getattr(dc_obj, name)
        # nested dataclass?
        if is_dataclass(cur) and isinstance(v, dict):
            _merge_dict_into_dataclass(cur, v, subpath)
        else:
            # allow list->tuple for things like quiver_color
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(dc_obj, name, v)

def load_yaml_config(path: str) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # start from defaults, then merge YAML
    cfg = RuntimeConfig()
    _merge_dict_into_dataclass(cfg, data)
    return cfg
