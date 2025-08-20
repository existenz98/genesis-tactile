from dataclasses import dataclass

@dataclass
class AppInfo:
    name: str = "Genesis.touch"
    version: str = "v0.9.0"
    build: str = "2025-08-20"

@dataclass
class SDKConfig:
    notify_ep: str = "ipc:///tmp/tacto6d.frame"
    ctrl_ep: str = "ipc:///tmp/tacto6d.ctrl"

@dataclass
class ViewParams:
    flow_stride: int = 16
    flow_scale: float = 8.0
    flow_minlen: float = 0.2
    force_quiver_scale: float = 6.0
    force_colormap: str = "turbo"  # visual only
    update_hz: float = 30.0

@dataclass
class AppConfig:
    info: AppInfo = AppInfo()
    sdk: SDKConfig = SDKConfig()
    view: ViewParams = ViewParams()
