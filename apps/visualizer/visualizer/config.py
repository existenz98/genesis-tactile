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
    flow_scale: float = 1.0
    flow_minlen: float = 0.2
    force_quiver_scale: float = 6.0
    force_colormap: str = "turbo"  # visual only
    update_hz: float = 30.0

@dataclass
class AppConfig:
    info: AppInfo = AppInfo()
    sdk: SDKConfig = SDKConfig()
    view: ViewParams = ViewParams()
