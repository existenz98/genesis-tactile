# -----------------------------------------------------------------------------
# 3D Visualizer — Application Source
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Yue Fei <feiyuefy@gmail.com>
#
# License (Code):
#   This source file is MIT-licensed. See LICENSE-APPS-MIT.
#
# Brand & UI Assets (Not MIT-licensed):
#   Project name, logo(s), icons, UI themes/skins, and other brand assets are
#   NOT licensed under MIT. Use requires a separate license. See:
#     - LICENSE-APPS-ASSETS
#     - TRADEMARKS.md
#
# Patents:
#   Except as may be granted under the applicable open-source license, no patent
#   rights are granted or implied. Users are responsible for third-party patent
#   clearances.
#
# Notes:
#   - Redistribution must retain this header and the referenced license files.
#   - Forks may not imply endorsement or use the original branding without
#     permission (see TRADEMARKS.md).
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
