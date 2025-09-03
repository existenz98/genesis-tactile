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


"""
Status bar

"""

from __future__ import annotations
from ..utils.qt import QtWidgets

class StatsBar(QtWidgets.QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QStatusBar{background:#1f1f22;color:#ddd;}")

    def update_stats(self, **kvs):
        parts = []
        for k, v in kvs.items():
            parts.append(f"{k}: {v}")
        self.showMessage("  |  ".join(parts))