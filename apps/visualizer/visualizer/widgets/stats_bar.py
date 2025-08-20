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