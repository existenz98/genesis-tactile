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


"""
Branding bar (big logo text)
"""


from __future__ import annotations
from ..utils.qt import QtCore, QtGui, QtWidgets

class BrandBar(QtWidgets.QWidget):
    """Top bar with large gradient logo + version/build + connection status dot."""
    def __init__(self, title_left: str = "Genesis", title_right: str = ".touch",
                 version: str = "v0.0.0", build: str = "-", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self._title_left = title_left
        self._title_right = title_right
        self._version = version
        self._build = build
        self._status = "Disconnected"  # or Connected
        self._status_color = QtGui.QColor(200, 80, 80)

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor(34,34,38))
        self.setPalette(pal)

    def set_status(self, text: str, color: QtGui.QColor):
        self._status = text
        self._status_color = color
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # Left brand text
        rect = QtCore.QRect(16, 6, self.width()//2, self.height()-12)
        f = QtGui.QFont()
        f.setPointSize(24); f.setBold(True)
        p.setFont(f)

        grad1 = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad1.setColorAt(0.0, QtGui.QColor(255, 160, 80))
        grad1.setColorAt(1.0, QtGui.QColor(240, 80, 60))
        pen = QtGui.QPen(QtGui.QBrush(grad1), 1)
        p.setPen(pen)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, self._title_left + " ")

        # Right part of brand
        rect2 = QtCore.QRect(rect.left() + p.fontMetrics().horizontalAdvance(self._title_left + " "),
                              rect.top(), rect.width(), rect.height())
        grad2 = QtGui.QLinearGradient(rect2.topLeft(), rect2.bottomLeft())
        grad2.setColorAt(0.0, QtGui.QColor(255, 230, 120))
        grad2.setColorAt(1.0, QtGui.QColor(180, 220, 120))
        p.setPen(QtGui.QPen(QtGui.QBrush(grad2), 1))
        p.drawText(rect2, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._title_right)

        # Version / build / status right side
        small = QtGui.QFont()
        small.setPointSize(10)
        p.setFont(small)
        text = f"{self._version}  ({self._build})"
        tw = p.fontMetrics().horizontalAdvance(text)
        right = self.width() - 16
        p.setPen(QtGui.QColor(180,180,185))
        p.drawText(right - tw, rect.bottom(), text)

        # Status dot + label
        dot_r = 6
        cx = right - tw - 18 - dot_r
        cy = rect.center().y()
        p.setBrush(self._status_color)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPoint(cx, cy), dot_r, dot_r)
        p.setPen(QtGui.QColor(200,200,205))
        p.drawText(cx - 120, rect.bottom(), f"{self._status}")
