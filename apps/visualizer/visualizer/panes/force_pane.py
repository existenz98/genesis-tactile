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


from __future__ import annotations
from ..utils.qt import QtWidgets
import numpy as np
import cv2
import pyqtgraph as pg
from ..utils.flow_vis import heatmap_rgb, draw_quiver_bgr

class ForcePane(QtWidgets.QWidget):
    def __init__(self, title="Force", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox()
        self.view.invertY(True)                             # <— keep origin at top-left
        self.view.setAspectLocked(True)
        #self.img = pg.ImageItem()
        self.img = pg.ImageItem(axisOrder='row-major')  # <— avoid W/H swap
        self.view.addItem(self.img)
        layout.addWidget(self.glw)
        self.scale = 6.0  # quiver scale

    def set_params(self, quiver_scale: float = 6.0):
        self.scale = float(quiver_scale)

    def update_force(self, p: np.ndarray, tx: np.ndarray, ty: np.ndarray):
        if p is None or tx is None or ty is None: return
        hm = heatmap_rgb(p)

        # upscale for visibility
        H, W = p.shape
        vis = cv2.resize(cv2.cvtColor(hm, cv2.COLOR_RGB2BGR), (W*16, H*16), interpolation=cv2.INTER_NEAREST)

        # draw shear quiver on coarse grid centers (note image coords: vy ~ ty, vx ~ tx)
        Ys, Xs = np.mgrid[0:H, 0:W]; Ys = Ys*16+8; Xs = Xs*16+8
        vis = draw_quiver_bgr(vis, Ys, Xs, ty, tx, scale=self.scale, min_len=0.0, color=(0,0,0))
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.img.setImage(rgb, levels=(0,255))
        self.view.autoRange(padding=0.02)
