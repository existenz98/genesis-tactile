from __future__ import annotations
from ..utils.qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import cv2
from ..utils.flow_vis import flow_to_color_rgb, downsample_vec, draw_quiver_bgr

class FlowPane(QtWidgets.QWidget):
    def __init__(self, title="Flow", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox(lockAspect=False, enableMenu=False)
        self.view.setBackgroundColor((21, 21, 21))
        self.view.invertY(True)

        self.img = pg.ImageItem(axisOrder='row-major')
        self.view.addItem(self.img)
        layout.addWidget(self.glw)

        # Default params
        self.block = 16
        self.scale = 8.0
        self.minlen = 0.2

        # Prime with placeholder + finite bounds
        self._primed = False
        ph = np.zeros((10, 10, 3), np.uint8)
        self.img.setImage(ph)
        self.view.setRange(QtCore.QRectF(0, 0, 10, 10), padding=0.0)

    def set_params(self, block=16, scale=8.0, minlen=0.2):
        self.block, self.scale, self.minlen = int(block), float(scale), float(minlen)

    def update_flow(self, vy: np.ndarray, vx: np.ndarray):
        if vy is None or vx is None:
            return

        # Render HSV color + quiver overlay
        flow_rgb = flow_to_color_rgb(vy, vx)
        bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
        Y, X, vyb, vxb = downsample_vec(vy, vx, block=self.block)
        bgr = draw_quiver_bgr(bgr, Y, X, vyb, vxb,
                              scale=self.scale, min_len=self.minlen, color=(0, 0, 0))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.img.setImage(rgb, levels=(0, 255))
        if not self._primed:
            h, w = rgb.shape[:2]
            self.view.setRange(QtCore.QRectF(0, 0, w, h), padding=0.0)
            self.view.setAspectLocked(True)
            self._primed = True
