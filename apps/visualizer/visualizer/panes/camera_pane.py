from __future__ import annotations
from ..utils.qt import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

class CameraPane(QtWidgets.QWidget):
    def __init__(self, title="Camera", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        # Do NOT lock aspect yet; do that after first valid image
        self.view = self.glw.addViewBox(lockAspect=False, enableMenu=False)
        self.view.setBackgroundColor((21, 21, 21))
        self.view.invertY(True)  # image coords: origin at top-left

        self.img = pg.ImageItem(axisOrder='row-major')  # (H,W,3) uint8 RGB is ok
        self.view.addItem(self.img)
        layout.addWidget(self.glw)

        # Prime with a tiny placeholder so the ViewBox has finite bounds
        self._primed = False
        ph = np.zeros((10, 10, 3), np.uint8)
        self.img.setImage(ph)
        self.view.setRange(QtCore.QRectF(0, 0, 10, 10), padding=0.0)
        # Keep aspect unlocked for now

    def set_image_rgb(self, rgb: np.ndarray):
        if rgb is None:
            return
        self.img.setImage(rgb, levels=(0, 255))
        if not self._primed:
            h, w = rgb.shape[:2]
            # Set a concrete range once, then lock aspect
            self.view.setRange(QtCore.QRectF(0, 0, w, h), padding=0.0)
            self.view.setAspectLocked(True)
            self._primed = True
