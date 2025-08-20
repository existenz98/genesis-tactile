from __future__ import annotations
from ..utils.qt import QtWidgets
import pyqtgraph as pg
import numpy as np

class CameraPane(QtWidgets.QWidget):
    def __init__(self, title="Camera", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.label = pg.LabelItem(title, color=(220,220,220))
        self.glw = pg.GraphicsLayoutWidget()
        self.view = self.glw.addViewBox()
        self.view.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.view.addItem(self.img)
        layout.addWidget(self.glw)

    def set_image_rgb(self, rgb: np.ndarray):
        if rgb is None: return
        # pyqtgraph expects (H, W, 3) uint8 in RGB
        self.img.setImage(rgb, levels=(0,255))
        self.view.autoRange(padding=0.02)
