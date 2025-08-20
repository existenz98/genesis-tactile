from __future__ import annotations
from ..utils.qt import QtWidgets
import numpy as np

# Try to embed PyVista. If not available, show a placeholder.
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
    _HAS_PV = True
except Exception:
    _HAS_PV = False

class View3DPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        if not _HAS_PV:
            layout.addWidget(QtWidgets.QLabel("PyVista not available"))
            self.plotter = None
            self._surf_actor = None
            self._glyph_actor = None
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("black", top="black")

        self._surf_actor = None
        self._glyph_actor = None
        self._scale_z = 1.0  # visual exaggeration

    def reset(self):
        if not _HAS_PV: return
        self.plotter.clear()
        self._surf_actor = None
        self._glyph_actor = None

    def update_scene(self, p: np.ndarray, tx: np.ndarray, ty: np.ndarray, cell_mm: float = 1.0):
        if not _HAS_PV: return
        H, W = p.shape
        xs = np.arange(W, dtype=np.float32) * float(cell_mm)
        ys = np.arange(H, dtype=np.float32) * float(cell_mm)
        X, Y = np.meshgrid(xs, ys)
        Z = p.astype(np.float32) * self._scale_z  # downward arrows are drawn separately

        # Rebuild lightweight geometry (small grids are fine to rebuild per frame)
        surf = pv.StructuredGrid(X, Y, Z)
        surf["p"] = p.ravel(order="C")
        if self._surf_actor is None:
            self._surf_actor = self.plotter.add_mesh(surf, scalars="p",
                                                     cmap="turbo", opacity=0.6, show_edges=False)
            self.plotter.view_isometric()
        else:
            # Replace actor
            self.plotter.remove_actor(self._surf_actor, reset_camera=False)
            self._surf_actor = self.plotter.add_mesh(surf, scalars="p",
                                                     cmap="turbo", opacity=0.6, show_edges=False)

        # Arrows: vectors = (tx, ty, -p)
        pts = np.c_[X.ravel(), Y.ravel(), (0*Z).ravel()]
        vec = np.c_[tx.ravel(), ty.ravel(), (-p).ravel()]
        cloud = pv.PolyData(pts)
        cloud["vec"] = vec
        arrows = cloud.glyph(orient="vec", scale=False, factor=float(cell_mm)*0.6)
        if self._glyph_actor is None:
            self._glyph_actor = self.plotter.add_mesh(arrows, color="yellow")
        else:
            self.plotter.remove_actor(self._glyph_actor, reset_camera=False)
            self._glyph_actor = self.plotter.add_mesh(arrows, color="yellow")
