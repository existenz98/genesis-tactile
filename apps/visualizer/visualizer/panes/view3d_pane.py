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
    """
    PyVista 3D pane that renders a translucent surface colored by pressure (p),
    and downward arrows with vectors (tx, ty, -p).

    - Builds geometry only once on first frame.
    - Thereafter, updates coordinates and scalars in place
    - Camera is auto-framed once; preserved across updates so user can orbit/zoom.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        # error proof
        if not _HAS_PV:
            layout.addWidget(QtWidgets.QLabel("PyVista not available"))
            self.plotter = None
            self._surf_actor = None
            self._glyph_actor = None
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#BABABA")

        # Persistent state
        self._initialized = False
        self._H = 0
        self._W = 0
        self._cell_mm = 1.0

        # Appearance settings
        self._scale_z = 10.0            # surface vertical exaggeration
        self._z_weight = 1.0            # de-emphasize z in arrow orientation
        self._glyph_factor = 40        # arrow size factor relative to cell size
        self._p_clim = (0, 0.1)         # tuple (vmin, vmax) in MPa; None => auto (percentiles)
        self._show_scalar_bar = True
        

        # Mesh & actors
        self._surf = None           # pv.StructuredGrid
        self._surf_actor = None     # vtkActor
        self._centers = None        # (N,3) base XY, z=0, Fortran-ordered
        self._glyph_actor = None    # vtkActor

    # -------- public knobs --------
    def set_params(self, glyph_factor: float | None = None, z_weight: float | None = None):
        if glyph_factor is not None:
            self._glyph_factor = float(glyph_factor)
        if z_weight is not None:
            self._z_weight = float(z_weight)

    def set_color_range(self, vmin: float | None, vmax: float | None):
        """
        Set fixed color limits for p.
        Pass None,None to return to auto.
        """
        self._p_clim = None if (vmin is None or vmax is None) else (float(vmin), float(vmax))
        # apply the new range, only if surface already exists
        if self._surf_actor is not None:
            try:
                self._surf_actor.mapper.SetScalarRange(*self._p_clim)
                lut = getattr(self._surf_actor.mapper, "lookup_table", None)
                if lut is not None:
                    lut.SetRange(*self._p_clim)
                # update scalar bar
                try:
                    self.plotter.update_scalar_bar_range(self._p_clim)
                except Exception:
                    pass
            except Exception:
                pass

    def reset(self):
        """Clear the scene; next update will re-initialize."""
        if not _HAS_PV:
            return
        self.plotter.clear()
        self._initialized = False
        self._surf = None
        self._surf_actor = None
        self._glyph_actor = None
        self._centers = None


    def update_scene(self, p: np.ndarray, tx: np.ndarray, ty: np.ndarray, cell_mm: float = 1.0):
        """
        Update the 3D scene with new force maps.

        Parameters
        ----------
        p : (H, W) float32
            Pressure map (MPa), used both as color scalar and Z warp.
        tx, ty : (H, W) float32
            Tangential shear components.
        cell_mm : float
            Physical cell size of the coarse grid.
        """

        if not _HAS_PV:
            return

        # create scene only once
        H, W = p.shape
        if not self._initialized:
            self._init_scene(H, W, cell_mm, p, tx, ty)
            return
        
        # If resolution or cell size changed, rebuild.
        if (H != self._H) or (W != self._W) or (abs(cell_mm - self._cell_mm) > 1e-9):
            self.reset()
            self._init_scene(H, W, cell_mm, p, tx, ty)
            return
        
        # --- Update surface geometry (Z) ---
        Z = (p.astype(np.float32) * self._scale_z).ravel(order="F")
        pts = self._surf.points
        pts[:, 2] = Z                           # update only Z column
        self._surf.points = pts                 # trigger VTK pipeline update
        self._surf.point_data["p"] = p.astype(np.float32).ravel(order="F")  # point scalars (color by pressure)

        # surface color
        if self._p_clim is not None:
            try:
                self._surf_actor.mapper.SetScalarRange(*self._p_clim)
                lut = getattr(self._surf_actor.mapper, "lookup_table", None)
                if lut is not None:
                    lut.SetRange(*self._p_clim)
                try:
                    self.plotter.update_scalar_bar_range(self._p_clim)
                except Exception:
                    pass
            except Exception:
                pass

        # Let pyvista push updates without re-adding actor
        try:
            self.plotter.update_coordinates(self._surf.points, render=False, mesh=self._surf)
            self.plotter.update_scalars(self._surf.point_data["p"], render=False, mesh=self._surf, scalars="p")
        except Exception:
            print("[View3DPane] Error updating surface points")
            pass

        # --- Update glyph arrows without removing the actor ---
        self._update_arrows(p, tx, ty)

        # Render once
        self.plotter.render()


    # ---------------------- internals -------------------------

    def _init_scene(self, H: int, W: int, cell_mm: float, p: np.ndarray, tx: np.ndarray, ty: np.ndarray):
        """
        Create grid, actors, and frame the camera.
        Called once on first frame.
        """

        self._H, self._W, self._cell_mm = H, W, float(cell_mm)

        xs = np.arange(W, dtype=np.float32) * self._cell_mm
        ys = np.arange(H, dtype=np.float32) * self._cell_mm
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        # Contiguous arrays for StructuredGrid, requires Fortran style data (col major)
        Xf = np.asfortranarray(X)
        Yf = np.asfortranarray(Y)
        Zf = np.asfortranarray(np.zeros_like(p, dtype=np.float32))

        # Build surface grid; color by p
        self._surf = pv.StructuredGrid(Xf, Yf, Zf)
        self._surf.point_data["p"] = p.astype(np.float32).ravel(order="F")  # to rol major

        # Surface plot color set range
        if self._p_clim is None:
            # one-time auto (robust) if user hasn't set manual limits yet
            vmin = float(np.percentile(p, 2.0))
            vmax = float(np.percentile(p, 98.0))
            if vmax <= vmin + 1e-9: vmax = vmin + 1.0
            clim = (vmin, vmax)
        else:
            clim = self._p_clim

        self._surf_actor = self.plotter.add_mesh(
            self._surf, scalars="p", cmap="turbo", opacity=0.6,
            show_edges=False, clim=clim, scalar_bar_args={"n_labels": 4}
        )

        # Precompute centers (z=0 baseline)
        self._centers = np.c_[
            Xf.ravel(order="F"),
            Yf.ravel(order="F"),
            np.zeros(H * W, dtype=np.float32),
        ]

        # Create initial arrows
        self._create_arrows_actor(p, tx, ty)

        # Camera only once
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self._initialized = True

    def _create_arrows_actor(self, p: np.ndarray, tx: np.ndarray, ty: np.ndarray):
        """
        Build glyph arrows actor once.
        """
        cloud = pv.PolyData(self._centers.copy())
        vec = np.c_[
            tx.astype(np.float32).ravel(order="F"),
            ty.astype(np.float32).ravel(order="F"),
            ((self._z_weight) * p.astype(np.float32)).ravel(order="F"),
        ]
        vec, mag = self._vec_and_mag(p, tx, ty, mode="3d")  # to unit vector + magnitude
        cloud["vec"] = vec
        cloud["mag"] = mag
        arrows = cloud.glyph(orient="vec", scale="mag", factor=self._cell_mm * self._glyph_factor, clamping=False)
        self._glyph_actor = self.plotter.add_mesh(arrows, color="#41719C")

    def _update_arrows(self, p: np.ndarray, tx: np.ndarray, ty: np.ndarray):
        """
        Update glyph arrows dataset without removing the actor.
        (Fallback to re-add if needed)
        """

        cloud = pv.PolyData(self._centers)  # centers unchanged
        vec = np.c_[
            tx.astype(np.float32).ravel(order="F"),
            ty.astype(np.float32).ravel(order="F"),
            ((self._z_weight) * p.astype(np.float32)).ravel(order="F"),
        ]
        vec, mag = self._vec_and_mag(p, tx, ty, mode="3d")
        cloud["vec"] = vec
        cloud["mag"] = mag
        arrows = cloud.glyph(orient="vec", scale="mag", factor=self._cell_mm * self._glyph_factor, clamping=False)

        try:
            # Update mapper input of the existing actor (no flicker, no camera reset)
            self._glyph_actor.mapper.SetInputData(arrows)   # VTK
            self._glyph_actor.mapper.Update()
        except Exception:
            print("[View3DPane] _update_arrows() update arrows failed, recreate arrows.")
            # Robust fallback: replace actor (still keeps camera)
            if self._glyph_actor is not None:
                self.plotter.remove_actor(self._glyph_actor, reset_camera=False)
            self._glyph_actor = self.plotter.add_mesh(arrows, color="#41719C")


    def _vec_and_mag(self, p, tx, ty, mode: str = "3d"):
        """
        Return 3D orientation vectors and a scalar magnitude for scaling.
        """
        vx = tx.astype(np.float32)
        vy = ty.astype(np.float32)
        vz = (self._z_weight * p.astype(np.float32))

        if mode == "shear":
            mag = np.sqrt(vx**2 + vy**2)                    # |tangential|
        elif mode == "normal":
            mag = np.abs(p.astype(np.float32))              # |normal|
        else:  # "3d" (default)
            mag = np.sqrt(vx**2 + vy**2 + (self._z_weight * p.astype(np.float32))**2)

        vec = np.c_[vx.ravel(order="F"), vy.ravel(order="F"), vz.ravel(order="F")]
        mag = mag.ravel(order="F")
        return vec, mag
