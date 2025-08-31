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
Live 3D visualization (PyVista) for tactile pressure and shear.

Must run this ON THE MAIN THREAD!   otherwise will 'block' or crash.

Let pipeline run on a worker thread and publish physics frames via FrameBus.
The viewer polls the latest frame in a timer callback and updates the scene.

Expected physics frame schema:
{
  'grid': {'cell_px': int, 'cell_mm': float, 'H': int, 'W': int},
  'u_mm': {'ux': HxW float32, 'uy': HxW float32},  # optional
  'p':    HxW float32,
  'tau':  {'tx': HxW float32, 'ty': HxW float32},
}
"""



from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pyvista as pv

from ..config.settings import Vis3DConfig
from ..core.frame_bus import FrameBus


class Vis3DLive:
    def __init__(self, bus: FrameBus, topic: str, cfg: Vis3DConfig, normal_gain: float):
        self.bus = bus
        self.topic = topic
        self.cfg = cfg
        self.normal_gain = max(float(normal_gain), 1e-9)

        # Scene objects
        self._plotter: Optional[pv.Plotter] = None
        self._surface_mesh: Optional[pv.PolyData] = None
        self._surface_actor = None
        self._arrows_actor = None

        self._initialized = False
        self._last_ver: Optional[int] = None

    # ---------- Public API (MAIN THREAD) ----------

    def start_blocking(self) -> None:
        """
        Create the window, wire the timer, and enter the event loop (blocking).
        """
        #print("vis3d - start_blocking()")

        if not self.cfg.enable:
            return

        #print("vis3d - create winodw, add axes")
        pv.global_theme.smooth_shading = True
        pl = pv.Plotter(window_size=(1100, 800), title="3D Force Field")
        self._plotter = pl
        pl.add_axes()
        pl.add_text("Tactile 3D — Pressure (surface) + Traction (arrows)", font_size=12)

        # Non-blocking show; returns immediately but opens an interactive window
        pl.show(interactive_update=True, auto_close=False)

        # Manual main-thread loop: process UI events + render + apply our updates
        dt_sec = max(self.cfg.update_ms, 1) / 1000.0
        while pl.render_window is not None:      # window still open
            self._update_once()                  # pull latest frame and update actors
            pl.update()                          # process events + render
            # NOTE: avoid time.sleep here if you see sluggish UI; pl.update() already pumps events


    # ---------- Internals (MAIN THREAD only) ----------

    def _update_once(self) -> None:
        #print("vis3d - update_once()")

        #print("vis3d - pull_latest >>>")
        ver, data = self.bus.pull_latest(self.topic)
        #print("vis3d - pull_latest <<<")

        if ver is None or data is None or ver == self._last_ver:
            return
        self._last_ver = ver

        grid = data["grid"]
        Hc, Wc = int(grid["H"]), int(grid["W"])
        cell_mm = float(grid["cell_mm"])

        p  = np.asarray(data["p"], dtype=np.float32).reshape(Hc, Wc)
        tx = np.asarray(data["tau"]["tx"], dtype=np.float32).reshape(Hc, Wc)
        ty = np.asarray(data["tau"]["ty"], dtype=np.float32).reshape(Hc, Wc)

        # Build plane once
        if not self._initialized:
            width_mm  = Wc * cell_mm
            height_mm = Hc * cell_mm
            plane = pv.Plane(
                i_size=width_mm, j_size=height_mm,
                i_resolution=Wc - 1, j_resolution=Hc - 1,
                direction=(0, 0, 1),
                center=(0.0, 0.0, 0.0),
            )
            self._surface_mesh = plane
            self._surface_mesh.point_data["p"] = np.zeros(Hc * Wc, dtype=np.float32)
            self._surface_actor = self._plotter.add_mesh(
                self._surface_mesh,
                scalars="p",
                cmap=self.cfg.colormap,
                opacity=float(self.cfg.surface_opacity),
                show_edges=False,
                clim=self._pressure_range(p),
            )
            self._plotter.camera_position = "iso"
            self._initialized = True

        # Update pressure scalars and color limits
        self._surface_mesh.point_data["p"] = p.reshape(-1)
        self._surface_actor.mapper.scalar_range = self._pressure_range(p)

        # Optional height field: z <- w ≈ p / normal_gain * height_gain
        if self.cfg.show_height:
            w = (p / self.normal_gain) * float(self.cfg.height_gain)
            pts = self._surface_mesh.points.copy()
            pts[:, 2] = w.reshape(-1)
            self._surface_mesh.points = pts

        # 3D traction arrows: t = (tau_x, tau_y, p) with anisotropic scaling
        if self.cfg.arrow_enable:
            stride = max(int(self.cfg.arrow_stride), 1)
            xs = np.linspace(-Wc * cell_mm / 2.0, Wc * cell_mm / 2.0, Wc, dtype=np.float32)
            ys = np.linspace(-Hc * cell_mm / 2.0, Hc * cell_mm / 2.0, Hc, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys)

            if self.cfg.show_height:
                w = (p / self.normal_gain) * float(self.cfg.height_gain)
                Z0 = w + float(self.cfg.lift_z_mm)
            else:
                Z0 = np.full_like(p, float(self.cfg.lift_z_mm), dtype=np.float32)

            s_t = self._auto_scale_t(tx, ty, cell_mm) if self.cfg.scale_t_auto else float(self.cfg.scale_t)
            s_n = self._auto_scale_n(p,  cell_mm)     if self.cfg.scale_n_auto else float(self.cfg.scale_n)
            Vx = s_t * tx
            Vy = s_t * ty
            Vz = s_n * p

            mag = np.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
            min_len_mm = float(self.cfg.arrow_min_len) * cell_mm
            mask = (mag >= min_len_mm)

            mask_stride = np.zeros_like(mask, dtype=bool)
            mask_stride[::stride, ::stride] = True
            mask &= mask_stride

            pts = np.column_stack([X[mask].ravel(), Y[mask].ravel(), Z0[mask].ravel()])
            vec = np.column_stack([Vx[mask].ravel(), Vy[mask].ravel(), Vz[mask].ravel()])

            # Recreate the arrows actor each update (safe on UI thread and fast for modest grids)
            if self._arrows_actor is not None:
                try:
                    self._plotter.remove_actor(self._arrows_actor)
                except Exception:
                    pass
                self._arrows_actor = None

            if len(pts) > 0:
                self._arrows_actor = self._plotter.add_arrows(pts, vec, mag=1.0, opacity=1.0, lighting=True)

        self._plotter.render()

    # ---------- Helpers ----------

    def _pressure_range(self, p: np.ndarray) -> tuple[float, float]:
        if (self.cfg.p_vmin is not None) and (self.cfg.p_vmax is not None):
            vmin, vmax = float(self.cfg.p_vmin), float(self.cfg.p_vmax)
            if vmax <= vmin:
                vmax = vmin + 1.0
            return (vmin, vmax)
        lo = float(np.percentile(p, 2.0))
        hi = float(np.percentile(p, 98.0))
        if hi <= lo + 1e-12:
            hi = lo + 1.0
        return (lo, hi)

    def _auto_scale_t(self, tx: np.ndarray, ty: np.ndarray, cell_mm: float) -> float:
        tau = np.sqrt(tx*tx + ty*ty)
        s = float(np.percentile(tau, 95.0))
        if s < 1e-9: s = 1.0
        return 0.8 * cell_mm / s

    def _auto_scale_n(self, p: np.ndarray, cell_mm: float) -> float:
        s = float(np.percentile(p, 95.0))
        if s < 1e-9: s = 1.0
        return 0.5 * cell_mm / s
