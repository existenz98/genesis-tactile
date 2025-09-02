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
Deformation field:
- "none": zero displacement
- "xdmf": read a dolfinx XDMF (mesh + vector Function 'u') and evaluate u(P) at given points
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

# dolfinx is optional; only imported if you use 'xdmf'
try:
    import dolfinx
    from dolfinx import io as dxio
    import dolfinx.geometry as dxgeo
    from mpi4py import MPI
except Exception:
    dolfinx = None


@dataclass
class DeformField:
    mode: str = "none"
    xdmf_path: Optional[str] = None

    # internal fields for xdmf mode
    _mesh: any = None
    _u: any = None
    _bb: any = None

    def load(self):
        if self.mode == "xdmf":
            print("[DeformField] Loading XDMF deformation field from:", self.xdmf_path)

            assert dolfinx is not None, "dolfinx is required for xdmf deformation"
            assert self.xdmf_path is not None, "xdmf_path must be set for xdmf mode"

            with dxio.XDMFFile(MPI.COMM_WORLD, self.xdmf_path, "r") as xf:
                self._mesh = xf.read_mesh()
            # Recreate the vector CG1 space and function
            from dolfinx import fem as dxfem
            V = dxfem.functionspace(self._mesh, ("Lagrange", 1, (self._mesh.geometry.dim,)))
            self._u = dxfem.Function(V, name="u")

            # load DOFs from the sidecar npz
            import numpy as np
            d = np.load(self.xdmf_path.replace(".xdmf", ".dofs.npz"))
            self._u.x.array[:] = d["dofs"]  # assign directly (same DoF ordering on same mesh)
            self._bb = dxgeo.bb_tree(self._mesh, self._mesh.topology.dim, padding=0.0)

    def sample(self, pts_mm: np.ndarray) -> np.ndarray:
        """
        Sample 'displacement u' at 3D points (N,3) in mm. Returns (N,3) in mm.
        """
        if self.mode == "none" or pts_mm.size == 0:
            return np.zeros_like(pts_mm, dtype=float)

        # xdmf mode
        assert self._mesh is not None and self._u is not None and self._bb is not None, "call load() first"
        P = pts_mm.astype(float)
        n = P.shape[0]

        # Find colliding cells for all points
        cand = dxgeo.compute_collisions_points(self._bb, P)
        coll = dxgeo.compute_colliding_cells(self._mesh, cand, P)

        offs, arr = coll.offsets, coll.array
        cells = np.full(n, -1, dtype=np.int32)
        for i in range(n):
            s, e = offs[i], offs[i+1]
            if e > s:
                cells[i] = arr[s]

        ok = cells >= 0
        disp = np.zeros((n, 3), float)
        if np.any(ok):
            disp[ok, :] = self._u.eval(P[ok], cells[ok])
        return disp
