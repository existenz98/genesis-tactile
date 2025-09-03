# -----------------------------------------------------------------------------
# SPDX-License-Identifier: LicenseRef-YF-Research-NC-1.0
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# Licensed for academic research and non-commercial use only.
# Any Commercial Use (including production use or any use for commercial
# advantage) requires a separate written license from the copyright holder.
# See LICENSE-SRC-RESEARCH-NC for definitions and terms.
#
# Patent Notice:
#   No patent license is granted or implied. Users are responsible for
#   third-party patent clearance.
#
# Citation:
#   Please cite associated publications when available.
# -----------------------------------------------------------------------------


"""
iFEM from full 3D displacement:
- Read mesh from XDMF, rebuild the vector CG1 space, and load u (DOFs) from sidecar NPZ.
- Compute stress sigma(u) with layered material (E, nu).
- Project sigma to a CG1 tensor field S (mass projection in the volume; stable & smooth).
- Sample traction t = S * n on the top surface at a user-chosen regular (Ny, Nx) grid.
- Save a NPZ with (X_mm, Y_mm, tx_mpa, ty_mpa, tz_mpa) for direct comparison/plotting.

Optionally: drop the normal displacement (u_z := 0) before computing stress,
to study identifiability when depth motion is unobserved.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
from mpi4py import MPI

from dolfinx import fem as dxfem
from dolfinx import io as dxio
from dolfinx import mesh as dxmesh
import dolfinx.geometry as dxgeo
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


# ---------------------------
# Materials
# ---------------------------

@dataclass
class MaterialLayers:
    """
    Layered Young's modulus (MPa) over z, with a global Poisson's ratio nu.

    layers: tuple of (z_min_mm, z_max_mm, E_MPa)
    """
    nu: float
    layers: Tuple[Tuple[float, float, float], ...]


def lame_from_Enu(E: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lambda, mu) from E (MPa) and nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def make_layered_lam_mu(mesh: dxmesh.Mesh, mat: MaterialLayers) -> Tuple[dxfem.Function, dxfem.Function]:
    """
    Build DG0 cell-wise constants for (lambda, mu) in MPa based on z-layer ranges.
    """
    Q = dxfem.functionspace(mesh, ("DG", 0))
    lam_fun = dxfem.Function(Q, name="lambda")
    mu_fun  = dxfem.Function(Q, name="mu")

    zlayers = np.array([[z0, z1, E] for (z0, z1, E) in mat.layers], dtype=float)

    def E_of_x(x):
        z = x[2]
        E = np.zeros_like(z)
        for z0, z1, Ei in zlayers:
            mask = (z >= z0) & (z <= z1)
            E[mask] = Ei
        return E

    E_fun = dxfem.Function(Q, name="E")
    E_fun.interpolate(E_of_x)
    lam_arr, mu_arr = lame_from_Enu(E_fun.x.array, mat.nu)
    lam_fun.x.array[:] = lam_arr
    mu_fun.x.array[:]  = mu_arr
    return lam_fun, mu_fun


# ---------------------------
# Utilities
# ---------------------------

def _mesh_extents(mesh: dxmesh.Mesh) -> Tuple[float, float, float]:
    """Return (Lx, Ly, Lz) extents (mm) from mesh coordinates."""
    X = mesh.geometry.x
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    L = maxs - mins
    return float(L[0]), float(L[1]), float(L[2])


def _top_z(mesh: dxmesh.Mesh) -> float:
    """Return top z coordinate (mm)."""
    X = mesh.geometry.x
    return float(X[:, 2].max())


def _build_top_grid(Lx: float, Ly: float, Nx: int, Ny: int, z_top: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular (Ny, Nx) grid on z = z_top plane (nudged slightly inside the domain).
    Returns points array P of shape (N, 3).
    """
    xs = np.linspace(0.0, Lx, Nx)
    ys = np.linspace(0.0, Ly, Ny)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = np.full_like(Xg, z_top - 1e-9) # very close to top surface
    #Zg = np.full_like(Xg, z_top - 5e-2) # 0.05 mm inside
    P = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
    return Xg, Yg, P


# ---------------------------
# Core: traction from displacement
# ---------------------------

@dataclass
class IFEMOptions:
    drop_uz: bool = False               # simulate "camera can't see z" by forcing u_z=0
    #ksp_type: str = "preonly"
    ksp_type: str = "cg"                # fast
    #pc_type: str = "lu"
    pc_type: str = "jacobi"             # fast
    ksp_rtol: float = 1e-10
    ksp_atol: float = 1e-12
    ksp_monitor: bool = True
    ksp_max_it: int = 200               # quick stop


def traction_from_u(
    mesh_xdmf: str,
    dofs_npz: str,
    mat: MaterialLayers,
    Nx: int,
    Ny: int,
    opts: IFEMOptions = IFEMOptions(),
) -> Dict[str, np.ndarray]:
    """
    Compute traction t = sigma n on the top surface from a full 3D displacement field u.

    Returns a dict with:
      X_mm, Y_mm: (Ny, Nx)
      tx_mpa, ty_mpa, tz_mpa: (Ny, Nx)
      meta: dict
    """
    comm = MPI.COMM_WORLD

    # --- Read mesh
    print("load xdmf")
    with dxio.XDMFFile(comm, mesh_xdmf, "r") as xf:
        mesh = xf.read_mesh()

    gdim = mesh.geometry.dim
    assert gdim == 3, "This code expects a 3D mesh."

    # --- Rebuild V and load u from DOFs
    print("load dofs")
    V = dxfem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u = dxfem.Function(V, name="u")
    d = np.load(dofs_npz)
    u.x.array[:] = d["dofs"]  # direct assign: same mesh & space

    # --- Optionally drop u_z
    ux, uy, uz = ufl.split(u)
    if opts.drop_uz:
        u_eff = ufl.as_vector((ux, uy, 0*uz))
    else:
        u_eff = u

    # --- Material
    lam_fun, mu_fun = make_layered_lam_mu(mesh, mat)
    I = ufl.Identity(gdim)

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    sigma_expr = lam_fun * ufl.tr(epsilon(u_eff)) * I + 2.0 * mu_fun * epsilon(u_eff)

    # --- Project sigma to a tensor CG1 for stable point evaluation
    T = dxfem.functionspace(mesh, ("Lagrange", 1, (gdim, gdim)))
    #T = dxfem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
    S = dxfem.Function(T, name="S")  # will hold sigma(u)

    S_trial = ufl.TrialFunction(T)
    R_test  = ufl.TestFunction(T)
    a_ufl = ufl.inner(S_trial, R_test) * ufl.dx
    L_ufl = ufl.inner(sigma_expr, R_test) * ufl.dx

    a = dxfem.form(a_ufl)
    L = dxfem.form(L_ufl)
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)

    print("kps create()")
    print(opts)
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType(opts.ksp_type)
    pc = ksp.getPC()
    pc.setType(opts.pc_type)
    #ksp.setTolerances(rtol=opts.ksp_rtol, atol=opts.ksp_atol)
    ksp.setTolerances(rtol=opts.ksp_rtol, atol=opts.ksp_atol, max_it=opts.ksp_max_it)
    if opts.ksp_monitor:
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"[iFEM KSP] it={its}, ||r||={rnorm:.3e}"))
    ksp.setFromOptions()
    
    print(">>> kps.solve()")
    ksp.solve(b, S.x.petsc_vec)
    print("<<<")

    S.x.scatter_forward()

    # --- Build a regular grid on the top and evaluate traction there
    Lx, Ly, Lz = _mesh_extents(mesh)
    z_top = _top_z(mesh)
    Xg, Yg, P = _build_top_grid(Lx, Ly, Nx, Ny, z_top)

    # Find cells for all P at once
    bb = dxgeo.bb_tree(mesh, mesh.topology.dim, padding=0.0)
    cand = dxgeo.compute_collisions_points(bb, P)
    coll = dxgeo.compute_colliding_cells(mesh, cand, P)
    offs, arr = coll.offsets, coll.array
    cells = np.full(P.shape[0], -1, dtype=np.int32)
    for i in range(P.shape[0]):
        s, e = offs[i], offs[i+1]
        if e > s:
            cells[i] = arr[s]
    mask = cells >= 0

    # Evaluate S (3x3) and compute traction t = S n with n = (0,0,1)
    out = np.zeros((P.shape[0], 3), dtype=float)
    if np.any(mask):
        vals = S.eval(P[mask], cells[mask])  # shape: (M, 9)
        # vals[i] = [S11,S12,S13,S21,S22,S23,S31,S32,S33] in row-major
        # traction with n=(0,0,1): t = S * n = column 3 -> [S13, S23, S33]
        tx = vals[:, 2]
        ty = vals[:, 5]
        tz = vals[:, 8]
        out[mask, 0] = tx
        out[mask, 1] = ty
        out[mask, 2] = tz

    tx = out[:, 0].reshape(Ny, Nx)
    ty = out[:, 1].reshape(Ny, Nx)
    tz = out[:, 2].reshape(Ny, Nx)

    meta = dict(
        mesh_xdmf=mesh_xdmf,
        dofs_npz=dofs_npz,
        drop_uz=opts.drop_uz,
        Lx_mm=Lx, Ly_mm=Ly, Lz_mm=Lz,
        Nx=Nx, Ny=Ny,
        note="Traction field sampled on z=top surface; units MPa (N/mm^2)."
    )

    return dict(
        X_mm=Xg, Y_mm=Yg,
        tx_mpa=tx, ty_mpa=ty, tz_mpa=tz,
        meta=meta
    )
