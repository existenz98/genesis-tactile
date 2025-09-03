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
Forward linear elasticity with layered material and Neumann traction on the top face.

- Material: piecewise (by z) linear elastic, near-incompressible (nu≈0.49).
- BCs: clamp bottom and all four sides (u = 0). Top is free (Neumann).
- RHS: surface traction t(x,y) applied on the top boundary (Vector CG1 function),
       typically created from a (Ny,Nx,3) grid via bilinear interpolation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

from dolfinx import fem as dxfem
from dolfinx import io as dxio
from dolfinx import mesh as dxmesh
from ufl import TrialFunction, TestFunction, Identity, inner, grad, sym, tr, dx, Measure, as_vector

from petsc4py import PETSc
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
)

from .geometry import MeshBundle, FACET_ID


# ---------------------------
# Utilities: Lamé parameters
# ---------------------------

def lame_from_Enu(E: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lamé parameters (lambda, mu) from Young's modulus E and Poisson ratio nu.
    E: array of MPa (can be scalar or cell-wise values)
    Returns (lambda, mu) in MPa.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


# --------------------------------------
# Build piecewise-constant material fields
# --------------------------------------

@dataclass
class MaterialLayers:
    nu: float
    # list of tuples: (z_min_mm, z_max_mm, E_MPa)
    layers: Tuple[Tuple[float, float, float], ...]


def make_layered_material_functions(mesh: dxmesh.Mesh, mat: MaterialLayers) -> Tuple[dxfem.Function, dxfem.Function]:
    """
    Create DG0 (cell-wise constant) Functions for lambda and mu (in MPa),
    according to z-layered Young's modulus.
    """
    Q = dxfem.functionspace(mesh, ("DG", 0))
    lam_fun = dxfem.Function(Q, name="lambda")
    mu_fun = dxfem.Function(Q, name="mu")

    # Interpolate E(z), then map to (lambda, mu)
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

    # Compute Lamé from E, nu
    lam_arr, mu_arr = lame_from_Enu(E_fun.x.array, mat.nu)
    lam_fun.x.array[:] = lam_arr
    mu_fun.x.array[:] = mu_arr
    return lam_fun, mu_fun


# --------------------------------------
# Traction function (from a regular grid)
# --------------------------------------

def make_traction_function_from_grid(
    mesh: dxmesh.Mesh,
    V: dxfem.FunctionSpace,
    top_z_mm: float,
    X_mm: np.ndarray,  # (Ny, Nx)
    Y_mm: np.ndarray,  # (Ny, Nx)
    fx_mpa: np.ndarray,  # (Ny, Nx)
    fy_mpa: np.ndarray,  # (Ny, Nx)
    fz_mpa: np.ndarray,  # (Ny, Nx)
    z_tol: float = 1e-9,
) -> dxfem.Function:
    """
    Create a Vector CG1 function t(x) that equals the given traction on the *top surface*
    and is zero elsewhere. Values are obtained by bilinear interpolation from the
    (Ny,Nx) regular grid defined by X_mm, Y_mm.
    """
    # Build interpolators on the regular grid (mm → MPa)
    # Grid coordinates are increasing along x and y by construction in synth.loads.make_surface_grid.
    xi = (Y_mm[:, 0], X_mm[0, :])  # (y_grid, x_grid) order for RegularGridInterpolator
    fx_interp = RegularGridInterpolator(xi, fx_mpa, bounds_error=False, fill_value=0.0)
    fy_interp = RegularGridInterpolator(xi, fy_mpa, bounds_error=False, fill_value=0.0)
    fz_interp = RegularGridInterpolator(xi, fz_mpa, bounds_error=False, fill_value=0.0)

    t = dxfem.Function(V, name="traction")

    # Interpolate into DOF points, but zero everywhere except near top surface (z≈top_z_mm)
    # Note: need to Flip the z component so that “positive means compression”.
    def eval_at_dofs(x: np.ndarray):
        n = x.shape[1]
        out = np.zeros((3, n), dtype=float)
        mask = np.isclose(x[2], top_z_mm, atol=z_tol)
        if np.any(mask):
            # Interpolators expect (y, x) pairs
            pts = np.stack([x[1, mask], x[0, mask]], axis=1)
            out[0, mask] = fx_interp(pts)
            out[1, mask] = fy_interp(pts)
            # Note: flip sign so +fz = downward compression
            out[2, mask] = - fz_interp(pts)
        return out

    t.interpolate(eval_at_dofs)
    return t


# --------------------------------------
# Forward elasticity solve
# --------------------------------------

@dataclass
class SolverOptions:
    """
    PETSc KSP options. Defaults favor robustness (direct LU); switch to iterative
    for large problems.
    """
    ksp_type: str = "preonly"
    pc_type: str = "lu"                 # 'lu' uses MUMPS/SuperLU if available
    ksp_rtol: float = 1e-8
    ksp_atol: float = 1e-12
    ksp_monitor: bool = False           # set True to print residuals
    pc_factor_mat_solver_type: Optional[str] = None  # e.g. "mumps", "superlu_dist"


def solve_forward(
    geom: MeshBundle,
    mat: MaterialLayers,
    force_grid: Dict[str, np.ndarray],
    solver_options: SolverOptions = SolverOptions(),
    save_xdmf: Optional[str] = None,
) -> dxfem.Function:
    """
    Solve linear elasticity with layered material and a given surface traction field.

    Parameters
    ----------
    geom : MeshBundle
        Mesh + facet tags from geometry.py
    mat : MaterialLayers
        Layered material (E in MPa) and Poisson ratio
    force_grid : dict
        Keys: 'X_mm','Y_mm','fx_mpa','fy_mpa','fz_mpa' (as produced by synth.loads)
    solver_options : SolverOptions
        PETSc KSP options (defaults to LU for stability)
    save_xdmf : str or None
        If set, path to XDMF to save the displacement field.

    Returns
    -------
    u : dolfinx.fem.Function
        Displacement field in mm (Vector CG1).
    """
    mesh = geom.mesh
    mt = geom.facet_tags
    facet_id = geom.facet_id

    # Function space (Vector CG1)
    V = dxfem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # Trial/test
    u = TrialFunction(V)
    v = TestFunction(V)

    # Material fields
    lam_fun, mu_fun = make_layered_material_functions(mesh, mat)
    I = Identity(mesh.geometry.dim)

    # Strain/stress (linearised)
    def epsilon(w):  # symmetric gradient
        return sym(grad(w))

    def sigma(w):
        return 2.0 * mu_fun * epsilon(w) + lam_fun * tr(epsilon(w)) * I

    # Dirichlet BCs: clamp bottom and all sides (u=0)
    fdim = mesh.topology.dim - 1
    def loc_bottom(x): return np.isclose(x[2], 0.0, atol=1e-9)
    def loc_top(x):    return np.isclose(x[2], geom.top_z_mm, atol=1e-9)
    def loc_x0(x):     return np.isclose(x[0], 0.0, atol=1e-9)
    def loc_xL(x):     return np.isclose(x[0], geom.size_mm[0], atol=1e-9)
    def loc_y0(x):     return np.isclose(x[1], 0.0, atol=1e-9)
    def loc_yL(x):     return np.isclose(x[1], geom.size_mm[1], atol=1e-9)

    # Locate DOFs for each boundary; we collapse to a single set (clamp all)
    # (You could clamp only bottom and allow lateral bulging by omitting x0/xL/y0/yL.)
    btm_dofs = dxfem.locate_dofs_geometrical(V, loc_bottom)
    x0_dofs  = dxfem.locate_dofs_geometrical(V, loc_x0)
    xL_dofs  = dxfem.locate_dofs_geometrical(V, loc_xL)
    y0_dofs  = dxfem.locate_dofs_geometrical(V, loc_y0)
    yL_dofs  = dxfem.locate_dofs_geometrical(V, loc_yL)
    all_fix = np.unique(np.concatenate([btm_dofs, x0_dofs, xL_dofs, y0_dofs, yL_dofs]))

    u0 = dxfem.Function(V)  # zero vector
    bc_all = dxfem.dirichletbc(u0, all_fix)

    # Neumann traction on top
    t = make_traction_function_from_grid(
        mesh, V, geom.top_z_mm,
        force_grid["X_mm"], force_grid["Y_mm"],
        force_grid["fx_mpa"], force_grid["fy_mpa"], force_grid["fz_mpa"]
    )

    ds = Measure("ds", domain=mesh, subdomain_data=mt)
    #a = inner(sigma(u), epsilon(v)) * dx
    #L = inner(t, v) * ds(facet_id["top"])  # only integrate traction on the top face
    # UFL expressions
    a_ufl = inner(sigma(u), epsilon(v)) * dx
    L_ufl = inner(t, v) * ds(facet_id["top"])

    # Compile to dolfinx forms (required in 0.9)
    a = dxfem.form(a_ufl)
    L = dxfem.form(L_ufl)

    # Assemble system
    A = assemble_matrix(a, bcs=[bc_all])
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc_all]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc_all])

    # Solve with PETSc KSP (robust defaults: preonly+lu)
    uh = dxfem.Function(V, name="u")

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)

    opts = ksp.getOptionsPrefix() or ""
    # Apply options directly for clarity
    ksp.setType(solver_options.ksp_type)
    pc = ksp.getPC()
    pc.setType(solver_options.pc_type)
    if solver_options.pc_factor_mat_solver_type:
        pc.setFactorSolverType(solver_options.pc_factor_mat_solver_type)
    ksp.setTolerances(rtol=solver_options.ksp_rtol, atol=solver_options.ksp_atol)
    if solver_options.ksp_monitor:
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"[KSP] it={its}, ||r||={rnorm:.3e}"))

    ksp.setFromOptions()
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # After having uh (dolfinx.fem.Function) solved, save to XDMF for Paraview/PyVista
    if save_xdmf and (mesh.comm.rank == 0):
        with dxio.XDMFFile(mesh.comm, save_xdmf, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)

    # NEW: save DOFs to NPZ for round-trip use
    if mesh.comm.rank == 0:
        
        npz_path = save_xdmf if save_xdmf else "data/output/u.xdmf"
        npz_path = npz_path.replace(".xdmf", ".dofs.npz")
        np.savez_compressed(
            npz_path,
            dofs=uh.x.array.copy(),              # 1D array of size ndofs*3
            family="Lagrange",
            degree=1,
            value_shape=(mesh.geometry.dim,),    # (3,)
            note="Assign this to a Function(V) with the same mesh & space."
        )
        print("[OK] saved DOFs:", npz_path)

    return uh
