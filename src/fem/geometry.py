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
Geometry and boundary tagging helpers for the 6D tactile FEM.

- Creates a 3D box mesh in *millimetres* (mm) using DOLFINx 0.9.
- Tags boundary facets (bottom, top, four sides) with integer IDs.
- Provides a small container class to pass geometry around cleanly.

This module avoids any physics; it only handles mesh + boundary meta.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dxmesh
from dolfinx import cpp as dxcpp
from dolfinx import fem as dxfem


# Facet IDs (keep stable; used by forward solver)
FACET_ID = {
    "bottom": 1,
    "top": 2,
    "x0": 3,   # x = 0 (left)
    "xL": 4,   # x = Lx (right)
    "y0": 5,   # y = 0 (front)
    "yL": 6,   # y = Ly (back)
}


@dataclass
class MeshBundle:
    mesh: dxmesh.Mesh
    facet_tags: dxmesh.MeshTags
    facet_id: Dict[str, int]
    size_mm: Tuple[float, float, float]  # (Lx, Ly, Lz)
    div: Tuple[int, int, int]            # mesh divisions
    top_z_mm: float
    comm: MPI.Comm


def create_box_mesh_mm(
    Lx_mm: float,
    Ly_mm: float,
    Lz_mm: float,
    nx: int,
    ny: int,
    nz: int,
    cell_type: dxcpp.mesh.CellType = dxcpp.mesh.CellType.tetrahedron,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dxmesh.Mesh:
    """
    Create a box mesh [0,Lx]×[0,Ly]×[0,Lz] in *mm* with (nx,ny,nz) divisions.
    Uses DOLFINx's built-in `create_box` (tetrahedra by default).
    """
    domain = [np.array([0.0, 0.0, 0.0]), np.array([Lx_mm, Ly_mm, Lz_mm])]
    mesh = dxmesh.create_box(
        comm,
        points=domain,
        n=(nx, ny, nz),
        cell_type=cell_type,
    )
    return mesh


def tag_boundary_facets(
    mesh: dxmesh.Mesh,
    Lx_mm: float,
    Ly_mm: float,
    Lz_mm: float,
    tol: float = 1e-9,
) -> dxfem.MeshTags:
    """
    Tag boundary facets by comparing coordinates to the six planes.
    Returns a MeshTags object on facets (tdim-1).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Markers (vectorised) for boundary detection
    def at_x0(x): return np.isclose(x[0], 0.0, atol=tol)
    def at_xL(x): return np.isclose(x[0], Lx_mm, atol=tol)
    def at_y0(x): return np.isclose(x[1], 0.0, atol=tol)
    def at_yL(x): return np.isclose(x[1], Ly_mm, atol=tol)
    def at_z0(x): return np.isclose(x[2], 0.0, atol=tol)
    def at_zL(x): return np.isclose(x[2], Lz_mm, atol=tol)

    # Collect facet indices for each boundary
    facets_x0 = dxmesh.locate_entities_boundary(mesh, fdim, at_x0)
    facets_xL = dxmesh.locate_entities_boundary(mesh, fdim, at_xL)
    facets_y0 = dxmesh.locate_entities_boundary(mesh, fdim, at_y0)
    facets_yL = dxmesh.locate_entities_boundary(mesh, fdim, at_yL)
    facets_z0 = dxmesh.locate_entities_boundary(mesh, fdim, at_z0)  # bottom
    facets_zL = dxmesh.locate_entities_boundary(mesh, fdim, at_zL)  # top

    # Stack and build tags
    indices = np.concatenate([facets_z0, facets_zL, facets_x0, facets_xL, facets_y0, facets_yL])
    values = np.concatenate([
        np.full_like(facets_z0, FACET_ID["bottom"]),
        np.full_like(facets_zL, FACET_ID["top"]),
        np.full_like(facets_x0, FACET_ID["x0"]),
        np.full_like(facets_xL, FACET_ID["xL"]),
        np.full_like(facets_y0, FACET_ID["y0"]),
        np.full_like(facets_yL, FACET_ID["yL"]),
    ])

    # Sorted by facet index (optional but nice)
    order = np.argsort(indices)
    mt = dxmesh.meshtags(
        mesh, fdim,
        indices[order].astype(np.int32, copy=False),
        values[order].astype(np.int32, copy=False)
    )
    return mt


def make_geometry(
    Lx_mm: float,
    Ly_mm: float,
    Lz_mm: float,
    nx: int,
    ny: int,
    nz: int,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MeshBundle:
    """
    Convenience: build mesh + facet tags + metadata in one call.
    """
    m = create_box_mesh_mm(Lx_mm, Ly_mm, Lz_mm, nx, ny, nz, comm=comm)
    mt = tag_boundary_facets(m, Lx_mm, Ly_mm, Lz_mm)
    bundle = MeshBundle(
        mesh=m,
        facet_tags=mt,
        facet_id=FACET_ID.copy(),
        size_mm=(Lx_mm, Ly_mm, Lz_mm),
        div=(nx, ny, nz),
        top_z_mm=Lz_mm,
        comm=comm,
    )
    return bundle
