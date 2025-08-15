
"""
Forward solve: surface traction (MPa) -> displacement field (mm).

Usage example:
  python src/scripts/run_forward.py \
    --force data/input/force_combo.npz \
    --config src/config/default.yaml \
    --xdmf data/output/u.xdmf \
    --viz_prefix data/output/disp_top \
    --sample_Nx 60 --sample_Ny 40

Fast:
  python src/scripts/run_forward.py \
    --force data/input/force_combo.npz \
    --config src/config/default.yaml \
    --xdmf data/output/u.xdmf \
    --viz_prefix data/output/disp_top \
    --sample_Nx 60 --sample_Ny 40 \
    --ksp cg --pc gamg --ksp_monitor

This saves:
  - XDMF (mesh + u) for ParaView/PyVista
  - PNGs of top-surface displacement (u_z heatmap, (u_x,u_y) quiver)
  - Optional NPZ of sampled top-surface displacement grid
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from mpi4py import MPI


# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from fem.geometry import make_geometry
from fem.forward import (
    MaterialLayers,
    SolverOptions,
    solve_forward,
)
from viz.plots2d import save_disp_maps


from typing import Dict, Any
import json
import yaml


def load_config_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_force_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path) as d:
        X_mm = d["X_mm"]
        Y_mm = d["Y_mm"]
        fx = d["fx_mpa"]
        fy = d["fy_mpa"]
        fz = d["fz_mpa"]
        meta_json = d.get("meta_json", None)
    meta = {}
    if meta_json is not None:
        try:
            meta = json.loads(str(meta_json))
        except Exception:
            meta = {}
    return dict(X_mm=X_mm, Y_mm=Y_mm, fx_mpa=fx, fy_mpa=fy, fz_mpa=fz, meta=meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward FEM (traction -> displacement).")
    p.add_argument("--force", type=Path, required=True, help="NPZ with (X_mm,Y_mm,fx,fy,fz) in MPa.")
    p.add_argument("--config", type=Path, default=Path("src/config/default.yaml"), help="YAML config.")
    p.add_argument("--xdmf", type=Path, default=Path("data/output/u.xdmf"), help="Output XDMF for u.")
    p.add_argument("--viz_prefix", type=Path, default=Path("data/output/disp_top"), help="Prefix for PNGs.")
    p.add_argument("--sample_Nx", type=int, default=60, help="Sampling grid on top-surface (x).")
    p.add_argument("--sample_Ny", type=int, default=40, help="Sampling grid on top-surface (y).")
    p.add_argument("--save_top_npz", type=Path, default=None, help="Optional NPZ to save sampled top-surface u.")
    p.add_argument("--ksp_monitor", action="store_true", help="Print KSP residuals each iteration.")
    p.add_argument("--pc", type=str, default="lu", help="PETSc PC type (lu, hypre, ilu, jacobi...).")
    p.add_argument("--ksp", type=str, default="preonly", help="PETSc KSP type (preonly, cg, gmres...).")
    return p.parse_args()


def _make_material_from_cfg(cfg) -> MaterialLayers:
    nu = float(cfg["material"]["poisson"])
    layers_cfg = cfg["material"]["layers"]
    # Ensure z_min/z_max are in ascending order
    layers = []
    for entry in layers_cfg:
        layers.append((float(entry["z_min"]), float(entry["z_max"]), float(entry["E_MPa"])))
    # Sort by z_min for safety
    layers = tuple(sorted(layers, key=lambda t: t[0]))
    return MaterialLayers(nu=nu, layers=layers)


def _check_force_extent_vs_geom(force, Lx_mm: float, Ly_mm: float) -> None:
    # Ensure force grid spans the geometry extent (0..Lx, 0..Ly)
    x_min, x_max = float(force["X_mm"].min()), float(force["X_mm"].max())
    y_min, y_max = float(force["Y_mm"].min()), float(force["Y_mm"].max())
    if abs(x_min) > 1e-9 or abs(y_min) > 1e-9 or abs(x_max - Lx_mm) > 1e-6 or abs(y_max - Ly_mm) > 1e-6:
        print("[WARN] Force grid extent does not match geometry. Expected X:[0,%.3f],Y:[0,%.3f], "
              "got X:[%.3f,%.3f],Y:[%.3f,%.3f]. Proceeding anyway."
              % (Lx_mm, Ly_mm, x_min, x_max, y_min, y_max))


def _sample_top_surface_regular(uh, geom, Nx: int, Ny: int):
    """
    Sample displacement at a regular (Ny,Nx) grid on the top surface z=top_z_mm.
    Returns X_mm, Y_mm, ux, uy, uz (all (Ny, Nx)).
    """
    import numpy as np
    import dolfinx.geometry as dxgeo

    Lx, Ly = geom.size_mm[0], geom.size_mm[1]
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Nudge slightly inside the domain to avoid "exactly-on-boundary" misses
    Z = np.full_like(X, geom.top_z_mm - 1e-9)

    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # (npts, 3)
    npts = pts.shape[0]

    # Vectorized collision search (dolfinx 0.9 API)
    bb = dxgeo.bb_tree(geom.mesh, geom.mesh.topology.dim, padding=0.0)
    candidates = dxgeo.compute_collisions_points(bb, pts)              # AdjacencyList
    colliding = dxgeo.compute_colliding_cells(geom.mesh, candidates, pts)  # AdjacencyList

    # Pick first colliding cell for each point
    offs = colliding.offsets
    arr  = colliding.array
    cells = np.full(npts, -1, dtype=np.int32)
    for i in range(npts):
        s = offs[i]
        e = offs[i+1]
        if e > s:
            cells[i] = arr[s]

    mask = cells >= 0
    vals = np.zeros((npts, 3), dtype=float)
    if np.any(mask):
        vals[mask, :] = uh.eval(pts[mask], cells[mask])

    ux = vals[:, 0].reshape(Ny, Nx)
    uy = vals[:, 1].reshape(Ny, Nx)
    uz = vals[:, 2].reshape(Ny, Nx)
    return X, Y, ux, uy, uz


def main():
    args = parse_args()
    cfg = load_config_yaml(args.config)

    # Geometry
    Lx, Ly, Lz = map(float, cfg["geometry"]["size_mm"])
    nx, ny, nz = map(int, cfg["geometry"]["mesh_div"])
    geom = make_geometry(Lx, Ly, Lz, nx, ny, nz, comm=MPI.COMM_WORLD)

    # Material
    mat = _make_material_from_cfg(cfg)

    # Force field
    force = load_force_npz(args.force)
    _check_force_extent_vs_geom(force, Lx, Ly)

    # Solve
    solver_opts = SolverOptions(
        ksp_type=args.ksp,
        pc_type=args.pc,
        ksp_monitor=args.ksp_monitor,
    )
    uh = solve_forward(
        geom=geom,
        mat=mat,
        force_grid=force,
        solver_options=solver_opts,
        save_xdmf=str(args.xdmf) if args.xdmf else None,
    )
    if MPI.COMM_WORLD.rank == 0:
        print("[OK] Forward solve done. XDMF:", args.xdmf)

    # Sample top surface onto a regular grid for plotting
    Xs, Ys, ux, uy, uz = _sample_top_surface_regular(uh, geom, args.sample_Nx, args.sample_Ny)

    # Visualization (2D): u_z heatmap + (u_x,u_y) quiver
    if MPI.COMM_WORLD.rank == 0:
        args.viz_prefix.parent.mkdir(parents=True, exist_ok=True)
        png1, png2 = save_disp_maps(Xs, Ys, ux, uy, uz, out_prefix=args.viz_prefix, title="top surface u")
        print("Saved previews:", png1, png2)

        if args.save_top_npz:
            np.savez_compressed(
                args.save_top_npz,
                X_mm=Xs, Y_mm=Ys, ux_mm=ux, uy_mm=uy, uz_mm=uz,
                note="Top-surface displacement (mm), sampled on regular grid."
            )
            print("Saved sampled top-surface displacement:", args.save_top_npz)


if __name__ == "__main__":
    main()
