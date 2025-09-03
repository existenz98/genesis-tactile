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
Compute top-surface traction from a full 3D displacement field (Problem 1 iFEM):

It will:
  - read mesh (XDMF) and u DOFs (NPZ),
  - compute sigma(u) with layered E, nu from YAML,
  - sample traction t = sigma n on the top plane on an (Ny,Nx) grid,
  - save NPZ and two PNG previews (tz heatmap; (tx,ty) quiver over |tz|).


Example:
# standard:
  python src/scripts/run_ifem_from_u.py \
    --mesh data/output/u.xdmf \
    --dofs data/output/u.dofs.npz \
    --config src/config/material.yaml \
    --Nx 60 --Ny 40 \
    --out data/output/traction_from_u.npz \
    --preview_prefix data/output/traction_from_u \

# fast:
  python src/scripts/run_ifem_from_u.py \
    --mesh data/output/u.xdmf \
    --dofs data/output/u.dofs.npz \
    --config src/config/material.yaml \
    --Nx 60 --Ny 40 \
    --pc jacobi --ksp cg --ksp_rtol 1e-7 --ksp_atol 0 --ksp_monitor

Note: 
1. solver parameters
Fast:
--ksp cg
--pc can choose from:  hypre - best,  gamg - good, jacobi - fastest.

“gold standard” baseline: Exact projection, slower but most accurate:
--ksp preonly --pc lu

2. Special test: force z to original mesh's z.  (only use the correct mesh x,y)
--drop_uz
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import yaml

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from fem.ifem_from_u import traction_from_u, MaterialLayers, IFEMOptions
from viz.plots2d import plot_force_maps


def _load_mat_layers(cfg_path: Path) -> MaterialLayers:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    nu = float(cfg["material"]["poisson"])
    layers_cfg = cfg["material"]["layers"]
    layers = []
    for item in layers_cfg:
        layers.append((float(item["z_min"]), float(item["z_max"]), float(item["E_MPa"])))
    layers = tuple(sorted(layers, key=lambda t: t[0]))
    return MaterialLayers(nu=nu, layers=layers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iFEM: traction from full 3D displacement (Problem 1).")
    p.add_argument("--mesh", type=Path, required=True, help="XDMF mesh file (the one you wrote in forward).")
    p.add_argument("--dofs", type=Path, required=True, help="NPZ with u.x.array saved (sidecar of XDMF).")
    p.add_argument("--config", type=Path, required=True, help="YAML with material layers (same as forward).")
    p.add_argument("--Nx", type=int, default=60, help="Sampling grid in x.")
    p.add_argument("--Ny", type=int, default=40, help="Sampling grid in y.")
    p.add_argument("--out", type=Path, default=Path("data/output/traction_from_u.npz"),
                   help="Output NPZ for (X,Y, tx,ty,tz).")
    p.add_argument("--preview_prefix", type=Path, default=Path("data/output/traction_from_u"),
                   help="Prefix for preview PNGs.")
    p.add_argument("--drop_uz", action="store_true", help="Set u_z to zero before computing stress/traction.")
    p.add_argument("--ksp_monitor", action="store_true", help="Monitor the sigma projection solver.")
    p.add_argument("--pc", type=str, default="lu", help="PC type (lu, hypre, gamg, ilu...).")
    p.add_argument("--ksp", type=str, default="preonly", help="KSP type (preonly, cg, gmres...).")
    p.add_argument("--ksp_rtol", type=float, default=1e-10)
    p.add_argument("--ksp_atol", type=float, default=1e-12)
    return p.parse_args()


def main():
    args = parse_args()
    mat = _load_mat_layers(args.config)

    opts = IFEMOptions(
        drop_uz=args.drop_uz,
        ksp_type=args.ksp,
        pc_type=args.pc,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_monitor=args.ksp_monitor
    )

    out = traction_from_u(
        mesh_xdmf=str(args.mesh),
        dofs_npz=str(args.dofs),
        mat=mat,
        Nx=args.Nx, Ny=args.Ny,
        opts=opts
    )

    # Save NPZ
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_mm=out["X_mm"], Y_mm=out["Y_mm"],
        tx_mpa=out["tx_mpa"], ty_mpa=out["ty_mpa"], tz_mpa=out["tz_mpa"],
        meta_json=json.dumps(out["meta"])
    )
    print("[OK] traction saved:", args.out)

    # Previews
    png_hm, png_qv = plot_force_maps(
        out["X_mm"], out["Y_mm"], out["tx_mpa"], out["ty_mpa"], out["tz_mpa"],
        out_prefix=args.preview_prefix,
        title=f"iFEM traction (drop_uz={args.drop_uz})"
    )
    print("Preview PNGs:", png_hm, png_qv)


if __name__ == "__main__":
    main()
