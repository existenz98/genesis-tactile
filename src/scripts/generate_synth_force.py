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
Generate a synthetic surface force field (MPa) and save:
- compressed NPZ with arrays (X_mm, Y_mm, fx_mpa, fy_mpa, fz_mpa)
- preview PNGs: t_z heatmap and (t_x, t_y) quiver
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth.loads import build_force_field
from viz.plots2d import plot_force_maps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic surface traction field (MPa).")
    p.add_argument("--preset", type=str, default="pressure",
                   choices=["pressure", "shear", "torque", "combo"],
                   help="Force pattern preset.")
    # If using Hertz:
    # p.add_argument("--ball_radius_mm", type=float, default=5.0)
    # p.add_argument("--indent_mm", type=float, default=0.3)
    p.add_argument("--Lx_mm", type=float, default=40.0, help="Width in mm.")
    p.add_argument("--Ly_mm", type=float, default=30.0, help="Height in mm.")
    p.add_argument("--Nx", type=int, default=80, help="Grid points in x.")
    p.add_argument("--Ny", type=int, default=60, help="Grid points in y.")
    p.add_argument("--cx_mm", type=float, default=None, help="Center x (mm). Default = Lx/2.")
    p.add_argument("--cy_mm", type=float, default=None, help="Center y (mm). Default = Ly/2.")
    p.add_argument("--sigma_mm", type=float, default=3.0, help="Gaussian std radius (mm).")
    p.add_argument("--fz_peak_mpa", type=float, default=0.10, help="Peak normal traction (MPa).")
    p.add_argument("--tau_shear_mpa", type=float, default=0.05, help="Peak shear traction (MPa).")
    p.add_argument("--tau_torque_mpa", type=float, default=0.05, help="Peak torque traction (MPa).")
    p.add_argument("--shear_dir_deg", type=float, default=0.0, help="Shear direction (deg).")
    p.add_argument("--torque_inner_mm", type=float, default=0.0, help="Inner dead zone radius (mm).")
    p.add_argument("--seed", type=int, default=None, help="Random seed (if any).")
    p.add_argument("--out", type=Path, default=Path("data/input/force.npz"),
                   help="Path to save NPZ.")
    p.add_argument("--preview_prefix", type=Path, default=Path("data/input/force_preview"),
                   help="Path prefix for preview PNGs (no extension).")
    return p.parse_args()


def main():
    args = parse_args()
    ff = build_force_field(
        preset=args.preset,
        Lx_mm=args.Lx_mm,
        Ly_mm=args.Ly_mm,
        Nx=args.Nx,
        Ny=args.Ny,
        cx_mm=args.cx_mm,
        cy_mm=args.cy_mm,
        sigma_mm=args.sigma_mm,
        fz_peak_mpa=args.fz_peak_mpa,
        # ball_radius_mm=args.ball_radius_mm,
        # indent_mm=args.indent_mm,
        tau_shear_mpa=args.tau_shear_mpa,
        tau_torque_mpa=args.tau_torque_mpa,
        shear_dir_deg=args.shear_dir_deg,
        torque_inner_mm=args.torque_inner_mm,
        seed=args.seed,
    )

    # Save NPZ
    ff.save_npz(args.out)

    # Visualization
    fz_png, quiv_png = plot_force_maps(
        ff.X_mm, ff.Y_mm, ff.fx_mpa, ff.fy_mpa, ff.fz_mpa,
        out_prefix=args.preview_prefix,
        title=f"preset={args.preset}"
    )

    # summary
    meta_pretty = json.dumps(ff.meta, indent=2)
    print("Saved:", args.out)
    print("Preview:", fz_png, quiv_png)
    print("Meta:", meta_pretty)


if __name__ == "__main__":
    main()
