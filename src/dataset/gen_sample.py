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
Generate ONE dataset sample:
- Every Sample will be stored in a directory.
- Draw random params (ball(s), shear, torque).
- Build combined force field (multi-press Hertz + shear + torque).
- Run forward FEM (fast settings) to compute u; save XDMF + DOFs; save top-surface disp previews.
- Render two images (I0 undeformed, I1 deformed).
- Save force map NPZ, metadata JSON, and debug PNGs.

Example:
python src/dataset/gen_sample.py \
    --outdir dataset/train/000001 \
    --material src/config/material.yaml \
    --renderer src/config/renderer.yaml \
    --mode pressure --gauss_sigma_min_mm 1.0 --gauss_sigma_max_mm 4.0   --fz_peak_min_mpa 0.05 --fz_peak_max_mpa 0.15   \
    --n_balls 2   --seed 123 \
    --save_flow \
    --debug_show

python src/dataset/gen_sample.py \
    --outdir dataset/val/000001 \
    --material src/config/material.yaml \
    --renderer src/config/renderer.yaml \
    --mode shear  \
    --n_balls 1  --seed 41 \
    --save_flow \
    --debug_show

python src/dataset/gen_sample.py  \
    --outdir dataset/val/000002     --material src/config/material.yaml     --renderer src/config/renderer.yaml  \
    --mode torque  \
    --n_balls 1  --seed 42 \
    --save_flow

python src/dataset/gen_sample.py \
    --outdir dataset/train/000002 \
    --material src/config/material.yaml \
    --renderer src/config/renderer.yaml \
    --mode combo \
    --n_balls 2 --seed 100 \
    --save_flow
"""

from __future__ import annotations
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml
import cv2

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse our modules
from synth.loads import make_surface_grid, gaussian_shear_patch, torque_patch, make_surface_grid, multi_gaussian_combo
from viz.plots2d import plot_force_maps
from viz.quiver import draw_quiver_bgr

def _safe_center_gauss(rng, L: float, sigma: float, factor: float, edge_min: float) -> float:
    """
    Sample center with margin = max(edge_min, factor*sigma).
    factor≈2.5 keeps ~99% Gaussian mass inside the domain visually.
    """
    m = max(edge_min, factor * sigma)
    if L - 2.0 * m <= 1e-6:
        m = max(0.0, min(L * 0.49, (L / 2.0) - 1e-3))
    lo, hi = m, L - m
    if hi <= lo:  # degenerate → center
        return L * 0.5
    return rng.uniform(lo, hi)

def _draw_params(Lx, Ly, rng, args) -> dict:
    """
    Randomly draw from parameter space

    User-set factor for how far from edges the ball center must be (in radii)
    1.0 => fully inside; 0.75 => ~3/4 area inside
    """

    n_objs = int(args.n_balls)   # reuse your flag name
    objs = []
    for _ in range(n_objs):
        sigma = float(rng.uniform(args.gauss_sigma_min_mm, args.gauss_sigma_max_mm))
        cx = _safe_center_gauss(rng, Lx, sigma, args.sigma_margin_factor, args.edge_margin_mm)
        cy = _safe_center_gauss(rng, Ly, sigma, args.sigma_margin_factor, args.edge_margin_mm)

        # magnitudes per mode
        if args.mode in ["pressure", "combo"]:
            fz_peak = float(rng.uniform(args.fz_peak_min_mpa, args.fz_peak_max_mpa))
        else:
            fz_peak = 0.0

        if args.mode in ["shear", "combo"]:
            tau_s = float(rng.uniform(0.0, args.tau_shear_max_mpa))
            sdir  = float(rng.uniform(0.0, 360.0))
        else:
            tau_s, sdir = 0.0, 0.0

        if args.mode in ["torque", "combo"]:
            tau_t = float(rng.uniform(0.0, args.tau_torque_max_mpa))
        else:
            tau_t = 0.0

        objs.append(dict(
            cx_mm=cx, cy_mm=cy, sigma_mm=sigma,
            fz_peak_mpa=fz_peak,
            tau_shear_mpa=tau_s, shear_dir_deg=sdir,
            tau_torque_mpa=tau_t
        ))

    # shear/torque footprint sigma when we still use single-patch helpers elsewhere (kept for metadata)
    sigma_mm = float(rng.uniform(args.gauss_sigma_min_mm, args.gauss_sigma_max_mm))
    return dict(
        n_balls=n_objs, objects=objs, mode=args.mode, sigma_mm=sigma_mm
    )


def _build_force(Lx, Ly, Nx, Ny, params):
    X, Y = make_surface_grid(Lx, Ly, Nx, Ny)
    fx, fy, fz = multi_gaussian_combo(X, Y, params["objects"])
    return dict(X_mm=X, Y_mm=Y, fx_mpa=fx, fy_mpa=fy, fz_mpa=fz)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--material", type=Path, required=True, help="YAML used by forward FEM.")
    p.add_argument("--renderer", type=Path, required=True, help="YAML used by OpenCV renderer.")

    p.add_argument("--mode", type=str, default="combo", choices=["pressure", "shear", "torque", "combo"])

    p.add_argument("--normal_shape", type=str, default="gaussian", choices=["gaussian"], help="Normal profile type (currently only gaussian).")

    p.add_argument("--gauss_sigma_min_mm", type=float, default=1.0)
    p.add_argument("--gauss_sigma_max_mm", type=float, default=5.0)
    p.add_argument("--fz_peak_min_mpa", type=float, default=0.05)
    p.add_argument("--fz_peak_max_mpa", type=float, default=0.15)
    p.add_argument("--tau_shear_max_mpa", type=float, default=0.06)
    p.add_argument("--tau_torque_max_mpa", type=float, default=0.05)
    p.add_argument("--sigma_margin_factor", type=float, default=2.5, help="Keep centers ≥ factor*sigma from edges (≈2.5σ leaves ~99% mass inside).")

    p.add_argument("--n_balls", type=int, default=1)
    p.add_argument("--Lx_mm", type=float, default=40.0)
    p.add_argument("--Ly_mm", type=float, default=30.0)
    p.add_argument("--Nx", type=int, default=80)
    p.add_argument("--Ny", type=int, default=60)
    p.add_argument("--edge_margin_mm", type=float, default=2.0)

    p.add_argument("--flow_method", type=str, default="dis", choices=["farneback", "dis", "tvl1"], help="Optical flow method.")
    p.add_argument("--save_flow", action="store_true", help="Compute and save optical flow from I0->I1.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug_show", action="store_true")
    return p.parse_args()



def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1) randomly draw params
    params = _draw_params(args.Lx_mm, args.Ly_mm, rng, args)

    # 2) build force field
    force = _build_force(args.Lx_mm, args.Ly_mm, args.Nx, args.Ny, params)
    force_npz = args.outdir / "force_top.npz"
    np.savez_compressed(force_npz, **force, unit="MPa")
    # preview
    plot_force_maps(force["X_mm"], force["Y_mm"], force["fx_mpa"], force["fy_mpa"], force["fz_mpa"],
                    out_prefix=args.outdir / "force_preview", title=f"mode={args.mode}")

    # show debug window
    if args.debug_show:
        img_force = cv2.imread(str((args.outdir / "force_preview.fz.png").resolve()))
        cv2.imshow("force tz heatmap", img_force)

    # 3) forward FEM (call script in a subprocess, good isolation/MPI safety)
    xdmf = args.outdir / "u.xdmf"
    dofs = args.outdir / "u.dofs.npz"
    disp_prefix = args.outdir / "disp_top"
    cmd_forward = [
        sys.executable, "src/scripts/run_forward.py",
        "--force", str(force_npz),
        "--config", str(args.material),
        "--xdmf", str(xdmf),
        "--viz_prefix", str(disp_prefix),
        "--sample_Nx", str(args.Nx),
        "--sample_Ny", str(args.Ny),
        "--save_dofs_npz", str(dofs),
        "--ksp", "cg", "--pc", "gamg", "--ksp_monitor"
    ]
    t0 = time.time()
    subprocess.run(cmd_forward, check=True)
    t1 = time.time()
    print(f"[forward] {args.outdir.name} done in {t1-t0:.2f}s")

    if args.debug_show:
        img_disp = cv2.imread(str((args.outdir / "disp_top.uz.png").resolve()))
        cv2.imshow("top uz heatmap", img_disp)

    # 4) render Camera observation images, I0 (undeformed) and I1 (deformed)
    I0 = args.outdir / "I0.png"
    I1 = args.outdir / "I1.png"

    # render undeformed (mode: none)
    cmd_ren0 = [
        sys.executable, "src/scripts/render_camera_frame.py",
        "--config", str(args.renderer),
        "--out", str(I0),
        "--seed", str(args.seed),
        "--supersample", "2",
    ]
    subprocess.run(cmd_ren0, check=True)

    # render deformed (mode: xdmf reads sidecar DOFs)
    # renderer.yaml must have deformation.mode: xdmf and xdmf_path will be set here by env var
    with open(args.renderer, "r") as f:
        ren_cfg = yaml.safe_load(f)
    ren_cfg = dict(ren_cfg)     # make a copy of config
    ren_cfg["deformation"] = {"mode": "xdmf", "xdmf_path": str(xdmf)}   # add deformation files
    rtmp = args.outdir / "renderer_used.yaml"   # save new config file (with deformation)
    with open(rtmp, "w") as f:
        yaml.safe_dump(ren_cfg, f)
    cmd_ren1 = [
        sys.executable, "src/scripts/render_camera_frame.py",
        "--config", str(rtmp),                  # use new config file (with deformation)
        "--out", str(I1),
        "--seed", str(args.seed),
        "--supersample", "2",
    ]
    subprocess.run(cmd_ren1, check=True)

    if args.debug_show:
        img_cam = cv2.imread(str(I1.resolve()))
        cv2.imshow("camera I1", img_cam)

    # 5) save optical flow
    from synth.optical_flow import FlowConfig, FlowMethod, to_gray_f32_bgr, compute_flow, flow_to_bgr
    if args.save_flow:
        I0p = cv2.imread(str(I0), cv2.IMREAD_COLOR)
        I1p = cv2.imread(str(I1), cv2.IMREAD_COLOR)
        if I0p is None or I1p is None:
            raise RuntimeError("Failed to read I0/I1 for flow.")

        g0 = to_gray_f32_bgr(I0p)
        g1 = to_gray_f32_bgr(I1p)
        cfg_flow = FlowConfig(method=FlowMethod(args.flow_method))
        vx, vy = compute_flow(g0, g1, cfg_flow)   # from I0 -> I1

        # Save numeric flow
        np.savez_compressed(args.outdir / "flow_f32.npz", vx=vx, vy=vy, note="flow from I0 to I1")

        # Save a visualization
        vis = flow_to_bgr(vx, vy, clip_mag=None)
        cv2.imwrite(str(args.outdir / "flow_vis.png"), vis)

        # Quiver render (use sensible defaults; tune as you like)
        quiv = draw_quiver_bgr(
            vy, vx,
            block=16,        # stride between arrows (px)
            pool=1,          # no pooling (or set e.g. 5)
            scale=4.0,       # arrow length gain
            thickness=1,
            color=(0, 255, 255),
            bg=(0, 0, 0),    # or pick avg of I0 if preferred
            min_px=0.5,
            draw_centers=False,
            center_color=(255, 255, 255),
        )
        cv2.imwrite(str(args.outdir / "flow_quiver.png"), quiv)

        if args.debug_show:
            cv2.imshow("flow vis", vis)
            cv2.imshow("flow quiver", quiv)


    # 6) metadata, save as .json in the same folder
    meta = dict(
        seed=int(args.seed),
        mode=str(args.mode),
        Lx_mm=float(args.Lx_mm), Ly_mm=float(args.Ly_mm),
        grid=[int(args.Nx), int(args.Ny)],
        params=params
    )
    with open(args.outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] sample saved to {args.outdir}")
    if args.debug_show:
        print("Press any key to close debug windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
