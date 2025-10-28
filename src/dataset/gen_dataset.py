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
Generate Dataset by spawning many per-sample processes.

Example:
  python src/dataset/gen_dataset.py \
    --root dataset/train \
    --material src/config/material.yaml \
    --sensor particle_vts \
    --render-config src/config/renderer.yaml \
    --n 1000 --jobs 8 \
    --mode_mix 0.25,0.25,0.25,0.25 \
    --n_balls_min 1 --n_balls_max 2
"""

from __future__ import annotations
import argparse, csv, os, random, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--material", type=Path, required=True)

    p.add_argument("--sensor", type=str, default="particle_vts", help="Registered sensor name (default: particle_vts)", choices=["particle_vts", "gelsight_style", "tac3d", "tac3d2"])
    p.add_argument("--render-config", type=Path, required=True)

    p.add_argument("--n", type=int, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--start_id", type=int, default=0)
    p.add_argument("--mode_mix", type=str, default="0.25,0.25,0.25,0.25",
                   help="probabilities for pressure,shear,torque,combo")
    p.add_argument("--n_balls_min", type=int, default=1)
    p.add_argument("--n_balls_max", type=int, default=2)
    p.add_argument("--Lx_mm", type=float, default=40.0)
    p.add_argument("--Ly_mm", type=float, default=30.0)
    p.add_argument("--Nx", type=int, default=80)
    p.add_argument("--Ny", type=int, default=60)
    p.add_argument("--debug_show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    mix = [float(x) for x in args.mode_mix.split(",")]
    assert len(mix) == 4 and abs(sum(mix) - 1.0) < 1e-6
    modes = ["pressure", "shear", "torque", "combo"]

    # Prepare manifest
    manifest_path = args.root / "manifest.csv"
    fcsv = open(manifest_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["id", "seed", "mode", "n_balls", "ok"])

    def launch_one(i: int):
        sid = args.start_id + i
        seed = sid * 17 + 12345
        # pick mode
        mode = random.choices(modes, weights=mix, k=1)[0]
        # random balls within bounds
        n_balls = random.randint(args.n_balls_min, args.n_balls_max)

        outdir = args.root / f"{sid:08d}"
        cmd = [
            sys.executable, "src/dataset/gen_sample.py",
            "--outdir", str(outdir),
            "--material", str(args.material),
            "--sensor", args.sensor,
            "--render-config", str(args.render_config),
            "--mode", mode,
            "--n_balls", str(n_balls),
            "--Lx_mm", str(args.Lx_mm),
            "--Ly_mm", str(args.Ly_mm),
            "--Nx", str(args.Nx),
            "--Ny", str(args.Ny),
            "--seed", str(seed),
            "--save_flow",
        ]
        if args.debug_show:
            cmd.append("--debug_show")

        try:
            subprocess.run(cmd, check=True)
            return (sid, seed, mode, n_balls, True)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] sample {sid} failed: {e}")
            return (sid, seed, mode, n_balls, False)
        
    # Use a thread pool to submit subprocesses
    ok_count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(launch_one, i) for i in range(args.n)]
        for fut in as_completed(futures):
            sid, seed, mode, n_balls, ok = fut.result()
            writer.writerow([sid, seed, mode, n_balls, int(ok)])    # add sample info to manifest file
            fcsv.flush()
            if ok:
                ok_count += 1

    fcsv.close()
    print(f"[DONE] generated {ok_count}/{args.n} samples at {args.root}")
    print("Manifest:", manifest_path)

if __name__ == "__main__":
    main()
