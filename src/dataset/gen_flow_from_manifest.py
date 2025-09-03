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
Retrofit optical flow for existing dataset based on manifest.csv.

Usage:
  python src/dataset/gen_flow_from_manifest.py \
    --manifest dataset/train/manifest.csv \
    --jobs 8 \
    --flow_method dis \
    --overwrite

For each row with ok==1, it expects a folder <root>/<id>/ with I0.png and I1.png.
Saves:
  - flow_f32.npz  (vx, vy float32)
  - flow_vis.png  (color wheel)
"""


from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth.optical_flow import FlowConfig, FlowMethod, to_gray_f32_bgr, compute_flow, flow_to_bgr
from viz.quiver import draw_quiver_bgr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--flow_method", type=str, default="farneback",
                   choices=["farneback","dis","tvl1"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def process_row(root: Path, sid: str, flow_method: str, overwrite: bool) -> tuple[str, bool, str]:
    sid = sid.zfill(8)     # '12' to '00000012'
    d = root / sid
    I0 = d / "I0.png"; I1 = d / "I1.png"
    print(I0, I1)
    out_npz = d / "flow_f32.npz"; out_vis = d / "flow_vis.png"

    if not I0.exists() or not I1.exists():
        return sid, False, "missing I0/I1"
    if out_npz.exists() and not overwrite:
        return sid, True, "exists"

    img0 = cv2.imread(str(I0), cv2.IMREAD_COLOR)
    img1 = cv2.imread(str(I1), cv2.IMREAD_COLOR)
    if img0 is None or img1 is None:
        return sid, False, "imread failed"

    g0 = to_gray_f32_bgr(img0); g1 = to_gray_f32_bgr(img1)
    cfg = FlowConfig(method=FlowMethod(flow_method))
    vx, vy = compute_flow(g0, g1, cfg)

    np.savez_compressed(out_npz, vx=vx, vy=vy, note="flow from I0 to I1")
    vis = flow_to_bgr(vx, vy)
    cv2.imwrite(str(out_vis), vis)

    quiv = draw_quiver_bgr(
        vy, vx,
        block=16, pool=1, scale=4.0,
        thickness=1, color=(0, 255, 255),
        bg=(0, 0, 0), min_px=0.5,
        draw_centers=False, center_color=(255, 255, 255)
    )
    cv2.imwrite(str(d / "flow_quiver.png"), quiv)
    
    return sid, True, "ok"

def main():
    args = parse_args()
    manifest = args.manifest
    root = manifest.parent

    rows = []
    with open(manifest, "r", newline="") as f:
        r = csv.DictReader(f)
        # deduce id column name
        id_key = "id" if "id" in r.fieldnames else r.fieldnames[0]
        ok_key = "ok" if "ok" in r.fieldnames else None
        for row in r:
            sid = row[id_key]
            if ok_key and row[ok_key] not in ("1", "True", "true", "yes", "y"):
                continue
            rows.append(sid)

    print(f"[flow] total rows to process: {len(rows)}")
    okc = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(process_row, root, sid, args.flow_method, args.overwrite) for sid in rows]
        for fut in as_completed(futures):
            sid, ok, msg = fut.result()
            if ok: okc += 1
            else:  print(f"[WARN] {sid}: {msg}")
    print(f"[DONE] flow created/kept for {okc}/{len(rows)} samples")

if __name__ == "__main__":
    main()
