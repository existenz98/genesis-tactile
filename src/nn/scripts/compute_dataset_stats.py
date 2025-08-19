from __future__ import annotations
import argparse, json
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--flow_file", type=str, default="flow_f32.npz")
    ap.add_argument("--force_file", type=str, default="force_top.npz")
    ap.add_argument("--out", type=str, required=True)  # stats.json
    args = ap.parse_args()

    root = Path(args.root)
    # gather ids
    ids = []
    with open(args.manifest, "r", newline="") as f:
        r = csv.DictReader(f)
        id_key = "id" if "id" in r.fieldnames else r.fieldnames[0]
        ok_key = "ok" if "ok" in r.fieldnames else None
        for row in r:
            if ok_key and row[ok_key] not in ("1","True","true","yes","y"):
                continue
            ids.append(row[id_key])

    # collect magnitudes
    mags = []                               # magnitude of flow (vx, vy)
    tz_vals, tx_vals, ty_vals = [], [], []  # magnitude of force (tx, ty, tz)
    for sid in tqdm(ids, desc="scan"):
        fn = sid.zfill(8)     # '12' to '00000012'
        d = root / fn
        flow_p = d / args.flow_file
        force_p = d / args.force_file
        if not flow_p.exists() or not force_p.exists():
            continue
        with np.load(flow_p) as f:
            vx, vy = f["vx"], f["vy"]
        mags.append(np.abs(vx).reshape(-1))
        mags.append(np.abs(vy).reshape(-1))

        with np.load(force_p) as g:
            tz_vals.append(np.abs(g["fz_mpa"]).reshape(-1))
            tx_vals.append(np.abs(g["fx_mpa"]).reshape(-1))
            ty_vals.append(np.abs(g["fy_mpa"]).reshape(-1))

    print(f"loaded from dataset, mag[{len(mags)}], fx[{len(tx_vals)}, fy[{len(ty_vals)}], fz[{len(tz_vals)}]]")

    mags = np.concatenate(mags, axis=0) if mags else np.array([1.0])
    tz_all = np.concatenate(tz_vals, axis=0) if tz_vals else np.array([1.0])
    tx_all = np.concatenate(tx_vals, axis=0) if tx_vals else np.array([1.0])
    ty_all = np.concatenate(ty_vals, axis=0) if ty_vals else np.array([1.0])

    stats = {
        "flow_scale_p95": float(np.percentile(mags, 95.0)),
        "flow_mean": 0.0,
        "force_scale_p99": [
            float(np.percentile(tz_all, 99.0)),
            float(np.percentile(tx_all, 99.0)),
            float(np.percentile(ty_all, 99.0)),
        ],
        # good starting alphas for log mapping (10-20th pct)
        "log_alpha_p10": [
            float(np.percentile(tz_all, 10.0)),
            float(np.percentile(tx_all, 10.0)),
            float(np.percentile(ty_all, 10.0)),
        ]
    }
    for k,v in stats.items(): print(k, "=", v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print("[ok] wrote", args.out)
    

if __name__ == "__main__":
    main()
