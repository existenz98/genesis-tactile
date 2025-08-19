from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from .config import load_config, Cfg
from .data import FlowForceDataset, read_manifest_ids
from .models.unet_basic import UNetBasic

# points to ./src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.plots2d import plot_force_maps



def inverse_target_scaling(y, cfg: Cfg, device):
    mode = cfg.scaling.target_mode.lower()
    if mode == "linear":
        scales = torch.tensor(cfg.scaling.force_scale, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        return y * scales
    elif mode == "log":
        alpha = torch.tensor(cfg.scaling.log_alpha, dtype=y.dtype, device=device).view(1, -1, 1, 1)
        return torch.sign(y) * alpha * (torch.exp(torch.abs(y)) - 1.0)
    else:
        raise ValueError

def save_npz_force(p: Path, tz, tx, ty):
    np.savez_compressed(p, 
                        fz_mpa=tz.astype(np.float32),
                        fx_mpa=tx.astype(np.float32),
                        fy_mpa=ty.astype(np.float32))

def _to_u8_abs(img: np.ndarray) -> np.ndarray:
    v = np.abs(img)
    v95 = np.percentile(v, 95.0) if v.size > 0 else 1.0
    v95 = max(v95, 1e-6)
    out = np.clip(v / v95, 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset: use all IDs in manifest
    ids = read_manifest_ids(cfg.dataset.manifest)
    ds = FlowForceDataset(cfg.dataset, cfg.scaling, cfg.augment, ids, is_train=False, seed=123)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = UNetBasic(in_ch=cfg.model.in_ch, out_ch=cfg.model.out_ch,
                      base_ch=cfg.model.channels, norm=cfg.model.norm,
                      nonneg_tz_head=cfg.model.nonneg_tz_head).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # dataset-level error accumulators
    se_tz = 0.0; se_tx = 0.0; se_ty = 0.0   # sum of squared errors
    ae_tz = 0.0; ae_tx = 0.0; ae_ty = 0.0   # sum of absolute errors
    n_pix  = 0                              # total pixels counted

    with torch.no_grad():
        for sample in tqdm(dl, desc="infer"):
            sid = sample["id"][0]
            x = sample["x"].to(device)       # (1,2,H,W)
            y_hat = model(x)                 # (1,3,H,W)

            # invert scaling to raw units (MPa)
            t_pred = inverse_target_scaling(y_hat, cfg, device)  # raw units
            t_pred = t_pred.squeeze(0).cpu().numpy()  # (3,H,W) order [tz, tx, ty]
            tz, tx, ty = t_pred[0], t_pred[1], t_pred[2]

            savedir = outdir / sid; savedir.mkdir(parents=True, exist_ok=True)
            save_npz_force(savedir / "force_pred.npz", tz, tx, ty)

            # --- abs |tz| PNG ---
            import cv2
            cv2.imwrite(str(savedir / "tz_pred.png"), _to_u8_abs(tz))
            cv2.imwrite(str(savedir / "tx_pred.png"), _to_u8_abs(tx))
            cv2.imwrite(str(savedir / "ty_pred.png"), _to_u8_abs(ty))

            # --- Quiver plot of fx, fy, over |fz| background ---
            # Try to use ground-truth X_mm, Y_mm if available; else fall back to pixel coords.
            X_mm = None; Y_mm = None
            gt_p = Path(cfg.dataset.root) / sid / cfg.dataset.force_file
            if gt_p.exists():
                with np.load(gt_p) as d:
                    if "X_mm" in d and "Y_mm" in d:
                        X_mm = d["X_mm"]; Y_mm = d["Y_mm"]
            if X_mm is None or Y_mm is None:
                H, W = tz.shape
                xs = np.linspace(0, W-1, W, dtype=np.float32)
                ys = np.linspace(0, H-1, H, dtype=np.float32)
                X_mm, Y_mm = np.meshgrid(xs, ys)

            _ = plot_force_maps(X_mm, Y_mm, tx, ty, tz,
                                #fz_clim=(0.0, 1.0),
                                out_prefix=savedir / "pred",
                                title="prediction")

            # --- Per-sample metrics (if GT exists) & dataset-level aggregation ---

            # optional metrics if gt exists
            if gt_p.exists():
                with np.load(gt_p) as d:
                    # accept either naming
                    if "tz_mpa" in d:
                        tz_gt, tx_gt, ty_gt = d["tz_mpa"], d["tx_mpa"], d["ty_mpa"]
                    else:
                        tz_gt, tx_gt, ty_gt = d["fz_mpa"], d["fx_mpa"], d["fy_mpa"]

                # If sizes differ, resize GT to pred size
                H, W = tz.shape
                if tz_gt.shape != tz.shape:
                    tz_gt = cv2.resize(tz_gt.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    tx_gt = cv2.resize(tx_gt.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    ty_gt = cv2.resize(ty_gt.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

                # Per-sample RMSEs
                rmse_tz = float(np.sqrt(np.mean((tz - tz_gt) ** 2)))
                rmse_tx = float(np.sqrt(np.mean((tx - tx_gt) ** 2)))
                rmse_ty = float(np.sqrt(np.mean((ty - ty_gt) ** 2)))
                with open(savedir / "metrics.json", "w") as f:
                    json.dump({"rmse_tz": rmse_tz, "rmse_tx": rmse_tx, "rmse_ty": rmse_ty}, f, indent=2)

                # Aggregate sums
                se_tz += float(np.sum((tz - tz_gt) ** 2))
                se_tx += float(np.sum((tx - tx_gt) ** 2))
                se_ty += float(np.sum((ty - ty_gt) ** 2))
                ae_tz += float(np.sum(np.abs(tz - tz_gt)))
                ae_tx += float(np.sum(np.abs(tx - tx_gt)))
                ae_ty += float(np.sum(np.abs(ty - ty_gt)))
                n_pix += (H * W)

    # ---- Print dataset-level averages (only if GT existed for at least one sample) ----
    if n_pix > 0:
        rmse_tz = np.sqrt(se_tz / n_pix)
        rmse_tx = np.sqrt(se_tx / n_pix)
        rmse_ty = np.sqrt(se_ty / n_pix)
        mae_tz  = ae_tz / n_pix
        mae_tx  = ae_tx / n_pix
        mae_ty  = ae_ty / n_pix
        print("\n[Dataset Avg Error]  (units = MPa)")
        print(f"  RMSE:  fz={rmse_tz:.6f}, fx={rmse_tx:.6f}, fy={rmse_ty:.6f}")
        print(f"  MAE :  fz={mae_tz:.6f}, fx={mae_tx:.6f}, fy={mae_ty:.6f}")
    else:
        print("\n[Dataset Avg Error]  skipped (no ground-truth found for any sample)")

    print("[done] inference results at", outdir)


if __name__ == "__main__":
    main()