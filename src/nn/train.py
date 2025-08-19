from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import load_config, Cfg
from .data import FlowForceDataset, read_manifest_ids
from .models.unet_basic import UNetBasic
from .losses import compose_loss


def set_seed(s: int = 123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders(cfg: Cfg):
    all_ids = read_manifest_ids(cfg.dataset.manifest)

    # simple training vs validation split: 90/10,  no randomization
    n = len(all_ids)
    n_val = max(1, int(0.1 * n))
    val_ids = all_ids[:n_val]
    tr_ids  = all_ids[n_val:]

    ds_tr = FlowForceDataset(cfg.dataset, cfg.scaling, cfg.augment, tr_ids, is_train=True, seed=cfg.train.seed)
    ds_va = FlowForceDataset(cfg.dataset, cfg.scaling, cfg.augment, val_ids, is_train=False, seed=cfg.train.seed+1)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False,
                       num_workers=cfg.train.num_workers, pin_memory=True, drop_last=False)
    return dl_tr, dl_va


def build_model(cfg: Cfg):
    if cfg.model.type == "unet_basic":
        return UNetBasic(in_ch=cfg.model.in_ch, out_ch=cfg.model.out_ch,
                         base_ch=cfg.model.channels, norm=cfg.model.norm,
                         nonneg_tz_head=cfg.model.nonneg_tz_head)
    else:
        raise NotImplementedError(f"Model type {cfg.model.type}")
    
def inverse_target_scaling(y, cfg: Cfg):
    """
    For output metrics in raw space.
    """
    mode = cfg.scaling.target_mode.lower()
    if mode == "linear":
        scales = torch.tensor(cfg.scaling.force_scale, dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
        return y * scales
    elif mode == "log":
        alpha = torch.tensor(cfg.scaling.log_alpha, dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
        return torch.sign(y) * alpha * (torch.exp(torch.abs(y)) - 1.0)
    else:
        raise ValueError


def train_one_epoch(model, dl, optimizer, scaler, cfg: Cfg, device):
    model.train()
    losses = []
    for batch in tqdm(dl, desc="train", leave=False):
        x = batch["x"].to(device)         # (B,2,H,W)
        y = batch["y"].to(device)         # (B,3,H,W)

        optimizer.zero_grad(set_to_none=True)
        if cfg.train.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss, parts = compose_loss(
                    y_hat, y,
                    w_mse=cfg.loss.mse, w_rel=cfg.loss.rel_l1, rel_eps=cfg.loss.rel_epsilon,
                    w_tv=cfg.loss.tv, w_nonneg=cfg.loss.nonneg_tz,
                    w_fric=cfg.loss.friction_cone_weight, mu=cfg.loss.friction_mu
                )
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_hat = model(x)
            loss, parts = compose_loss(
                y_hat, y,
                w_mse=cfg.loss.mse, w_rel=cfg.loss.rel_l1, rel_eps=cfg.loss.rel_epsilon,
                w_tv=cfg.loss.tv, w_nonneg=cfg.loss.nonneg_tz,
                w_fric=cfg.loss.friction_cone_weight, mu=cfg.loss.friction_mu
            )
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_one_epoch(model, dl, cfg: Cfg, device):
    model.eval()
    losses = []
    for batch in tqdm(dl, desc="valid", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = model(x)
        loss, parts = compose_loss(
            y_hat, y,
            w_mse=cfg.loss.mse, w_rel=cfg.loss.rel_l1, rel_eps=cfg.loss.rel_epsilon,
            w_tv=cfg.loss.tv, w_nonneg=cfg.loss.nonneg_tz,
            w_fric=cfg.loss.friction_cone_weight, mu=cfg.loss.friction_mu
        )
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_tr, dl_va = make_dataloaders(cfg)
    model = build_model(cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.cuda.amp.GradScaler() if (cfg.train.amp and device.type == "cuda") else None

    ckpt_dir = Path(cfg.train.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); best_path = ckpt_dir / "best.pth"

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, cfg, device)
        va_loss = eval_one_epoch(model, dl_va, cfg, device)
        scheduler.step()

        print(f"[epoch {epoch:03d}] train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val or not cfg.train.save_best_only:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "cfg": vars(cfg)}, best_path)
            print(f"  saved: {best_path}")

    print("[done] best val:", best_val)

if __name__ == "__main__":
    main()
