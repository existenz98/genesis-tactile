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


from __future__ import annotations
from typing import Tuple, List, Dict
from pathlib import Path
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DatasetCfg, ScalingCfg, AugmentCfg


# ---------- resizing helpers -----------

def resize_flow(vx: np.ndarray, vy: np.ndarray, new_w: int, new_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize 2D flow field to (new_w, new_h), and scale vectors properly.
    """
    h, w = vx.shape[:2]
    if (w, h) == (new_w, new_h):
        return vx, vy
    sx = new_w / float(w)   # scaling factor
    sy = new_h / float(h)
    vx_r = cv2.resize(vx, (new_w, new_h), interpolation=cv2.INTER_AREA) * sx
    vy_r = cv2.resize(vy, (new_w, new_h), interpolation=cv2.INTER_AREA) * sy
    return vx_r, vy_r

def resize_map(m: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resize scalar map with bilinear.
    """
    if m.shape[1] == new_w and m.shape[0] == new_h:
        return m
    return cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# ---------- data augmentations (on flow maps) ----------
def augment_flow(vx: np.ndarray, vy: np.ndarray, aug: AugmentCfg, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if not aug.enabled:
        return vx, vy
    H, W = vx.shape
    vxs, vys = vx.copy(), vy.copy()

    # Choose a small subset of effects per sample
    # 50% AWGN
    if rng.uniform() < 0.5:
        std = rng.uniform(aug.awgn_std_min, aug.awgn_std_max)
        vxs += rng.normal(0.0, std, size=(H, W)).astype(np.float32)
        vys += rng.normal(0.0, std, size=(H, W)).astype(np.float32)

    # 50% low-frequency bias
    if rng.uniform() < 0.5 and (aug.bias_lf_amp_max > 0):
        gh, gw = aug.bias_lf_grid[1], aug.bias_lf_grid[0]
        bx = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
        by = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
        # upsample & gaussian blur
        bx = cv2.resize(bx, (W, H), interpolation=cv2.INTER_LINEAR)
        by = cv2.resize(by, (W, H), interpolation=cv2.INTER_LINEAR)
        ax = rng.uniform(aug.bias_lf_amp_min, aug.bias_lf_amp_max)
        ay = rng.uniform(aug.bias_lf_amp_min, aug.bias_lf_amp_max)
        vxs += ax * cv2.GaussianBlur(bx, (0,0), sigmaX=3.0)
        vys += ay * cv2.GaussianBlur(by, (0,0), sigmaX=3.0)

    # 50% gain jitter
    if rng.uniform() < 0.5:
        g = 1.0 + rng.uniform(aug.gain_min, aug.gain_max)
        vxs *= g; vys *= g

    # 30% constant drift
    if rng.uniform() < 0.3:
        dx = rng.uniform(-aug.drift_abs_max, aug.drift_abs_max)
        dy = rng.uniform(-aug.drift_abs_max, aug.drift_abs_max)
        vxs += dx; vys += dy

    # 30% blur
    if rng.uniform() < 0.3 and (aug.blur_sigma_max > 0):
        s = rng.uniform(aug.blur_sigma_min, aug.blur_sigma_max)
        if s > 0:
            vxs = cv2.GaussianBlur(vxs, (0,0), s)
            vys = cv2.GaussianBlur(vys, (0,0), s)

    # Dropout rectangles
    if rng.uniform() < aug.dropout_prob and aug.dropout_boxes > 0:
        for _ in range(aug.dropout_boxes):
            w = min(aug.dropout_box_wh[0], W)
            h = min(aug.dropout_box_wh[1], H)
            x0 = rng.integers(0, max(1, W - w))
            y0 = rng.integers(0, max(1, H - h))
            vxs[y0:y0+h, x0:x0+w] = 0.0
            vys[y0:y0+h, x0:x0+w] = 0.0

    return vxs, vys


# ---------- dataset ----------

class FlowForceDataset(Dataset):
    """
    Reads samples listed in manifest.csv.
    Each folder is a sample, must contain:
      - flow_f32.npz : vx, vy (float32)
      - force_top.npz: tx_mpa, ty_mpa, tz_mpa (float32)
    """

    def __init__(self,
                 dataset: DatasetCfg,
                 scaling: ScalingCfg,
                 augment: AugmentCfg,
                 split_ids: List[str],
                 is_train: bool = True,
                 seed: int = 123) -> None:
        self.ds = dataset
        self.sc = scaling
        self.aug = augment if is_train else AugmentCfg(enabled=False)
        self.split_ids = split_ids
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

        # target resolution
        self.target_wh = None
        if self.ds.resize_to is not None:
            self.target_wh = (int(self.ds.resize_to[0]), int(self.ds.resize_to[1]))  # (W, H)
    
    def __len__(self) -> int:
        return len(self.split_ids)
    
    def _load_npz(self, p: Path) -> Dict[str, np.ndarray]:
        with np.load(p) as d:
            return {k: d[k] for k in d.files if isinstance(d[k], np.ndarray)}
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.split_ids[idx]
        sid = sid.zfill(8)     # e.g. '12' to '00000012'
        root = Path(self.ds.root) / sid
        flow_p  = root / self.ds.flow_file
        force_p = root / self.ds.force_file

        # Load flow
        f = self._load_npz(flow_p)
        vx, vy = f["vx"].astype(np.float32), f["vy"].astype(np.float32)
        Hf, Wf = vx.shape       # data's resolution

        # Resize flow (and scale) to target
        #TODO don't need to resize if Hf,Wf==target w,h
        if self.target_wh is not None:
            vx, vy = resize_flow(vx, vy, self.target_wh[0], self.target_wh[1])

        # flip y axis, and negate the value
        if self.ds.flip_y:
            vy = -vy[::-1, :].copy()

        # Augment (train only)
        vx, vy = augment_flow(vx, vy, self.aug, self.rng)

        # Global scaling (constant over dataset)
        vx = (vx - self.sc.flow_mean) / max(self.sc.flow_scale, 1e-6)
        vy = (vy - self.sc.flow_mean) / max(self.sc.flow_scale, 1e-6)

        # Stack to 2-ch tensor
        x = np.stack([vx, vy], axis=0)  # (2, H, W)

        # Load force (tz, tx, ty) as training GT
        g = self._load_npz(force_p)
        tx = g["fx_mpa"].astype(np.float32)
        ty = g["fy_mpa"].astype(np.float32)
        tz = g["fz_mpa"].astype(np.float32)

        # Resize force maps to target resolution (no magnitude scaling)
        if self.target_wh is not None:
            Wt, Ht = self.target_wh
            tx = resize_map(tx, Wt, Ht)
            ty = resize_map(ty, Wt, Ht)
            tz = resize_map(tz, Wt, Ht)

        # Pack target order [tz, tx, ty]
        t = np.stack([tz, tx, ty], axis=0)

        # Apply target scaling
        if self.sc.target_mode.lower() == "linear":
            scales = np.array(self.sc.force_scale, dtype=np.float32).reshape(-1, 1, 1)
            y = t / np.maximum(scales, 1e-6)
        elif self.sc.target_mode.lower() == "log":
            alpha = np.array(self.sc.log_alpha, dtype=np.float32).reshape(-1, 1, 1)
            y = np.sign(t) * np.log1p(np.abs(t) / np.maximum(alpha, 1e-6))
        else:
            raise ValueError(f"Unknown target_mode: {self.sc.target_mode}")

        sample = {
            "id": sid,
            "x": torch.from_numpy(x),       # (2,H,W)
            "y": torch.from_numpy(y),       # (3,H,W) scaled/logged
            "t_raw": torch.from_numpy(t),   # (3,H,W) unscaled (for metrics)
        }
        return sample
    

# ---------- manifest utilities ----------

def read_manifest_ids(manifest_path: str | Path) -> List[str]:
    ids = []
    with open(manifest_path, "r", newline="") as f:
        r = csv.DictReader(f)
        id_key = "id" if "id" in r.fieldnames else r.fieldnames[0]
        ok_key = "ok" if "ok" in r.fieldnames else None
        for row in r:
            if ok_key and row[ok_key] not in ("1", "True", "true", "yes", "y"):
                continue
            ids.append(row[id_key])
    return ids
