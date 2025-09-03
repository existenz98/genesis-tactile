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


from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import yaml
import json

@dataclass
class DatasetCfg:
    root: str
    manifest: str
    flow_file: str = "flow_f32.npz"        # vx, vy
    force_file: str = "force_top.npz"      # tx_mpa, ty_mpa, tz_mpa
    resize_to: Optional[List[int]] = None  # [W, H]  (normally dataset is of resolution [80,60])
    stats_json: str = "dataset/stats.json" # Data statistics, computed by compute_dataset_stats.py
    flip_y: bool = True                    # need to flip flow observation because camera view from bottom,
                                           # but force is viewed from the top

@dataclass
class ScalingCfg:
    # Input (Flow value) normalization (global constants measured once)
    flow_scale: float = 1.0
    flow_mean: float = 0.0

    # Target (output Force value) scaling (global constants measured once)
    target_mode: str = "linear"          # "linear" | "log"
    force_scale: List[float] = field(default_factory=lambda:[1.0, 1.0, 1.0])  # [tz, tx, ty]
    log_alpha:  List[float] = field(default_factory=lambda:[0.1, 0.05, 0.05]) # [tz, tx, ty], only used in log mode

@dataclass
class AugmentCfg:
    """
    Data Augmentation Settings
    """
    enabled: bool = True
    # Additive white noise (std in vx, vy values (pixels))
    awgn_std_min: float = 0.0
    awgn_std_max: float = 0.15
    # Low-frequency bias amplitude (pixels), Gaussian upsampled
    bias_lf_amp_min: float = 0.0
    bias_lf_amp_max: float = 0.2
    bias_lf_grid: Tuple[int,int] = (10, 8)  # small grid to upsample from
    # Gain jitter
    gain_min: float = -0.1
    gain_max: float =  0.1
    # Constant drift vector (pixels)
    drift_abs_max: float = 0.1
    # Gaussian blur (sigma in pixels)
    blur_sigma_min: float = 0.0
    blur_sigma_max: float = 1.0
    # Dropout rectangles
    dropout_prob: float = 0.0     # probability to apply
    dropout_boxes: int = 0
    dropout_box_wh: Tuple[int,int] = (8, 8)

@dataclass
class ModelCfg:
    type: str = "unet_basic"       # or "adapter_backbone" (to be implemented)
    in_ch: int = 2                 # (vx, vy)
    out_ch: int = 3                # (tz, tx, ty)  <-- fixed order
    channels: List[int] = field(default_factory=lambda:[16, 32, 64, 128])
    norm: str = "group"            # "group" | "batch" (use group normalization for stability)
    nonneg_tz_head: bool = False   # keep linear head; enforce tz>=0 via loss

@dataclass
class LossCfg:
    mse: float = 1.0                # MSE on scaled targets (dominant term)
    rel_l1: float = 0.2             # Relative L1, make small forces meaningful.
    tv: float = 0.0                 # denoise artifacts
    nonneg_tz: float = 0.1          # PINN:  Non negative tz (soft)
    friction_cone_weight: float = 0.1   # PINN: Friction cone soft penalty
    friction_mu: float = 0.6
    rel_epsilon: float = 1e-3

@dataclass
class TrainCfg:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = False
    num_workers: int = 4
    ckpt_dir: str = "outputs/ckpt"
    log_dir: str = "outputs/logs"
    seed: int = 123
    val_every: int = 1
    grad_clip: float = 1.0
    save_best_only: bool = True

@dataclass
class Cfg:
    dataset: DatasetCfg
    scaling: ScalingCfg
    augment: AugmentCfg
    model: ModelCfg
    loss: LossCfg
    train: TrainCfg

def load_config(path: str | Path) -> Cfg:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # allow partials
    return Cfg(
        dataset=DatasetCfg(**cfg_dict["dataset"]),
        scaling=ScalingCfg(**cfg_dict["scaling"]),
        augment=AugmentCfg(**cfg_dict.get("augment", {})),
        model=ModelCfg(**cfg_dict["model"]),
        loss=LossCfg(**cfg_dict["loss"]),
        train=TrainCfg(**cfg_dict["train"]),
    )

def save_stats_json(path: str | Path, stats: Dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)

def load_stats_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

