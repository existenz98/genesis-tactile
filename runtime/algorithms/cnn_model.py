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
CNN-based force-field solver (adapter to `src/nn`).

- Input: dense optical flow (vy, vx) in image coordinates (float32, pixels per frame).
- Preprocess: resize flow to model size (need to pay attention to: vector scaling, flip-y as in training).
- Inference: model specified in src/nn config, runs on device (cpu/cuda).
- Postprocess: inverse target scaling -> MPa; map to runtime schema: p=tz, tau=(tx, ty).
- Grid metric: cell_mm computed from the physical size of the flow field and model output size.

It does not depend on dataset/loader, uses the following:
  - src.nn.config.load_config
  - src.nn.data.resize_flow
  - src.nn.infer.inverse_target_scaling
  - src.nn.models.unet_basic.UNetBasic
"""

from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import dummy
from pathlib import Path
from pyexpat import model
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2

import torch

# Make project root importable so `src.nn` can be imported when running from `runtime/`
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo_root>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.nn.infer import inverse_target_scaling as nn_inverse_target_scaling
from src.nn.data import resize_flow as nn_resize_flow
from src.nn.config import load_config as nn_load_config
from src.nn.config import Cfg as NNCfg
from src.nn.models.unet_basic import UNetBasic

from runtime.config.settings import CnnConfig
from runtime.utils.prof import Prof


class CnnForceSolver:
    """Wraps the UNet model and exposes a solve_from_flow(vy, vx, mm_per_px) API."""

    def __init__(self, cfg: CnnConfig) -> None:
        self.cfg = cfg

        # Load nn YAML (scaling & model meta)
        self.nn_cfg: NNCfg = nn_load_config(self.cfg.model_cfg)

        # Decide input (W,H)
        self.model_wh: Tuple[int, int]
        if self.cfg.force_input_size is not None:
            self.model_wh = (int(self.cfg.force_input_size[0]), int(self.cfg.force_input_size[1]))
        elif self.nn_cfg.dataset.resize_to is not None:
            w, h = self.nn_cfg.dataset.resize_to
            self.model_wh = (int(w), int(h))
        else:
            # Fallback: infer from checkpoint or require explicit
            raise RuntimeError("Model input size not specified. Set cfg.force_input_size or nn_cfg.dataset.resize_to.")
        print(f"CnnForceSolver.self.model_wh={self.model_wh}")

        # flip-y policy (match training)
        self.flip_y: bool = (
            bool(self.cfg.flip_y_override)
            if self.cfg.flip_y_override is not None
            else bool(self.nn_cfg.dataset.flip_y)
        )

        # Device & model
        device_str = self.cfg.device
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print("[cnn_model] CUDA not available, falling back to CPU")
            device_str = "cpu"
        else:
            print(f"[cnn_model] CUDA is available, device='{device_str}'")
        self.device = torch.device(device_str)

        torch.backends.cudnn.benchmark = True      # pick fastest conv alg for fixed (NCHW/NHWC,H,W)
        torch.backends.cudnn.allow_tf32 = True     # if Ampere+, enables fast TF32 convs for fp32

        # profiling
        self.prof = Prof(enable=True, report_every=20)
        if self.device.type == "cuda":
            self._evt_start = torch.cuda.Event(enable_timing=True)
            self._evt_end   = torch.cuda.Event(enable_timing=True)

        # Build model
        self.model = self._build_model(self.nn_cfg).to(self.device).eval()
        # Optional fp16
        self._amp_dtype = torch.float16 if (self.cfg.use_half and self.device.type == "cuda") else torch.float32

        self.model = self.model.to(memory_format=torch.channels_last)   # Channels-last memory format

        # Load checkpoint
        ckpt_path = Path(self.cfg.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=self.device)
        # Support either {"model": state_dict} or pure state_dict
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[cnn_model] Warning: missing={missing}, unexpected={unexpected}")

        # Try torch.compile to cut launch overhead (need PyTorch 2+)
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        
        # Warmup
        # Warm up to trigger compilation & autotuning
        print(f"[cnn_model] Warming up model on {self.device} ...")
        Wm, Hm = self.model_wh
        vx_r = np.zeros((Hm, Wm), np.float32)
        vy_r = np.zeros((Hm, Wm), np.float32)
        x = np.stack([vx_r, vy_r], axis=0)[None, ...]  # (1,2,H,W)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            for _ in range(3):
                x_t = torch.from_numpy(x).to(self.device, dtype=self._amp_dtype, non_blocking=True)
                x_t = x_t.contiguous(memory_format=torch.channels_last)
                _ = self.model(x_t)
        print(f"[cnn_model] Ready. Using device: {self.device}, model input (W,H)={self.model_wh}, flip_y={self.flip_y}")

        # Precompute normalization constants
        sc = self.nn_cfg.scaling
        self.flow_scale = float(sc.flow_scale) if sc.flow_scale is not None else 1.0
        self.flow_mean = float(sc.flow_mean) if sc.flow_mean is not None else 0.0



    def _build_model(self, cfg: NNCfg) -> torch.nn.Module:
        if cfg.model.type != "unet_basic":
            raise NotImplementedError(f"Unsupported model.type={cfg.model.type}")
        return UNetBasic(
            in_ch=cfg.model.in_ch,
            out_ch=cfg.model.out_ch,
            base_ch=cfg.model.channels,
            norm=cfg.model.norm,
            nonneg_tz_head=cfg.model.nonneg_tz_head,
        )

    def _block_reduce_avg(self, m: np.ndarray, Wd: int, Hd: int) -> np.ndarray:
        """
        Average-pool 2D map m from (Hm,Wm) -> (Hd,Wd). Prefer exact integer factors; otherwise fall back to INTER_AREA.
        """
        Hm, Wm = m.shape
        fx = Wm / float(Wd)
        fy = Hm / float(Hd)
        # If both factors are (close to) integers, do manual pooling for exact averages
        if abs(fx - round(fx)) < 1e-6 and abs(fy - round(fy)) < 1e-6:
            fx_i, fy_i = int(round(fx)), int(round(fy))
            # reshape => (Hd, fy, Wd, fx) then mean over fy,fx
            m = m[:Hd*fy_i, :Wd*fx_i]
            m = m.reshape(Hd, fy_i, Wd, fx_i).mean(axis=(1,3))
            return m.astype(np.float32)
        # Otherwise, use area interpolation (good for downsampling)
        return cv2.resize(m.astype(np.float32), (Wd, Hd), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _downsample_to(self, tz: np.ndarray, tx: np.ndarray, ty: np.ndarray,
                    Wd: int, Hd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Downsample scalar/ vector component maps to (Hd,Wd).
        Just simple average (no magnitude scaling).
        """
        tz_d = self._block_reduce_avg(tz, Wd, Hd)  # scalar pressure
        tx_d = self._block_reduce_avg(tx, Wd, Hd)  # vector component
        ty_d = self._block_reduce_avg(ty, Wd, Hd)
        return tz_d, tx_d, ty_d



    # --------------------- Public API ---------------------

    def solve_from_flow(self, vy: np.ndarray, vx: np.ndarray, mm_per_px: float) -> Dict[str, Any]:
        """
        Args:
            vy, vx: dense flow in pixels (image coords, +y downward). float32, shape (H, W).
            mm_per_px: physical scale at the flow's resolution (mm per pixel of vy/vx arrays).

        Returns:
            dict with keys: 'grid', 'p', 'tau'
        """
        assert vy.shape == vx.shape, "vy and vx must have same shape"
        H_src, W_src = vy.shape

        # Resize flow to model size; scale vectors appropriately
        Wm, Hm = self.model_wh
        vx_r, vy_r = nn_resize_flow(vx.astype(np.float32), vy.astype(np.float32), new_w=Wm, new_h=Hm)

        # vertical flip/sign to match training convention (from camera bottom view to top view)
        if self.flip_y:
            vy_r = -vy_r[::-1, :].copy()

        # Normalize inputs (global constants from training stats)
        vx_n = (vx_r - self.flow_mean) / max(self.flow_scale, 1e-6)
        vy_n = (vy_r - self.flow_mean) / max(self.flow_scale, 1e-6)

        # Torch tensor (1,2,H,W)
        x = np.stack([vx_n, vy_n], axis=0)[None, ...]  # (1,2,Hm,Wm)

        # Inference
        #with torch.inference_mode():
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            y = None

            #x_t = torch.from_numpy(x).to(self.device, dtype=self._amp_dtype)
            x_t = torch.from_numpy(x).to(self.device, dtype=self._amp_dtype, non_blocking=True)
            x_t = x_t.contiguous(memory_format=torch.channels_last)
            # TODO:  Reuse tensors; no per-frame .from_numpy().to(device)

            #if self.device.type == "cuda":
            #    self._evt_start.record()

            with self.prof("[cnn] model()"):
                y = self.model(x_t)  # (1,3,Hm,Wm) order [tz, tx, ty]

            #if self.device.type == "cuda":
            #    self._evt_end.record()
            #    torch.cuda.synchronize()
            #    ms = self._evt_start.elapsed_time(self._evt_end)
            #    print(f"[cnn] forward {ms:.3f} ms")

            # De-scale to raw units (MPa) using the same function as training
            y_raw = nn_inverse_target_scaling(y, self.nn_cfg, device=self.device)  # (1,3,Hm,Wm)
            y_np = y_raw.squeeze(0).detach().float().cpu().numpy()

        self.prof.tick()

        tz, tx, ty = y_np[0], y_np[1], y_np[2]  # MPa

        # ---- Downsample to target grid if requested ----
        Wm, Hm = self.model_wh
        if self.cfg.out_grid_wh is not None:
            Wd, Hd = int(self.cfg.out_grid_wh[0]), int(self.cfg.out_grid_wh[1])
            tz, tx, ty = self._downsample_to(tz, tx, ty, Wd, Hd)
            W_out, H_out = Wd, Hd
        else:
            W_out, H_out = Wm, Hm

        # Grid metrics: keep physical extent same as the source flow field
        width_mm  = W_src * float(mm_per_px)
        height_mm = H_src * float(mm_per_px)
        cell_mm_x = width_mm / float(W_out)
        cell_mm_y = height_mm / float(H_out)
        # For our viewers/SDK we use one scalar; assume square pixels → average
        cell_mm = float((cell_mm_x + cell_mm_y) * 0.5)
        cell_px = float(W_src / float(W_out))  # source-pixels per model-cell (for reference)

        result: Dict[str, Any] = {
            "grid": {"H": H_out, "W": W_out, "cell_mm": cell_mm, "cell_px": cell_px},
            "p": tz.astype(np.float32),                        # MPa
            "tau": {"tx": tx.astype(np.float32), "ty": ty.astype(np.float32)},  # MPa
            # 'u_mm' could be added if your model predicts displacement
        }
        return result
