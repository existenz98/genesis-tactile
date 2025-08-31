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
import torch
import torch.nn.functional as F

def mse_loss(y_hat, y):
    return F.mse_loss(y_hat, y)

def rel_l1_loss(y_hat, y, eps: float = 1e-3):
    # per-pixel per-channel relative error
    denom = torch.abs(y) + eps
    return torch.mean(torch.abs((y_hat - y) / denom))

def tv_loss(y, weight: float = 1.0):
    # total variation on each channel
    loss = 0.0
    for c in range(y.size(1)):
        ch = y[:, c:c+1, :, :]
        loss += (torch.mean(torch.abs(ch[:, :, 1:, :] - ch[:, :, :-1, :])) +
                 torch.mean(torch.abs(ch[:, :, :, 1:] - ch[:, :, :, :-1])))
    return weight * loss

def nonneg_tz_penalty(y_hat, weight: float = 1.0):
    tz = y_hat[:, 0:1, ...]
    return weight * torch.mean(F.relu(-tz)**2)

def friction_cone_penalty(y_hat, mu: float = 0.6, weight: float = 1.0):
    tz = y_hat[:, 0:1, ...]
    tx = y_hat[:, 1:2, ...]
    ty = y_hat[:, 2:3, ...]
    shear = torch.sqrt(tx**2 + ty**2 + 1e-12)
    viol = F.relu(shear - mu * tz)
    return weight * torch.mean(viol**2)

def compose_loss(cfg,
                 y_hat, y,
                 w_mse=1.0, w_rel=0.0, rel_eps=1e-3,
                 w_tv=0.0, w_nonneg=0.0,
                 w_fric=0.0, mu=0.6):
    #print(w_mse, w_rel, w_tv, w_nonneg, w_fric)

    loss = 0.0

    l_mse = mse_loss(y_hat, y) * w_mse; loss += l_mse               # L2
    l_rel = rel_l1_loss(y_hat, y, rel_eps) * w_rel; loss += l_rel   # L1
    l_tv  = tv_loss(y_hat, w_tv); loss += l_tv      # total variation penalty

    # PINN losses
    scales = torch.tensor([cfg.scaling.force_scale[0],
                       cfg.scaling.force_scale[1],
                       cfg.scaling.force_scale[2]],
                      device=y_hat.device, dtype=y_hat.dtype).view(1,3,1,1)
    y_raw = y_hat * scales      # first convert to physical unit (MPa)
    l_nn  = nonneg_tz_penalty(y_raw, w_nonneg); loss += l_nn        # none-negative penalty
    l_fc  = friction_cone_penalty(y_raw, mu, w_fric); loss += l_fc  # friction-cone penalty


    parts = dict(mse=float(l_mse.item()), rel=float(l_rel.item()),
                 tv=float(l_tv.item()), nonneg=float(l_nn.item()), fric=float(l_fc.item()))
    return loss, parts
