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

def compose_loss(y_hat, y,
                 w_mse=1.0, w_rel=0.0, rel_eps=1e-3,
                 w_tv=0.0, w_nonneg=0.0,
                 w_fric=0.0, mu=0.6):
    loss = 0.0
    l_mse = mse_loss(y_hat, y) * w_mse; loss += l_mse
    l_rel = rel_l1_loss(y_hat, y, rel_eps) * w_rel; loss += l_rel
    l_tv  = tv_loss(y_hat, w_tv); loss += l_tv
    l_nn  = nonneg_tz_penalty(y_hat, w_nonneg); loss += l_nn
    l_fc  = friction_cone_penalty(y_hat, mu, w_fric); loss += l_fc
    parts = dict(mse=float(l_mse.item()), rel=float(l_rel.item()),
                 tv=float(l_tv.item()), nonneg=float(l_nn.item()), fric=float(l_fc.item()))
    return loss, parts
