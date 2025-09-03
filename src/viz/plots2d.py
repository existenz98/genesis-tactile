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
2D visualization util
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_force_maps(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    fx_mpa: np.ndarray,
    fy_mpa: np.ndarray,
    fz_mpa: np.ndarray,
    out_prefix: Path,
    title: Optional[str] = None,
    cmap: str = "viridis",
    quiver_stride: int = 3,
    fz_clim: Optional[Tuple[float, float]] = None,
    dpi: int = 180,
) -> Tuple[Path, Path]:
    """
    Save:
      - heatmap of fz (MPa)
      - quiver of (fx, fy) overlaid on |fz| background
    Parameters
    ----------------
    fz_clim : optional (vmin, vmax) tuple to fix the display range of |fz|.
              If None, auto-scales.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Sanity check, verify magnitudes.  should align with --tau_shear_mpa, --tau_torque_mpa, --fz_peak_mpa
    print("max |fx,fy,fz| =", np.max(np.abs(fx_mpa)), np.max(np.abs(fy_mpa)), np.max(np.abs(fz_mpa)), " mpa")

    # 1) fz heatmap plot
    fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im1 = ax1.imshow(
        fz_mpa, origin="lower", extent=[X_mm.min(), X_mm.max(), Y_mm.min(), Y_mm.max()],
        cmap=cmap, interpolation="bilinear"
    )
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("t_z (MPa)")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    if title:
        ax1.set_title(f"{title} — t_z heatmap")
    fz_path = out_prefix.with_suffix(".fz.png")
    fig1.savefig(fz_path, dpi=dpi)
    plt.close(fig1)

    # 2) quiver plot on top of |fz| background
    fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bg = np.abs(fz_mpa)
    vmax = np.percentile(bg, 99) if np.any(bg > 0) else 1.0
    #if vmax<0.015:
    #    vmax = 1.0
    if fz_clim is not None:
        vmin, vmax = float(fz_clim[0]), float(fz_clim[1])
    im2 = ax2.imshow(
        bg, origin="lower", extent=[X_mm.min(), X_mm.max(), Y_mm.min(), Y_mm.max()],
        cmap="Greys", vmin=0.0, vmax=vmax, interpolation="bilinear"
    )
    # decimate for quiver
    sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
    ax2.quiver(
        X_mm[sl], Y_mm[sl],
        fx_mpa[sl], fy_mpa[sl],
        color="tab:blue", angles="xy", scale_units="xy", scale=None, width=0.003
    )
    #print (fx_mpa)
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("|t_z| (MPa)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    if title:
        ax2.set_title(f"{title} — (t_x, t_y) quiver")
    quiv_path = out_prefix.with_name(out_prefix.name + ".quiver.png")
    fig2.savefig(quiv_path, dpi=dpi)
    plt.close(fig2)

    return fz_path, quiv_path




def save_disp_maps(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    ux_mm: np.ndarray,
    uy_mm: np.ndarray,
    uz_mm: np.ndarray,
    out_prefix: Path,
    title: Optional[str] = None,
    cmap: str = "viridis",
    quiver_stride: int = 2,
    dpi: int = 180,
) -> Tuple[Path, Path]:
    """
    Save:
      - heatmap of u_z (mm)
      - quiver of (u_x, u_y) overlaid on |u_z| background
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) u_z heatmap
    fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im1 = ax1.imshow(
        uz_mm, origin="lower",
        extent=[X_mm.min(), X_mm.max(), Y_mm.min(), Y_mm.max()],
        cmap=cmap, interpolation="bilinear"
    )
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("u_z (mm)")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    if title:
        ax1.set_title(f"{title} — u_z heatmap")
    png_hm = out_prefix.with_suffix(".uz.png")
    fig1.savefig(png_hm, dpi=dpi)
    plt.close(fig1)

    # 2) quiver (u_x, u_y) on |u_z| background with explicit scaling
    fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bg = np.abs(uz_mm)
    vmax = np.percentile(bg, 99) if np.any(bg > 0) else 1.0
    im2 = ax2.imshow(
        bg, origin="lower",
        extent=[X_mm.min(), X_mm.max(), Y_mm.min(), Y_mm.max()],
        cmap="Greys", vmin=0.0, vmax=vmax, interpolation="bilinear"
    )

    sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
    mag = np.hypot(ux_mm, uy_mm)
    mag_max = float(mag.max()) if np.any(mag) else 1.0
    target = 0.05 * min(float(X_mm.max() - X_mm.min()), float(Y_mm.max() - Y_mm.min()))
    s = target / (mag_max + 1e-12)
    ax2.quiver(
        X_mm[sl], Y_mm[sl],
        (ux_mm * s)[sl], (uy_mm * s)[sl],
        color="tab:blue", angles="xy", scale_units="xy", scale=1.0, width=0.003
    )
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("|u_z| (mm)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    if title:
        ax2.set_title(f"{title} — (u_x, u_y) quiver (scaled ×{s:.2e})")
    png_qv = out_prefix.with_name(out_prefix.name + ".u_quiver.png")
    fig2.savefig(png_qv, dpi=dpi)
    plt.close(fig2)

    return png_hm, png_qv


