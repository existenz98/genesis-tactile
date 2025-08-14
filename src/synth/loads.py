"""
Synthetic surface force field generator for 6D tactile FEM.

All forces here are *surface tractions* on the top face, in MPa
(i.e., N/mm^2). Positive tz means pushing *into* the silicone.

Coordinate convention:
- Physical size: Lx_mm (x-direction), Ly_mm (y-direction).
- Grid shape: (Ny, Nx). X increases to the right, Y increases upward.
- Origin (0,0) at the top-left corner; use (cx_mm, cy_mm) to place patterns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import numpy as np

@dataclass
class ForceField:
    X_mm: np.ndarray  # shape (Ny, Nx)
    Y_mm: np.ndarray  # shape (Ny, Nx)
    fx_mpa: np.ndarray  # shape (Ny, Nx)
    fy_mpa: np.ndarray  # shape (Ny, Nx)
    fz_mpa: np.ndarray  # shape (Ny, Nx)
    meta: Dict

    def save_npz(self, path: Path) -> None:
        """store multiple NumPy arrays in a single compressed archive (.npz)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X_mm=self.X_mm,
            Y_mm=self.Y_mm,
            fx_mpa=self.fx_mpa,
            fy_mpa=self.fy_mpa,
            fz_mpa=self.fz_mpa,
            unit="MPa",
            meta_json=json.dumps(self.meta, ensure_ascii=False),
        )

def make_surface_grid(Lx_mm: float, Ly_mm: float, Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a top-surface grid (Ny, Nx) with physical coordinates in mm."""
    x = np.linspace(0.0, Lx_mm, Nx)
    y = np.linspace(0.0, Ly_mm, Ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def _gaussian(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, sigma_mm: float) -> np.ndarray:
    """Normalized 2D Gaussian centered at (cx, cy) mm with std=sigma_mm."""
    rx = (X - cx)
    ry = (Y - cy)
    r2 = (rx * rx + ry * ry)
    g = np.exp(-0.5 * r2 / (sigma_mm ** 2))
    g /= (2.0 * np.pi * sigma_mm * sigma_mm)  # integrate to ~1 over R^2 (continuous)
    return g

def pressure_patch(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    cx_mm: float,
    cy_mm: float,
    fz_peak_mpa: float,
    sigma_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Axis-symmetric pressure (normal force) patch: fz = fz_peak * Gaussian, fx=fy=0.
    """
    base = _gaussian(X_mm, Y_mm, cx_mm, cy_mm, sigma_mm)
    # Scale, so peak is fz_peak_mpa (_gaussian is normalized by integral instead of peak value)
    # Here we set the amplitude such that the center approaches fz_peak_mpa times a constant factor.
    # For intuitive control, we simply rescale to have max == fz_peak_mpa.
    fz = base / base.max() * fz_peak_mpa
    fx = np.zeros_like(fz)
    fy = np.zeros_like(fz)
    return fx, fy, fz

def shear_patch(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    cx_mm: float,
    cy_mm: float,
    tau_shear_mpa: float,
    sigma_mm: float,
    dir_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shear (tangential force) patch with a Gaussian footprint. Direction is given by dir_deg.
    """
    base = _gaussian(X_mm, Y_mm, cx_mm, cy_mm, sigma_mm)
    amp = base / base.max() * tau_shear_mpa
    theta = np.deg2rad(dir_deg)
    fx = amp * np.cos(theta)
    fy = amp * np.sin(theta)
    fz = np.zeros_like(amp)
    return fx, fy, fz

def torque_patch(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    cx_mm: float,
    cy_mm: float,
    tau_torque_mpa: float,
    sigma_mm: float,
    inner_mm: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Torsional load around (cx,cy) by a tangential traction field.
    The magnitude grows with radius then decays by a Gaussian envelope.

    tau(r) = tau_torque * (r / r0) * exp(-0.5 * r^2/sigma^2), with r0 chosen so that
    max(tau) ≈ tau_torque. inner_mm can set a 'dead zone' near the center.
    """
    rx = X_mm - cx_mm
    ry = Y_mm - cy_mm
    r = np.hypot(rx, ry)
    r0 = max(sigma_mm, 1e-6)    # Avoid division by zero
    env = np.exp(-0.5 * (r ** 2) / (sigma_mm ** 2))
    raw = (r / r0) * env

    # Normalize so that max ~= 1, then scale to tau_torque_mpa
    if raw.max() > 0:
        raw = raw / raw.max()
    amp = raw * tau_torque_mpa

    # inner dead zone
    if inner_mm > 0:
        amp = np.where(r < inner_mm, 0.0, amp)

    # Tangential unit vector: [-sinθ, cosθ]
    with np.errstate(invalid="ignore", divide="ignore"):
        tx = np.where(r > 0, -ry / r, 0.0)
        ty = np.where(r > 0,  rx / r, 0.0)

    fx = amp * tx
    fy = amp * ty
    fz = np.zeros_like(amp)
    return fx, fy, fz


# Experimental: add a “ball press” profile closer to Hertzian contact
# make the normal press look like a rigid sphere on a soft half-space (instead of Gaussian)
# map the ball radius to the footprint:
def hertz_normal_patch(
    X_mm, Y_mm, cx_mm, cy_mm, R_mm: float, delta_mm: float, p0_mpa: Optional[float] = None
):
    """
    Approximate Hertz contact: pressure p(r) = p0 * sqrt(1 - (r/a)^2) for r<=a, else 0,
    where a = sqrt(R*delta). You can set p0 (MPa) explicitly, or normalize so max=p0.
    """
    rx, ry = X_mm - cx_mm, Y_mm - cy_mm
    r = np.hypot(rx, ry)
    a = np.sqrt(max(R_mm * delta_mm, 1e-9))
    inside = r <= a
    p = np.zeros_like(r)
    p[inside] = np.sqrt(1.0 - (r[inside] / a) ** 2)
    if p0_mpa is not None:
        p *= p0_mpa
    else:
        p /= (p.max() + 1e-12)  # normalize peak to 1
    fx = np.zeros_like(p)
    fy = np.zeros_like(p)
    fz = p  # MPa
    return fx, fy, fz, a



def combine_fields(*fields: Tuple[np.ndarray, np.ndarray, np.ndarray]
                   )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pressure + Shear + Torque fields (fx, fy, fz)."""
    fx_total = None
    fy_total = None
    fz_total = None
    for fx, fy, fz in fields:
        if fx_total is None:
            fx_total = fx.copy()
            fy_total = fy.copy()
            fz_total = fz.copy()
        else:
            fx_total += fx
            fy_total += fy
            fz_total += fz
    return fx_total, fy_total, fz_total


def build_force_field(
    preset: str,
    Lx_mm: float,
    Ly_mm: float,
    Nx: int,
    Ny: int,
    cx_mm: Optional[float] = None,
    cy_mm: Optional[float] = None,
    sigma_mm: float = 3.0,
    fz_peak_mpa: float = 0.1,
    tau_shear_mpa: float = 0.05,
    tau_torque_mpa: float = 0.05,
    shear_dir_deg: float = 0.0,
    torque_inner_mm: float = 0.0,
    seed: Optional[int] = None,
) -> ForceField:
    """
    Build a surface traction field on a (Ny, Nx) grid over [0,Lx]x[0,Ly] mm.

    Presets:
      - 'pressure' : normal patch (fz)
      - 'shear'    : shear patch (fx,fy)
      - 'torque'   : torsional shear around center (fx,fy)
      - 'combo'    : pressure (fz) + shear + torque
    """
    rng = np.random.default_rng(seed)
    X, Y = make_surface_grid(Lx_mm, Ly_mm, Nx, Ny)
    cx = Lx_mm / 2.0 if cx_mm is None else cx_mm
    cy = Ly_mm / 2.0 if cy_mm is None else cy_mm

    if preset == "pressure":
        fx, fy, fz = pressure_patch(X, Y, cx, cy, fz_peak_mpa, sigma_mm)
    #elif preset == "hertz":
    #    fx, fy, fz, a = hertz_normal_patch(X, Y, cx, cy, R_mm=ball_radius_mm, delta_mm=indent_mm, p0_mpa=fz_peak_mpa)
    elif preset == "shear":
        fx, fy, fz = shear_patch(X, Y, cx, cy, tau_shear_mpa, sigma_mm, shear_dir_deg)
    elif preset == "torque":
        fx, fy, fz = torque_patch(X, Y, cx, cy, tau_torque_mpa, sigma_mm, inner_mm=torque_inner_mm)
    elif preset == "combo":
        f1 = pressure_patch(X, Y, cx, cy, fz_peak_mpa, sigma_mm)
        f2 = shear_patch(X, Y, cx, cy, tau_shear_mpa, sigma_mm, shear_dir_deg)
        f3 = torque_patch(X, Y, cx, cy, 0.5 * tau_torque_mpa, 1.2 * sigma_mm, inner_mm=torque_inner_mm)
        fx, fy, fz = combine_fields(f1, f2, f3)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    meta = dict(
        preset=preset,
        Lx_mm=Lx_mm,
        Ly_mm=Ly_mm,
        Nx=Nx,
        Ny=Ny,
        cx_mm=cx,
        cy_mm=cy,
        sigma_mm=sigma_mm,
        fz_peak_mpa=fz_peak_mpa,
        tau_shear_mpa=tau_shear_mpa,
        tau_torque_mpa=tau_torque_mpa,
        shear_dir_deg=shear_dir_deg,
        torque_inner_mm=torque_inner_mm,
        seed=None if seed is None else int(seed),
        unit="MPa",
        note="Surface tractions at the top face; +tz pushes into silicone.",
    )
    return ForceField(X, Y, fx, fy, fz, meta)

