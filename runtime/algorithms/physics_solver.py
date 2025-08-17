from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from runtime.config.settings import PhysicsConfig

 
class PhysicsSolver:
    """
    Baseline physics solver (single flow field):
      p ≈ k_n * w, where w ≈ h * div(u_hat)
      tau ≈ (shear_gain) * u_hat + (slope_gain) * ∇w
    The solver:
      - robustly downsamples dense flow to a coarse grid (weighted / Huber / median),
      - smooths, then computes divergence via central differences on the coarse grid,
      - outputs pressure map and shear vectors on that coarse grid.
    """

    def __init__(self, cfg: PhysicsConfig):
        self.cfg = cfg

    # ---------- robust block pooling ----------
    def _block_reduce(self, vy: np.ndarray, vx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Weighted/robust pooling from dense flow (pixels) to coarse grid (in pixels).
        Returns vy_ds, vx_ds and the block size used.
        """
        H, W = vy.shape
        B = self.cfg.ds_block
        h = H // B
        w = W // B
        if h == 0 or w == 0:
            return vy.astype(np.float32), vx.astype(np.float32), B

        vyC = vy[:h*B, :w*B].reshape(h, B, w, B).transpose(0, 2, 1, 3).reshape(h, w, B*B)
        vxC = vx[:h*B, :w*B].reshape(h, B, w, B).transpose(0, 2, 1, 3).reshape(h, w, B*B)

        pool = self.cfg.ds_pool.lower()
        if pool == "mean":
            vy_ds = vyC.mean(axis=2)
            vx_ds = vxC.mean(axis=2)
        elif pool == "median":
            vy_ds = np.median(vyC, axis=2)
            vx_ds = np.median(vxC, axis=2)
        else:
            # Huber weights around the block median (robust to outliers)
            vy_med = np.median(vyC, axis=2, keepdims=True)
            vx_med = np.median(vxC, axis=2, keepdims=True)
            r = np.sqrt((vyC - vy_med)**2 + (vxC - vx_med)**2)  # residual magnitude
            s = max(self.cfg.huber_sigma_px, 1e-6)
            wts = 1.0 / (1.0 + (r / s)**2)                      # Cauchy-like; simple, smooth
            # Avoid all-zero weights
            wsum = wts.sum(axis=2, keepdims=True) + 1e-6
            vy_ds = (wts * vyC).sum(axis=2) / wsum.squeeze(2)
            vx_ds = (wts * vxC).sum(axis=2) / wsum.squeeze(2)

        return vy_ds.astype(np.float32), vx_ds.astype(np.float32), B
    
    
    # ---------- central-difference gradients on coarse grid ----------

    @staticmethod
    def _gauss_blur(img: np.ndarray, sigma_cells: float) -> np.ndarray:
        if sigma_cells <= 1e-6:
            return img
        # Convert sigma in cells to an odd kernel size
        k = int(round(sigma_cells * 6)) | 1
        return cv2.GaussianBlur(img, (k, k), sigma_cells)
    
    @staticmethod
    def _grad_centered(img: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Central differences with proper physical spacing.
        spacing: physical spacing between neighboring grid nodes (in mm).
        """
        gy = np.zeros_like(img, dtype=np.float32)
        gx = np.zeros_like(img, dtype=np.float32)
        # x-gradient
        gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / (2.0 * spacing)
        gx[:, 0]    = (img[:, 1] - img[:, 0])     / max(spacing, 1e-6)
        gx[:, -1]   = (img[:, -1] - img[:, -2])   / max(spacing, 1e-6)
        # y-gradient
        gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / (2.0 * spacing)
        gy[0, :]    = (img[1, :] - img[0, :])    / max(spacing, 1e-6)
        gy[-1, :]   = (img[-1, :] - img[-2, :])  / max(spacing, 1e-6)
        return gy, gx
    
    def solve_from_dense(self, vy_px: np.ndarray, vx_px: np.ndarray) -> Dict[str, Any]:
        """
        Input:
          - vy_px, vx_px: dense forward optical flow (pixels), first -> current.
        Output dict contains:
          - grid: {'cell_px', 'cell_mm', 'H', 'W'}
          - u_mm: {'ux','uy'} coarse displacement fields in mm
          - p: pressure map on coarse grid (same HxW), arbitrary units (depends on gains)
          - tau: {'tx','ty'} shear on coarse grid
        """
        cfg = self.cfg
        # 1) Denoise tiny flows before pooling
        mag = np.sqrt(vy_px**2 + vx_px**2)
        mask = mag < cfg.min_flow_px
        vy = vy_px.copy(); vx = vx_px.copy()
        vy[mask] = 0.0; vx[mask] = 0.0

        # 2) Robust downsampling (pixels)
        vy_ds_px, vx_ds_px, B = self._block_reduce(vy, vx)

        # 3) Convert to millimeters
        s_mm = cfg.mm_per_px
        vy_ds_mm = vy_ds_px * s_mm
        vx_ds_mm = vx_ds_px * s_mm
        cell_mm = B * s_mm

        # 4) Smooth on the coarse grid
        vy_s = self._gauss_blur(vy_ds_mm, cfg.smooth_sigma_cells)
        vx_s = self._gauss_blur(vx_ds_mm, cfg.smooth_sigma_cells)

        # 5) Divergence and surface normal displacement
        #    u_hat = (ux, uy) in mm; divergence units: (mm/mm) = unitless
        ux = vx_s
        uy = vy_s
        du_y, du_x = self._grad_centered(uy, cell_mm)
        dv_y, dv_x = self._grad_centered(ux, cell_mm)
        # Here du_x is ∂uy/∂x, dv_y is ∂ux/∂y ; we need ∂ux/∂x + ∂uy/∂y
        div = dv_x + du_y                                   # ∇·u_hat

        w0 = cfg.thickness_mm * div                         # surface normal displacement (mm)
        p  = cfg.normal_gain * w0                           # pressure map (units by calibration)

        # 6) Shear with slope correction
        #    tau ≈ shear_gain * u_hat + slope_gain * ∇w
        if cfg.slope_gain != 0.0:
            dw_y, dw_x = self._grad_centered(w0, cell_mm)   # ∇w in 1 (since w in mm, /mm -> 1)
            tx = cfg.shear_gain * ux + cfg.slope_gain * dw_x
            ty = cfg.shear_gain * uy + cfg.slope_gain * dw_y
        else:
            tx = cfg.shear_gain * ux
            ty = cfg.shear_gain * uy

        Hc, Wc = p.shape
        return dict(
            grid=dict(cell_px=B, cell_mm=cell_mm, H=Hc, W=Wc),
            u_mm=dict(ux=ux, uy=uy),
            p=p.astype(np.float32),
            tau=dict(tx=tx.astype(np.float32), ty=ty.astype(np.float32)),
        )

