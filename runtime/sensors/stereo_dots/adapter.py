# -----------------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later WITH LicenseRef-YF-Device-Interface-Exception
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# This file is part of the Runtime of the tactile vision platform.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE-RUNTIME-AGPL for details.
#
# Special Exception (Device Interface Exception):
#   Proprietary or separately-licensed device drivers or hardware interface
#   modules that communicate with the Runtime solely through the documented
#   TSI/plugin/IPC interfaces are not considered derivative works of the
#   Runtime by this project, and thus are not subject to the copyleft
#   obligations of the AGPL, provided they do not include or modify Runtime code.
#   See LICENSE-EXCEPTIONS for the full text.
#
# Patent Notice:
#   Except for any rights granted under the applicable open-source license,
#   no patent license is granted or implied. Users are responsible for ensuring
#   their use does not infringe third-party patents (e.g., tactile sensor
#   hardware or methods).
#
# Citation:
#   If you use this software in academic work, please cite the associated
#   publications when available.
# -----------------------------------------------------------------------------


"""
mirror-stereo vision tactile sensor
outputs 3D per-dot deformation

Usage:
python -m runtime.run_demo   \
    --source video  --input dataset/sequences/tac3d2_zigzag/video/sequence.mkv  --loop \
    --config runtime/config/runtime_config_tac3d2.yaml \
    --vis2d --vis3d
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2


from ..base import SensorAdapter
from ...tir.types import Deformation, TirMeta

from .geom import (
    rot_yaw_pitch_roll_deg, build_K, extrinsics_world_to_cam,
    reflect_point_plane, virtual_camera_from_mirror, triangulate_linear,
    reproj_error, z_cam_sign
)
from .detect import detect_bright_blobs
from .index import match_to_predictions, robust_index_half, assign_grid_to_predictions


class Tac3D2Adapter(SensorAdapter):
    """Mirror-stereo (e.g. Tac3D)
    - Uses config.sensor.params for geometry (camera, mirrors, panel).
    - During prepare(first_frame):
        * builds two virtual cameras from mirror planes
        * predicts pixel locations of every (r,c) in both halves
        * detects dots and matches to predictions (greedy gated NN)
        * triangulates rest 3D positions (panel/world frame)
    - During process(frame):
        * detects dots in each half
        * tracks each (r,c) by proximity to last observed pixels
        * triangulates current 3D, computes Δ vs rest, and returns Deformation(kind='3d')
    Output shape: (L=1, H=rows, W=cols, C=3) in millimeters (ux, uy, uz) with valid_mask.
    """

    def __init__(self, cfg, dbg_disp=None, dbg_writers=None):
        super().__init__(cfg, dbg_disp, dbg_writers)
        self.params: Dict[str, Any] = dict(getattr(cfg.sensor, 'params', {}) or {})

        # sensor params
        self.params: Dict[str, Any] = dict(getattr(cfg.sensor, 'params', {}) or {})
        P = self.params

        # Camera intrinsics
        self.fx = float(P.get('fx_px', 180.0))
        self.fy = float(P.get('fy_px', self.fx))
        self.cx = float(P.get('cx_px', 320.0))
        self.cy = float(P.get('cy_px', 240.0))
        self.K = build_K(self.fx, self.fy, self.cx, self.cy)

        # Camera pose (relative to world/panel)
        Cx, Cy, Cz = (P.get('cam_center_mm') or [0.0, 0.0, -70.0])
        yaw, pitch, roll = (P.get('cam_yaw_pitch_roll_deg') or [0.0, 0.0, 0.0])
        self.C_w = np.array([Cx, Cy, Cz], dtype=float)
        self.R_wc = rot_yaw_pitch_roll_deg(yaw, pitch, roll)



        # Mirrors (support 2)
        self.mirrors = []
        m1c = np.array(P.get('mirror1_center_mm', [0.0, 0.0, 30.0]), dtype=float)
        m1ypr = P.get('mirror1_yaw_pitch_roll_deg', [0.0, 0.0, 0.0])
        Rm1 = rot_yaw_pitch_roll_deg(*m1ypr)
        n1 = (Rm1 @ np.array([0.0, 0.0, 1.0], dtype=float))
        self.mirrors.append({'name': 'm1','center': m1c, 'n': n1, 'R': Rm1})

        m2c = np.array(P.get('mirror2_center_mm', [0.0, 0.0, 30.0]), dtype=float)
        m2ypr = P.get('mirror2_yaw_pitch_roll_deg', [0.0, 0.0, 0.0])
        Rm2 = rot_yaw_pitch_roll_deg(*m2ypr)
        n2 = (Rm2 @ np.array([0.0, 0.0, 1.0], dtype=float))
        self.mirrors.append({'name': 'm2','center': m2c, 'n': n2, 'R': Rm2})
        if len(self.mirrors) != 2:
            print(f"[Tac3DAdapter] Warning: expected 2 mirrors; got {len(self.mirrors)}.")

        # Panel grid
        self.rows = int(P.get('panel_rows', 20))
        self.cols = int(P.get('panel_cols', 20))
        self.spacing_mm = float(P.get('panel_spacing_mm', 1.5))

        # Split
        self.split_mode = str(P.get('split_mode', 'half'))
        self.seam_px = int(P.get('split_seam_px', 0))

        # Detection params
        self.det_min_area = float(P.get('detect_min_area_px', 6.0))
        self.det_max_area = float(P.get('detect_max_area_px', 1e6))
        self.det_min_circ = float(P.get('detect_min_circularity', 0.3))

        # Matching gates
        self.match_gate_px = float(P.get('match_gating_px', 14.0))
        self.track_radius_px = float(P.get('search_radius_px', 16.0))

        # Triangulation thresholds
        self.max_reproj_err = float(P.get('triang_max_reproj_err_px', 2.0))
        self.min_angle_deg = float(P.get('triang_min_angle_deg', 0.5))

        # Indexing mode: 'track' (default) or 'robust'
        self.index_mode = str(P.get('index_mode', 'track')).lower()
        self.robust_anchor_bins = int(P.get('robust_anchor_bins', self.rows))
        self.robust_gate_px = float(P.get('robust_gate_px', 12.0))
        self.robust_max_step_mult = float(P.get('robust_max_step_mult', 1.8))

        # State
        self.vcams = []                      # list of dict per mirror: {P, R_cw, t_c, name}
        self.pred_uv = {}                    # mirror name -> (R*C,2) predicted pixels at rest
        self.rest_X = None                   # (R,C,3) rest 3D in panel/world frame
        self.prev_uv = {}                    # per mirror, per (r,c) last pixel locations (R,C,2)
        self.valid_mask = None               # (R,C) bool

    # ---------- helper ----------
    def _panel_grid_uv(self) -> np.ndarray:
        # (R*C,3) points on z=0, centered grid in panel/world
        u_vals = (np.arange(self.cols) - (self.cols - 1)/2.0) * self.spacing_mm
        v_vals = (np.arange(self.rows) - (self.rows - 1)/2.0) * self.spacing_mm
        uu, vv = np.meshgrid(u_vals, v_vals)   # rows as v (down), cols as u (right)
        pts = np.stack([uu, vv, np.zeros_like(uu)], axis=-1).reshape(-1, 3)
        return pts

    def _build_virtual_cameras(self):
        self.vcams.clear()
        for idx, m in enumerate(self.mirrors):
            Cw_p, Rwc_p = virtual_camera_from_mirror(self.C_w, self.R_wc, m['n'], m['center'])
            R_cw, t_c = extrinsics_world_to_cam(Cw_p, Rwc_p)    # camera coords
            P = self.K @ np.hstack([R_cw, t_c.reshape(3,1)])    # camera matrix
            #name = f"m{idx+1}"
            name = m.get('name', f"m{idx+1}")
            self.vcams.append({'name': name, 'P': P, 'R_cw': R_cw, 't_c': t_c, 'mirror': m})

    def _predict_pixels_for_grid(self) -> Dict[str, np.ndarray]:
        """
        Project ideal panel grid (z=0) to each virtual camera (no deformation).
        Returns dict: mirror name -> (N,2) predicted pixels.
        """
        pts = self._panel_grid_uv()  # (N,3)
        pred = {}

        # For each virtual camera (real camera reflected by two mirrors)
        for vc in self.vcams:
            P = vc['P']     # camera matrix
            Xh = np.hstack([pts, np.ones((pts.shape[0],1))])    # (N,4)
            xh = (P @ Xh.T).T   # proj homog (N,3)
            uv = xh[:, :2] / np.maximum(1e-9, xh[:, 2:3])   # image pixels (N,2)
            pred[vc['name']] = uv  # (N,2)
        return pred

    def _split_halves(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Simply split image from center into left/right halves.
        """
        H, W = gray.shape[:2]
        mid = W // 2
        pad = max(0, int(self.seam_px))
        left = gray[:, :mid-pad]
        right = gray[:, mid+pad:]
        return left, right, mid

    def _detect_halves(self, gray: np.ndarray):
        """
        Detect dots in left/right halves.
        """
        left, right, mid = self._split_halves(gray)
        ptsL = detect_bright_blobs(left, self.det_min_area, self.det_max_area, self.det_min_circ)
        ptsR = detect_bright_blobs(right, self.det_min_area, self.det_max_area, self.det_min_circ)
        # convert to full-image coordinates
        ptsL[:,0] += 0.0
        ptsR[:,0] += (mid + max(0, int(self.seam_px)))
        return ptsL, ptsR

    def prepare(self, first_bgr: np.ndarray) -> None:
        self.frame_id = 0
        
        #--- Build virtual cameras ---
        self._build_virtual_cameras()
        if len(self.vcams) < 2:
            print("[Tac3DAdapter] ERROR: need 2 mirrors/virtual cameras.")
            return

        #--- Predict locations for each (r,c) ---
        # predicted pixel locations at rest
        self.pred_uv = self._predict_pixels_for_grid()  # dict of two mirrors name -> (N,2)
        uvL_pred = self.pred_uv[self.vcams[0]['name']]
        uvR_pred = self.pred_uv[self.vcams[1]['name']]

        #--- Detect blobs in the first frame ---
        gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
        ptsL, ptsR = self._detect_halves(gray)

        #--- Indexing dots ---
        N = self.rows * self.cols
        uvL = np.full((N, 2), np.nan, dtype=float)
        uvR = np.full((N, 2), np.nan, dtype=float)

        if self.index_mode == "robust":
            # Robust, tracking-free indexing on each half
            gridL, maskL = robust_index_half(
                ptsL, self.rows, self.cols, side="left",
                anchor_bins=self.robust_anchor_bins,
                gate_px=self.robust_gate_px,
                max_step_mult=self.robust_max_step_mult,
            )
            gridR, maskR = robust_index_half(
                ptsR, self.rows, self.cols, side="right",
                anchor_bins=self.robust_anchor_bins,
                gate_px=self.robust_gate_px,
                max_step_mult=self.robust_max_step_mult,
            )
            uvL = assign_grid_to_predictions(gridL, maskL, uvL_pred, self.match_gate_px)
            uvR = assign_grid_to_predictions(gridR, maskR, uvR_pred, self.match_gate_px)
        else:
            # prediction-to-detection matching
            _, idx_pt_for_pred_L = match_to_predictions(ptsL, uvL_pred, self.match_gate_px)
            _, idx_pt_for_pred_R = match_to_predictions(ptsR, uvR_pred, self.match_gate_px)
            okL = (idx_pt_for_pred_L >= 0)
            okR = (idx_pt_for_pred_R >= 0)
            for j in np.where(okL & okR)[0]:
                uvL[j] = ptsL[idx_pt_for_pred_L[j]]
                uvR[j] = ptsR[idx_pt_for_pred_R[j]]

        #--- Triangulate rest 3D ---
        P1 = self.vcams[0]['P']; P2 = self.vcams[1]['P']
        R1 = self.vcams[0]['R_cw']; t1 = self.vcams[0]['t_c']
        R2 = self.vcams[1]['R_cw']; t2 = self.vcams[1]['t_c']
        Xw = np.full((N,3), np.nan, dtype=float)
        mask = np.zeros((N,), dtype=bool)
        for j in range(N):
            if not np.isfinite(uvL[j,0]) or not np.isfinite(uvR[j,0]):
                continue
            X = triangulate_linear(P1, P2, uvL[j], uvR[j])
            # reproj & depth checks
            e1 = reproj_error(P1, X, uvL[j]); e2 = reproj_error(P2, X, uvR[j])
            if e1 > self.max_reproj_err or e2 > self.max_reproj_err:
                continue
            if z_cam_sign(R1, t1, X) <= 0 or z_cam_sign(R2, t2, X) <= 0:
                continue
            Xw[j] = X; mask[j] = True

        # Save rest state (panel/world coords)
        self.rest_X = Xw.reshape(self.rows, self.cols, 3)
        self.valid_mask = mask.reshape(self.rows, self.cols)

        # Initialize per-mirror last seen pixel positions from matches; will be refined next frames
        self.prev_uv = {
            self.vcams[0]['name']: uvL.reshape(self.rows, self.cols, 2),
            self.vcams[1]['name']: uvR.reshape(self.rows, self.cols, 2),
        }
        print(f"[Tac3DAdapter] prepare(): rest valid dots = {int(mask.sum())}/{N}")


    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        """
        Return TIR Deformation
        Includes both 3D and 2D deformation fields.
        """

        self.frame_id += 1
        if self.rest_X is None:
            print("[Tac3DAdapter] ERROR: call prepare() first.")
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect
        ptsL, ptsR = self._detect_halves(gray)

        # Indexing
        name1, name2 = self.vcams[0]['name'], self.vcams[1]['name']
        if self.index_mode == "robust":
            # Per-frame robust indexing (no tracking)
            uvL_pred = self.pred_uv[name1]
            uvR_pred = self.pred_uv[name2]
            gridL, maskL = robust_index_half(
                ptsL, self.rows, self.cols, side="left",
                anchor_bins=self.robust_anchor_bins,
                gate_px=self.robust_gate_px,
                max_step_mult=self.robust_max_step_mult,
            )
            gridR, maskR = robust_index_half(
                ptsR, self.rows, self.cols, side="right",
                anchor_bins=self.robust_anchor_bins,
                gate_px=self.robust_gate_px,
                max_step_mult=self.robust_max_step_mult,
            )
            uvL = assign_grid_to_predictions(gridL, maskL, uvL_pred, self.match_gate_px)
            uvR = assign_grid_to_predictions(gridR, maskR, uvR_pred, self.match_gate_px)
            # For continuity of debug/state, update prev_uv with what we measured
            self.prev_uv[name1] = np.where(
                np.isfinite(uvL.reshape(self.rows, self.cols, 2)[..., 0])[..., None],
                uvL.reshape(self.rows, self.cols, 2),
                self.prev_uv[name1]
            )
            self.prev_uv[name2] = np.where(
                np.isfinite(uvR.reshape(self.rows, self.cols, 2)[..., 0])[..., None],
                uvR.reshape(self.rows, self.cols, 2),
                self.prev_uv[name2]
            )
        else:
            # Tracking from last frame
            uvL = self._track_by_proximity(ptsL, self.prev_uv[name1], self.track_radius_px)
            uvR = self._track_by_proximity(ptsR, self.prev_uv[name2], self.track_radius_px)


        # Triangulate current frame
        P1 = self.vcams[0]['P']; P2 = self.vcams[1]['P']
        R1 = self.vcams[0]['R_cw']; t1 = self.vcams[0]['t_c']
        R2 = self.vcams[1]['R_cw']; t2 = self.vcams[1]['t_c']

        N = self.rows*self.cols
        Xw = np.full((N,3), np.nan, dtype=float)
        mask = np.zeros((N,), dtype=bool)
        for j in range(N):
            if not np.isfinite(uvL[j,0]) or not np.isfinite(uvR[j,0]):
                continue
            X = triangulate_linear(P1, P2, uvL[j], uvR[j])
            e1 = reproj_error(P1, X, uvL[j]); e2 = reproj_error(P2, X, uvR[j])
            if e1 > self.max_reproj_err or e2 > self.max_reproj_err:
                continue
            if z_cam_sign(R1, t1, X) <= 0 or z_cam_sign(R2, t2, X) <= 0:
                continue
            Xw[j] = X; mask[j] = True

        Xw_grid = Xw.reshape(self.rows, self.cols, 3)
        mask_grid = mask.reshape(self.rows, self.cols)

        # Update trackers: where we had valid measurements, update prev_uv
        uvL_grid = uvL.reshape(self.rows, self.cols, 2)
        uvR_grid = uvR.reshape(self.rows, self.cols, 2)
        for nm, grid in ((name1, uvL_grid), (name2, uvR_grid)):
            prev = self.prev_uv[nm]
            upd = np.where(np.isfinite(grid[...,0])[...,None], grid, prev)
            self.prev_uv[nm] = upd

        # Compute displacement in panel/world coords against rest
        disp_3d = np.full_like(Xw_grid, np.nan, dtype=float)    # (R,C,3)
        valid_3d = mask_grid & self.valid_mask                  # (R,C) bool
        disp_3d[valid_3d] = Xw_grid[valid_3d] - self.rest_X[valid_3d]   # only write valid dots, invalid stay NaN

        # Pack Deformation (mm)
        H, W = self.rows, self.cols
        data = np.zeros((1, H, W, 3), dtype=np.float32)
        # Fill NaNs with zeros, but mark mask in meta
        dfill = np.where(np.isfinite(disp_3d), disp_3d, 0.0).astype(np.float32)
        data[0] = dfill

        meta3d = TirMeta(
            H=H, W=W,
            mm_per_px=float(self.cfg.physics.mm_per_px),
            downscale=float(self.cfg.downscale),
            frame_id=self.frame_id,
            timestamp_usec=0,
            valid_mask=valid_3d.astype(np.uint8),
            sensor_frame="S"
        )

        deform_3d = Deformation(
            data=data,
            kind='3d',
            meta=meta3d,
            z_of_layer=np.array([0.0], dtype=np.float32),
            debug={
                'uv_left': uvL_grid, 'uv_right': uvR_grid,
                'mask_meas': mask_grid.astype(np.uint8),
            }
        )

        # build 2D 
        H_img, W_img = gray.shape[:2]
        vx_dense, vy_dense, valid2d = self._dot_dis_to_pixel_disp(H_img, W_img, disp_3d, valid_3d)

        # Pack 2D Deformation (vy, vx) in pixels
        data2d = np.zeros((1, H_img, W_img, 2), dtype=np.float32)
        data2d[0, ..., 0] = vy_dense   # channel 0 = vy (pixels, y-down)
        data2d[0, ..., 1] = vx_dense   # channel 1 = vx (pixels, x-right)

        meta2d = TirMeta(
            H=H_img, W=W_img,
            mm_per_px=float(self.cfg.physics.mm_per_px),  # not used for '2d' flows, kept for consistency
            downscale=float(self.cfg.downscale),
            frame_id=self.frame_id,
            timestamp_usec=0,
            valid_mask=valid2d,
            sensor_frame="S"
        )

        deform_2d = Deformation(
            data=data2d,
            kind='2d',
            meta=meta2d,
            z_of_layer=None,
        )

        return deform_2d

    # ---------- internal methods ----------

    def _track_by_proximity(self, pts: np.ndarray, prev_uv_grid: np.ndarray, radius: float) -> np.ndarray:
        """
        Track each (r,c) dot by nearest neighbor search from last known location.
        Return (R*C,2) measured pixels by nearest neighbor to last location (within radius), else NaN.
        """
        N = prev_uv_grid.shape[0]*prev_uv_grid.shape[1]
        out = np.full((N,2), np.nan, dtype=float)
        if pts.shape[0]==0:
            return out
        prev = prev_uv_grid.reshape(N,2)
        # For each prev location, find nearest current point within radius
        for j in range(N):
            p = prev[j]
            if not np.isfinite(p[0]): 
                continue
            d = np.linalg.norm(pts - p[None,:], axis=1)
            k = np.argmin(d)
            if d[k] <= radius:
                out[j] = pts[k]
        return out

    def _virtual_scales_fit_image(self, W_img: int, H_img: int) -> tuple[float, float]:
        """
        Compute virtual camera's mm->px scales so that the panel grid tightly fits the image.
        sx = (W_img-1) / panel_width_mm,  sy = (H_img-1) / panel_height_mm
        and find max to keep full panel in view.
        """
        width_mm  = max(1e-9, (self.cols - 1) * self.spacing_mm)
        height_mm = max(1e-9, (self.rows - 1) * self.spacing_mm)
        sx = (W_img - 1) / width_mm
        sy = (H_img - 1) / height_mm
        scale = max(sx, sy)
        return float(scale), float(scale)

    def _dot_dis_to_pixel_disp(self, H_img: int, W_img: int, disp_3d: np.ndarray, valid_3d):
        """
        Build virtual fronto-parallel 2D flow (pixels) from sparse 20x20 dots
        
        Treat the panel as a fronto-parallel "virtual camera" view and map
        panel (u,v) in mm -> image pixels with auto-inferred scales so the panel fits the image.
        """

        sx, sy = self._virtual_scales_fit_image(W_img, H_img)   # mm -> px

        # Coarse (rows x cols) flow at dot lattice positions (defined at REST indices)
        # Note: image coords are x-right, y-down. We choose vy = +dv*sy, vx = +du*sx.
        vx_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
        vy_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
        w_coarse  = valid_3d.astype(np.float32)   # weights (1 for valid dots, 0 for missing)

        # disp[...,0]=Δu_mm, disp[...,1]=Δv_mm (panel/world mm)
        vx_coarse[valid_3d] = (disp_3d[..., 0][valid_3d] * sx).astype(np.float32)
        vy_coarse[valid_3d] = (disp_3d[..., 1][valid_3d] * sy).astype(np.float32)

        # Upscale the sparse lattice to full resolution using mask-weighted resize.
        # We upsample both the flow and the weight, then normalize: F_up / (W_up + eps).
        vx_up = cv2.resize(vx_coarse, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
        vy_up = cv2.resize(vy_coarse, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
        w_up  = cv2.resize(w_coarse,  (W_img, H_img), interpolation=cv2.INTER_LINEAR)

        eps = 1e-6
        vx_dense = vx_up / (w_up + eps)
        vy_dense = vy_up / (w_up + eps)

        # Where the upsampled weight is near zero (no support), set zeros and mark invalid.
        valid2d = (w_up > 0.01).astype(np.uint8)
        vx_dense = np.where(valid2d.astype(bool), vx_dense, 0.0).astype(np.float32)
        vy_dense = np.where(valid2d.astype(bool), vy_dense, 0.0).astype(np.float32)

        return vx_dense, vy_dense, valid2d
