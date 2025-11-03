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
Mirror-Stereo Dots grid Vision Tactile Sensor
E.g. Tac3D from Acorn.

Outputs per-dot 3D and 2D deformation

Usage:
python -m runtime.run_demo   \
    --source video  --input dataset/sequences/tac3d_zigzag/video/sequence.mkv  --loop \
    --config runtime/config/runtime_config_stereodots.yaml \
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
from .index import match_to_predictions, index_grid_from_points

from ...output.visualizer import flow_to_color_bgr, draw_quiver_bgr

from ...utils.prof import Prof


class StereoDotsAdapter(SensorAdapter):
    """
    Mirror-stereo dots grid vision tactile sensor
    (e.g. Tac3D)
    - Uses config.sensor.params for geometry (camera, mirrors, panel).
    - During prepare(first_frame):
        * builds two virtual cameras from mirror planes
        * predicts pixel locations of every (r,c) in both halves
        * detects dots and matches to predictions (greedy gated NN)
        * triangulates rest 3D positions (panel/world frame)
    - During process(frame):
        * detects dots in each half
        * tracks each (r,c) by proximity to last observed pixels
        * triangulates current 3D, computes dots' 3D displacement vs rest, and returns Deformation(kind='3d' and '2d')
    """

    def __init__(self, cfg, dbg_disp=None, dbg_writers=None):
        super().__init__(cfg, dbg_disp, dbg_writers)


        # profiler
        self.prof = Prof(enable=True, report_every=5)

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
            print(f"[StereoDotsAdapter] Warning: expected 2 mirrors; got {len(self.mirrors)}.")

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

        # robust hyperparams
        self.robust_k_nn = int(P.get('robust_k_nn', 4))
        self.robust_angle_bins = int(P.get('robust_angle_bins', 90))
        self.robust_angle_tol_deg = float(P.get('robust_angle_tol_deg', 35.0))
        self.robust_max_len_mult = float(P.get('robust_max_len_mult', 3.0))

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
    
    def _uv_from_pt_of_rc(self, pts: np.ndarray, pt_of_rc: np.ndarray) -> np.ndarray:
        """
        Convert pt_of_rc to per-cell UV grid.
        pts: (N,2) detections (full-image coords). pt_of_rc: (R,C) int indices (or -1).
        Returns uv_grid: (R,C,2) with NaNs where missing.
        """
        uv_grid = np.full((self.rows, self.cols, 2), np.nan, dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                k = int(pt_of_rc[r, c])
                if k >= 0:
                    uv_grid[r, c] = pts[k]
        return uv_grid


    # ---------- main methods ----------

    def prepare(self, first_bgr: np.ndarray) -> None:
        self.frame_id = 0
        
        #--- Build virtual cameras ---
        self._build_virtual_cameras()
        if len(self.vcams) < 2:
            print("[StereoDotsAdapter] ERROR: need 2 mirrors/virtual cameras.")
            return

        #--- Predict locations for each (r,c) ---
        # predicted pixel locations at rest
        self.pred_uv = self._predict_pixels_for_grid()  # dict of two mirrors name -> (N,2)
        uvL_pred = self.pred_uv[self.vcams[0]['name']]
        uvR_pred = self.pred_uv[self.vcams[1]['name']]

        #--- Detect blobs in the first frame ---
        gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
        ptsL, ptsR = self._detect_halves(gray)      # shape of (n,2), u,v pixel coords
        num_dots_L = ptsL.shape[0]
        num_dots_R = ptsR.shape[0]
        print(f"[StereoDotsAdapter] prepare(): detected {num_dots_L} dots in left, {num_dots_R} in right.")

        #--- Indexing dots ---
        N = self.rows * self.cols
        uvL = np.full((N, 2), np.nan, dtype=float)
        uvR = np.full((N, 2), np.nan, dtype=float)

        if self.index_mode == "robust":
            uvL_grid, maskL = index_grid_from_points(ptsL, self.rows, self.cols, anchor="top-right")    # grid is (rows,cols,2)
            uvR_grid, maskR = index_grid_from_points(ptsR, self.rows, self.cols, anchor="top-left")
            uvL = uvL_grid.reshape(N, 2)    # convert from (rows,cols,2) to (N,2)
            uvR = uvR_grid.reshape(N, 2)
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
        print(f"[StereoDotsAdapter] prepare(): rest valid dots = {int(mask.sum())}/{N}")

        #--- Debug display ---
        if self.cfg.display.enable and self.cfg.display.show_seg_color:
            uvL_grid = self.prev_uv[self.vcams[0]['name']]
            uvR_grid = self.prev_uv[self.vcams[1]['name']]
            mask_grid = self.valid_mask
            # draw over the original first frame for context
            self._draw_index_overlay(first_bgr, uvL_grid, mask_grid, tag="index left rest")
            self._draw_index_overlay(first_bgr, uvR_grid, mask_grid, tag="index right rest")

    def process(self, bgr: np.ndarray) -> Optional[Deformation]:
        """
        Return TIR Deformation
        Includes both 3D and 2D deformation fields.
        """

        self.frame_id += 1
        if self.rest_X is None:
            print("[StereoDotsAdapter] ERROR: call prepare() first.")
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect
        with self.prof("1) detect halves"):
            ptsL, ptsR = self._detect_halves(gray)

        # 2. Indexing
        with self.prof("2) index dots"):
            uvL, uvR = self._build_index(ptsL, ptsR)

        # 3. Triangulate (dots from left panel vs right panel)
        with self.prof("3) triangulate 3D"):
            #Xw_grid, mask_grid = self._triangulate(uvL, uvR)
            Xw_grid, mask_grid = self._triangulate_fast_numpy(uvL, uvR)

        # Update trackers: where we had valid measurements, update prev_uv
        uvL_grid = uvL.reshape(self.rows, self.cols, 2)
        uvR_grid = uvR.reshape(self.rows, self.cols, 2)
        name1, name2 = self.vcams[0]['name'], self.vcams[1]['name']
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

        # --- Debug display ---
        if self.cfg.display.enable:
            H_img, W_img = gray.shape[:2]
            sx, sy = self._virtual_scales_fit_image(W_img, H_img)   # isotropic; sx==sy
            s = sx
            # build coarse (rows x cols) pixel flows (vx,vy) in the panel view
            vx_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
            vy_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
            vx_coarse[valid_3d] = (disp_3d[..., 0][valid_3d] * s).astype(np.float32)  # du_mm -> px
            vy_coarse[valid_3d] = (disp_3d[..., 1][valid_3d] * s).astype(np.float32)  # dv_mm -> px

        # Draw indexing overlays
        if self.cfg.display.enable and self.cfg.display.show_seg_color:
            self._draw_index_overlay(bgr, uvL_grid, mask_grid, tag="index left")
            self._draw_index_overlay(bgr, uvR_grid, mask_grid, tag="index right")
        # Draw sparse arrows on dot grid
        if self.cfg.display.enable and self.cfg.display.show_flow_quiver_raw:
            self._draw_sparse_panel_arrows(H_img, W_img, vx_coarse, vy_coarse, valid_3d, tag="dot displacement")

        self.prof.tick()

        return deform_2d

    # ---------- internal methods ----------

    def _build_index(self, ptsL: np.ndarray, ptsR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build per-half (N,2) uv arrays by indexing/tracking detected pts.
        Uses either robust indexing or tracking from last frame.
        """
        name1, name2 = self.vcams[0]['name'], self.vcams[1]['name']
        if self.index_mode == "robust":
            N = self.rows * self.cols
            # Stateless single-frame indexing on each half
            uvL_grid, maskL = index_grid_from_points(ptsL, self.rows, self.cols, anchor="top-right")    # grid is (rows,cols,2)
            uvR_grid, maskR = index_grid_from_points(ptsR, self.rows, self.cols, anchor="top-left")

            uvL = uvL_grid.reshape(N, 2)    # convert from (rows,cols,2) to (N,2)
            uvR = uvR_grid.reshape(N, 2)
            # Update prev_uv for visualization continuity
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

        return uvL, uvR

    def _triangulate(self, uvL: np.ndarray, uvR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate 3D points from left/right pixel locations.
        uvL, uvR: (N,2) pixel locations for each dot (NaN if missing)
        Returns:
            Xw_grid: (rows,cols,3) 3D points in panel/world frame (NaN if missing)
            mask_grid: (rows,cols) bool valid mask
        """
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

        return Xw_grid, mask_grid

    def _triangulate_fast_numpy(self, uvL: np.ndarray, uvR: np.ndarray):
        """
        Vectorized DLT triangulation for speed.
        Assumes:
        - P1, P2: 3x4 projection matrices (float64 or float32)
        - R_cw: 3x3 (camera->world), t_c: (3,) camera center in world
        """
        P1 = np.ascontiguousarray(self.vcams[0]['P'])
        P2 = np.ascontiguousarray(self.vcams[1]['P'])
        R1 = self.vcams[0]['R_cw']; t1 = self.vcams[0]['t_c']
        R2 = self.vcams[1]['R_cw']; t2 = self.vcams[1]['t_c']

        N = self.rows * self.cols
        valid = np.isfinite(uvL).all(axis=1) & np.isfinite(uvR).all(axis=1)
        idx = np.flatnonzero(valid)
        M = idx.size

        Xw = np.full((N, 3), np.nan, dtype=P1.dtype)
        mask = np.zeros(N, dtype=bool)
        if M == 0:
            return Xw.reshape(self.rows, self.cols, 3), mask.reshape(self.rows, self.cols)

        u1, v1 = uvL[idx, 0], uvL[idx, 1]
        u2, v2 = uvR[idx, 0], uvR[idx, 1]
        p10, p11, p12 = P1[0], P1[1], P1[2]
        p20, p21, p22 = P2[0], P2[1], P2[2]

        # Build A for all points: shape (M, 4, 4)
        A = np.empty((M, 4, 4), dtype=P1.dtype)
        A[:, 0, :] = u1[:, None] * p12 - p10
        A[:, 1, :] = v1[:, None] * p12 - p11
        A[:, 2, :] = u2[:, None] * p22 - p20
        A[:, 3, :] = v2[:, None] * p22 - p21

        # Batched SVD; take last row of Vh as homogeneous X
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        Xh = Vh[:, -1, :]                         # (M, 4)
        X = Xh[:, :3] / Xh[:, 3:4]                # (M, 3)

        # Reprojection error, vectorized
        Xh1 = np.concatenate([X, np.ones((M, 1), dtype=P1.dtype)], axis=1)  # (M, 4)
        x1 = Xh1 @ P1.T                               # (M, 3)
        x2 = Xh1 @ P2.T
        uv1_hat = x1[:, :2] / x1[:, 2:3]
        uv2_hat = x2[:, :2] / x2[:, 2:3]
        e1 = np.linalg.norm(uv1_hat - uvL[idx], axis=1)
        e2 = np.linalg.norm(uv2_hat - uvR[idx], axis=1)
        ok_err = (e1 <= self.max_reproj_err) & (e2 <= self.max_reproj_err)

        # Positive depth in both cameras
        Rwc1 = R1.T; Rwc2 = R2.T
        z1 = (X - t1) @ Rwc1[2]                     # row 2 is the camera z-axis in world
        z2 = (X - t2) @ Rwc2[2]
        ok_z = (z1 > 0) & (z2 > 0)

        ok = ok_err & ok_z
        if np.any(ok):
            Xw[idx[ok]] = X[ok]
            mask[idx[ok]] = True

        return Xw.reshape(self.rows, self.cols, 3), mask.reshape(self.rows, self.cols)

    

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
        keep full panel in view and keep aspect ratio by choosing min scale.
        """
        width_mm  = max(1e-9, (self.cols - 1) * self.spacing_mm)
        height_mm = max(1e-9, (self.rows - 1) * self.spacing_mm)
        sx = (W_img - 1) / width_mm
        sy = (H_img - 1) / height_mm
        # keep aspect ratio
        scale = min(sx, sy)
        return float(scale), float(scale)

    def _dot_dis_to_pixel_disp(self, H_img: int, W_img: int, disp_3d: np.ndarray, valid_3d):
        """
        Build virtual fronto-parallel 2D flow (pixels) from sparse 20x20 dots
        
        Treat the panel as a fronto-parallel "virtual camera" view and map
        panel (u,v) in mm -> image pixels with auto-inferred scales so the panel fits the image.
        """

        # 1. Compute virtual camera's mm->px scales
        sx, sy = self._virtual_scales_fit_image(W_img, H_img)   # mm -> px

        # 2.Coarse (rows x cols) 2D displacement (at dot lattice positions defined at REST indices)
        vx_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
        vy_coarse = np.zeros((self.rows, self.cols), dtype=np.float32)
        w_coarse  = valid_3d.astype(np.float32)   # weights (1 for valid dots, 0 for missing)

        # scale 3D displacement, disp[...,0]=du_mm, disp[...,1]=dv_mm (panel/world mm) to 2D pixel displacement
        vx_coarse[valid_3d] = (disp_3d[..., 0][valid_3d] * sx).astype(np.float32)
        vy_coarse[valid_3d] = (disp_3d[..., 1][valid_3d] * sy).astype(np.float32)

        # Note: image coords are x-right, y-down. We choose vy = +dv*sy, vx = +du*sx.
        vy_coarse = -vy_coarse   # because panel v is y-down but camera image y is down

        # 3. Upscale the sparse lattice to full resolution using mask-weighted resize.

        # 1) compute square-preserving panel rect in the image
        width_mm  = max(1e-9, (self.cols - 1) * self.spacing_mm)
        height_mm = max(1e-9, (self.rows - 1) * self.spacing_mm)
        # sx == sy == scale (isotropic) from _virtual_scales_fit_image
        s = sx  # since _virtual_scales_fit_image returns (scale, scale)

        W_panel = max(1, int(round(s * width_mm)))
        H_panel = max(1, int(round(s * height_mm)))
        # center the panel rectangle (letterbox/pillarbox as needed)
        u0 = (W_img - W_panel) // 2
        v0 = (H_img - H_panel) // 2

        # 2) upsample only to the panel rect, keep aspect ratio
        vx_panel = cv2.resize(vx_coarse, (W_panel, H_panel), interpolation=cv2.INTER_CUBIC)
        vy_panel = cv2.resize(vy_coarse, (W_panel, H_panel), interpolation=cv2.INTER_CUBIC)

        # validity: keep it binary (0/1). Use nearest so we don't create fractional weights.
        w_panel = cv2.resize(w_coarse, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
        mask_panel = (w_panel > 0.5).astype(np.uint8)

        # 3) paste into full-size images; zeros outside the panel rect
        vx_dense = np.zeros((H_img, W_img), dtype=np.float32)
        vy_dense = np.zeros((H_img, W_img), dtype=np.float32)
        valid2d  = np.zeros((H_img, W_img), dtype=np.uint8)

        vx_dense[v0:v0+H_panel, u0:u0+W_panel] = vx_panel
        vy_dense[v0:v0+H_panel, u0:u0+W_panel] = vy_panel
        valid2d [v0:v0+H_panel, u0:u0+W_panel] = mask_panel

        # 4) force zero displacement in panel pixels with bad validity
        vx_dense[v0:v0+H_panel, u0:u0+W_panel] *= mask_panel
        vy_dense[v0:v0+H_panel, u0:u0+W_panel] *= mask_panel

        return vx_dense, vy_dense, valid2d

    # ------ Debug display ------
    def _panel_rect_from_scale(self, W_img: int, H_img: int, scale: float) -> tuple[int, int, int, int]:
        """
        Compute letterboxed panel rectangle (u0, v0, W_panel, H_panel).
        """
        width_mm  = max(1e-9, (self.cols - 1) * self.spacing_mm)
        height_mm = max(1e-9, (self.rows - 1) * self.spacing_mm)
        W_panel = max(1, int(round(scale * width_mm)))
        H_panel = max(1, int(round(scale * height_mm)))
        u0 = (W_img - W_panel) // 2
        v0 = (H_img - H_panel) // 2
        return u0, v0, W_panel, H_panel

    def _draw_index_overlay(self, base_bgr: np.ndarray, uv_grid: np.ndarray, mask_grid: np.ndarray, tag: str, scale: float = 4.0) -> None:
        """
        Draw per-dot indices (r,c) over an image for quick visual validation.
        Colors encode row; text shows "r,c".
        """
        if not self.cfg.display.enable or not getattr(self.cfg.display, "show_seg_color", False):
            return

        H, W = base_bgr.shape[:2]
        vis = base_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # scale
        H, W = base_bgr.shape[:2]
        s = max(1.0, scale)  # keep >=1 for readability

        # scale the background FIRST
        Wd, Hd = int(round(W * s)), int(round(H * s))
        vis = cv2.resize(base_bgr, (Wd, Hd), interpolation=cv2.INTER_LINEAR)

        # scale-aware drawing params
        r_px = max(1, int(round(2 * s)))                 # dot radius
        thk  = max(1, int(round(1 * s)))                 # thickness
        fsc  = 0.5                                       # font scale
        txt_thk = 1
        txt_off = max(3, int(round(3 * s)))              # small offset for label

        # simple row color ramp
        for r in range(self.rows):
            color = (int(255 * r / max(1, self.rows - 1)), 50, 255 - int(255 * r / max(1, self.rows - 1)))
            for c in range(self.cols):
                if not mask_grid[r, c]:
                    continue
                u, v = uv_grid[r, c]
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                # scale coordinates after background was scaled
                p = (int(round(u * s)), int(round(v * s)))
                cv2.circle(vis, p, r_px, color, -1, lineType=cv2.LINE_AA)
                cv2.putText(vis, f"{r},{c}",
                            (p[0] + txt_off, p[1] - txt_off),
                            font, fsc, (255, 255, 255), txt_thk, cv2.LINE_AA)
        self.dbg_disp.show(tag, vis)

    def _draw_sparse_panel_arrows(self, H_img: int, W_img: int,
                                  vx_coarse: np.ndarray, vy_coarse: np.ndarray,
                                  valid: np.ndarray, tag: str, scale: float = 3.0) -> None:
        """
        Draw sparse arrows at panel lattice locations on a blank image of size (H_img, W_img).
        """
        if not self.cfg.display.enable or not getattr(self.cfg.display, "show_flow_quiver_raw", False):
            return

        s, _ = self._virtual_scales_fit_image(W_img, H_img)
        u0, v0, Wp, Hp = self._panel_rect_from_scale(W_img, H_img, s)
        vis = np.zeros((H_img, W_img, 3), dtype=np.uint8)

        # lattice pixel centers within panel rect
        if self.cols > 1:
            u_frac = np.linspace(0.0, 1.0, self.cols)
        else:
            u_frac = np.array([0.5], dtype=float)
        if self.rows > 1:
            v_frac = np.linspace(0.0, 1.0, self.rows)
        else:
            v_frac = np.array([0.5], dtype=float)
        for r in range(self.rows):
            v = int(round(v0 + v_frac[r] * Hp))
            for c in range(self.cols):
                if not valid[r, c]:
                    continue
                u = int(round(u0 + u_frac[c] * Wp))
                du = float(vx_coarse[r, c]) * scale
                dv = float(vy_coarse[r, c]) * scale
                p0 = (u, v)
                p1 = (int(round(u + du)), int(round(v + dv)))
                cv2.arrowedLine(vis, p0, p1, (0, 255, 255), 1, tipLength=0.25)
        self.dbg_disp.show(tag, vis)
