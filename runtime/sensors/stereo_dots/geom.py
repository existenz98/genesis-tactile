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
 Geometry utilities
 World (= panel) frame, gel plane z=0, normal +Z.
 Camera frame (OpenCV): X right, Y down, Z forward.
"""


from __future__ import annotations
import numpy as np


def rot_yaw_pitch_roll_deg(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Return rotation matrix R (3x3) from yaw, pitch, roll
    """
    y = np.deg2rad(yaw or 0.0)
    p = np.deg2rad(pitch or 0.0)
    r = np.deg2rad(roll or 0.0)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    R_y = np.array([[ cy, 0.0,  sy],[0.0,1.0,0.0],[-sy,0.0, cy]], dtype=float)
    R_x = np.array([[1.0, 0.0, 0.0],[0.0, cp,-sp],[0.0, sp, cp]], dtype=float)
    R_z = np.array([[ cr,-sr, 0.0],[ sr, cr, 0.0],[0.0,0.0,1.0]], dtype=float)
    return R_y @ R_x @ R_z

def reflect_point_plane(P: np.ndarray, n: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """
    Reflect points P across plane defined by normal n and point p0.
    """
    P = np.asarray(P, dtype=float).reshape(-1,3)
    n = np.asarray(n, dtype=float).reshape(1,3)
    p0 = np.asarray(p0, dtype=float).reshape(1,3)
    n = n / max(1e-12, np.linalg.norm(n))
    v = P - p0
    d = (v * n).sum(axis=1, keepdims=True)
    return P - 2.0 * d * n

def reflect_dirs_plane(V: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Reflect direction vectors V across plane with normal n.
    """
    V = np.asarray(V, dtype=float).reshape(-1,3)
    n = np.asarray(n, dtype=float).reshape(1,3)
    n = n / max(1e-12, np.linalg.norm(n))
    d = (V * n).sum(axis=1, keepdims=True)
    return V - 2.0 * d * n

def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Build camera intrinsic matrix K.
    """
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

def extrinsics_world_to_cam(C_w: np.ndarray, R_wc: np.ndarray):
    """
    Convert extrinsics from world to camera coordinates.
    Return (R_cw, t_c) such that x_c = R_cw (x_w - C_w).
    """
    R_cw = R_wc.T
    t_c = - R_cw @ C_w.reshape(3,)
    return R_cw, t_c

def virtual_camera_from_mirror(C_w: np.ndarray, R_wc: np.ndarray, n: np.ndarray, p0: np.ndarray):
    """
    Reflect real camera across plane (n,p0) to get virtual camera in world coords.
    Returns (Cw', R_wc').
    """
    H = np.eye(3) - 2.0 * np.outer(n, n) / max(1e-12, np.dot(n,n))
    Cw_p = reflect_point_plane(C_w.reshape(1,3), n, p0).reshape(3,)
    R_wc_p = H @ R_wc
    return Cw_p, R_wc_p

def triangulate_linear(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Linear triangulation for a single correspondence.
    x1,x2 are pixel coords (u,v). Returns X in world (non-homog).
    """
    u1,v1 = float(x1[0]), float(x1[1])
    u2,v2 = float(x2[0]), float(x2[1])
    A = np.zeros((4,4), dtype=float)
    A[0] = u1*P1[2] - P1[0]
    A[1] = v1*P1[2] - P1[1]
    A[2] = u2*P2[2] - P2[0]
    A[3] = v2*P2[2] - P2[1]
    _,_,Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    Xh /= (Xh[3] if abs(Xh[3])>1e-12 else 1.0)
    return Xh[:3]

def reproj_error(P: np.ndarray, Xw: np.ndarray, x: np.ndarray) -> float:
    """
    Compute reprojection error (in pixels) of 3D point Xw to image point x via P.
    """
    Xh = np.append(Xw, 1.0)
    xh = P @ Xh
    if abs(xh[2]) < 1e-12: return 1e9
    u = xh[0]/xh[2]; v = xh[1]/xh[2]
    return float(np.hypot(u - x[0], v - x[1]))

def z_cam_sign(R_cw: np.ndarray, t_c: np.ndarray, Xw: np.ndarray) -> float:
    """
    Compute Z coordinate in camera frame of 3D point Xw.
    Positive means in front of camera.
    """
    Xc = R_cw @ Xw.reshape(3,) + t_c.reshape(3,)
    return float(Xc[2])
