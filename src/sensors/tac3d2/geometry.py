
"""
Geometry and projection utilities for Tac3D synthetic rendering
"""


from __future__ import annotations
import numpy as np

# Camera frame: X right, Y down, Z forward (OpenCV convention).
# Image axes: u right, v down.

def euler_ypr_deg(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Return rotation matrix R (3x3) from yaw(Z), pitch(X), roll
    defined as:
      - yaw about +Y (camera down) axis
      - pitch about +X axis
      - roll about +Z axis
    This definition matches common Y-X-Z order in the OpenCV camera frame.
    """
    # Convert degrees to radians
    y = np.deg2rad(yaw_deg or 0.0)
    p = np.deg2rad(pitch_deg or 0.0)
    r = np.deg2rad(roll_deg or 0.0)

    # Rotation about Y (yaw)
    cy, sy = np.cos(y), np.sin(y)
    R_y = np.array([[ cy, 0.0,  sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0,  cy]], dtype=float)

    # Rotation about X (pitch)
    cp, sp = np.cos(p), np.sin(p)
    R_x = np.array([[1.0, 0.0, 0.0],
                    [0.0,  cp, -sp],
                    [0.0,  sp,  cp]], dtype=float)

    # Rotation about Z (roll)
    cr, sr = np.cos(r), np.sin(r)
    R_z = np.array([[ cr, -sr, 0.0],
                    [ sr,  cr, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)

    # Intrinsic rotations: R = R_z * R_x * R_y (roll, pitch, yaw) 
    R = R_z @ R_x @ R_y
    return R


def reflect_points_x_plane(points_xyz: np.ndarray, d_mm: float) -> np.ndarray:
    """
    Reflect 3D points across plane x = -d (mirror to the LEFT of camera).

    For plane normal +X (unit), the reflection of a point is:
      x' = -2d - x,  y' = y,  z' = z
    """
    pts = np.asarray(points_xyz, dtype=float).copy()
    pts[:, 0] = -2.0 * d_mm - pts[:, 0]
    return pts


def reflect_dirs_x_plane(vecs: np.ndarray) -> np.ndarray:
    """
    Reflect direction vectors across plane with normal +X.
    Only the X component flips sign.
    """
    v = np.asarray(vecs, dtype=float).copy()
    v[:, 0] *= -1.0
    return v


def project_pinhole(points_xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    Project 3D points to image (u,v) using pinhole intrinsics
    returns
    - uv
    - valid_mask : valid only if Z > 0 (in front of camera).
    """
    P = np.asarray(points_xyz, dtype=float)
    Z = P[:, 2]
    valid = Z > 1e-9
    uv = np.zeros((P.shape[0], 2), dtype=float)
    Z_safe = np.where(valid, Z, 1.0)
    uv[:, 0] = fx * (P[:, 0] / Z_safe) + cx
    uv[:, 1] = fy * (P[:, 1] / Z_safe) + cy
    return uv, valid


def jacobian_uv_wrt_panel_uv_at_point(X: float, Y: float, Z: float,
                                      r1: np.ndarray, r2: np.ndarray,
                                      fx: float, fy: float) -> np.ndarray:
    """
    Compute the 2x2 Jacobian J = d(u,v) / d(u_p, v_p) at a world point.

    For rendering dots as ellipses, we need to know how a small change
    in panel coordinates (u_p, v_p) maps to image coordinates (u,v).

    r1, r2 are the *world* tangent unit vectors corresponding to +u_p and +v_p directions
    on the panel surface (columns 0 and 1 of the panel rotation matrix).

    Using u = fx * X/Z + cx, v = fy * Y/Z + cy:
      du/du_p = fx * (Z * r1x - X * r1z) / Z^2
      du/dv_p = fx * (Z * r2x - X * r2z) / Z^2
      dv/du_p = fy * (Z * r1y - Y * r1z) / Z^2
      dv/dv_p = fy * (Z * r2y - Y * r2z) / Z^2
    """
    Z2 = Z * Z
    if Z2 <= 1e-9:
        return np.zeros((2,2), dtype=float)

    r1x, r1y, r1z = float(r1[0]), float(r1[1]), float(r1[2])
    r2x, r2y, r2z = float(r2[0]), float(r2[1]), float(r2[2])

    J = np.array([
        [fx * (Z * r1x - X * r1z) / Z2, fx * (Z * r2x - X * r2z) / Z2],
        [fy * (Z * r1y - Y * r1z) / Z2, fy * (Z * r2y - Y * r2z) / Z2],
    ], dtype=float)
    return J


def ellipse_from_jacobian(r_dot_mm: float, J: np.ndarray):
    """
    Return (a_px, b_px, angle_deg) for the image ellipse of a panel-space circle.

    - Sigma_img = r^2 * J J^T.
    - Eigenvalues -> squared axes.
    - eigenvectors -> orientation.
    
    Angle is in degrees, measured CCW from +u (image x axis) for cv2.ellipse.
    """
    # Numerical safety
    if not np.all(np.isfinite(J)):
        return 0.0, 0.0, 0.0

    S = (r_dot_mm ** 2) * (J @ J.T)  # 2x2
    # Symmetrize to reduce numerical noise
    S = 0.5 * (S + S.T)

    w, V = np.linalg.eigh(S)  # ascending order
    # Clamp negatives from numeric issues
    w = np.clip(w, 0.0, None)
    a = np.sqrt(w[1])
    b = np.sqrt(w[0])
    # Major axis direction (eigenvector associated with largest eigenvalue)
    v = V[:, 1]
    angle = float(np.degrees(np.arctan2(v[1], v[0])))
    return float(a), float(b), angle
