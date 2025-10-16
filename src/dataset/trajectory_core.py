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
Core feature: Generate 3D force field from keyframe spec.

- loads/validates spec
- flattens keyframes
- interpolates per frame
- enforces bounds + friction cap
- builds per-frame force params.
"""




from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import json
import yaml
import numpy as np

# load builders
from synth.loads import make_surface_grid, multi_gaussian_combo


# ---------------------------
# Helper: curve easing functions
# ---------------------------
def _smoothstep(a: float) -> float:
    return a * a * (3.0 - 2.0 * a)

def _smootherstep(a: float) -> float:
    return a * a * a * (a * (6.0 * a - 15.0) + 10.0)

def _cosine_ease(a: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * a))

def _apply_ease(a: float, kind: str) -> float:
    if kind == "linear":
        return a
    if kind == "smoothstep":
        return _smoothstep(a)
    if kind == "smootherstep":
        return _smootherstep(a)
    if kind == "cosine":
        return _cosine_ease(a)
    if kind == "hold":
        return 0.0
    # default
    return _smootherstep(a)


# ---------------------------
# Angle interpolation (degrees, shortest path)
# ---------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b

def _angle_lerp_deg(a0: float, a1: float, t: float) -> float:
    # Map delta to [-180,180)
    d = ((a1 - a0 + 180.0) % 360.0) - 180.0
    ang = a0 + d * t
    # Normalize to [0,360)
    return ang % 360.0


# ---------------------------
# Trajectory Spec structures
# ---------------------------
@dataclass
class Keyframe:
    t: float
    cx_mm: Optional[float] = None
    cy_mm: Optional[float] = None
    fz_mpa: Optional[float] = None
    shear_mpa: Optional[float] = None
    shear_dir_deg: Optional[float] = None
    torque_mpa: Optional[float] = None
    sigma_mm: Optional[float] = None
    ease_to_next: str = "smootherstep"  # linear|smoothstep|smootherstep|cosine|hold


@dataclass
class ObjectSpec:
    # Defaults can be overridden by keyframes
    sigma_mm: float
    keyframes: List[Keyframe]


@dataclass
class TrajectorySpec:
    duration_s: float
    fps: int
    mu_friction: float
    Lx_mm: float
    Ly_mm: float
    Nx: int
    Ny: int
    edge_margin_mm: float
    sigma_margin_factor: float
    n_objects: int
    objects: List[ObjectSpec]
    include_endpoint: bool = True  # produce a frame exactly at t=duration_s


# ---------------------------
# Loading and validation
# ---------------------------
_REQUIRED_TOP = [
    "duration_s",
    "fps",
    "mu_friction",
    "Lx_mm",
    "Ly_mm",
    "Nx",
    "Ny",
    "edge_margin_mm", "sigma_margin_factor",
    "objects"
]

def load_spec_yaml(spec_path) -> TrajectorySpec:
    with open(spec_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Provide simple defaults for a few that are not trajectory (ok to default)
    cfg.setdefault("edge_margin_mm", 2.0)
    cfg.setdefault("sigma_margin_factor", 2.5)
    cfg.setdefault("Nx", 80)
    cfg.setdefault("Ny", 60)
    cfg.setdefault("Lx_mm", 40.0)
    cfg.setdefault("Ly_mm", 30.0)
    cfg.setdefault("mu_friction", 0.8)
    cfg.setdefault("include_endpoint", True)

    # Validate top level
    for k in _REQUIRED_TOP:
        if k not in cfg:
            raise ValueError(f"Missing required field in spec: '{k}'")

    # objects and keyframes
    objs: List[ObjectSpec] = []
    for i, od in enumerate(cfg["objects"]):
        if "sigma_mm" not in od:
            raise ValueError(f"objects[{i}] missing 'sigma_mm'")
        if "keyframes" not in od or not od["keyframes"]:
            raise ValueError(f"objects[{i}] must have non-empty 'keyframes'")
        kfs: List[Keyframe] = []
        for j, kd in enumerate(od["keyframes"]):
            if "t" not in kd:
                raise ValueError(f"objects[{i}].keyframes[{j}] missing 't'")
            kfs.append(Keyframe(
                t=float(kd["t"]),
                cx_mm=kd.get("cx_mm"),
                cy_mm=kd.get("cy_mm"),
                fz_mpa=kd.get("fz_mpa"),
                shear_mpa=kd.get("shear_mpa"),
                shear_dir_deg=kd.get("shear_dir_deg"),
                torque_mpa=kd.get("torque_mpa"),
                sigma_mm=kd.get("sigma_mm"),
                ease_to_next=kd.get("ease_to_next", "smootherstep"),
            ))

        # Sort keyframes by time
        kfs.sort(key=lambda k: k.t)

        # Basic checks
        if kfs[0].t > 0.0 + 1e-9:
            raise ValueError(f"objects[{i}] first keyframe must start at t=0 (got {kfs[0].t})")
        if any(kfs[j+1].t <= kfs[j].t for j in range(len(kfs)-1)):
            raise ValueError(f"objects[{i}] keyframe times must be strictly increasing.")
        objs.append(ObjectSpec(sigma_mm=float(od["sigma_mm"]), keyframes=kfs))

    return TrajectorySpec(
        duration_s=float(cfg["duration_s"]),
        fps=int(cfg["fps"]),
        mu_friction=float(cfg["mu_friction"]),
        Lx_mm=float(cfg["Lx_mm"]),
        Ly_mm=float(cfg["Ly_mm"]),
        Nx=int(cfg["Nx"]),
        Ny=int(cfg["Ny"]),
        edge_margin_mm=float(cfg["edge_margin_mm"]),
        sigma_margin_factor=float(cfg["sigma_margin_factor"]),
        n_objects=len(objs),
        objects=objs,
        include_endpoint=bool(cfg.get("include_endpoint", True))
    )


# ---------------------------
# Keyframe flattening
# ---------------------------
_ALL_FIELDS = ["cx_mm", "cy_mm", "fz_mpa", "shear_mpa", "shear_dir_deg", "torque_mpa", "sigma_mm"]

def _flatten_object_keyframes(obj: ObjectSpec) -> List[Dict]:
    """
    Returns a list of keyframes
    - where each kf has all fields present,
    - with 'ease_to_next' preserved.
    """
    flat = []
    cur = dict(
        cx_mm=None, cy_mm=None, fz_mpa=0.0,
        shear_mpa=0.0, shear_dir_deg=0.0, torque_mpa=0.0,
        sigma_mm=obj.sigma_mm
    )

    for kf in obj.keyframes:
        # start from 'cur'
        k = dict(cur)
        # override with provided values
        if kf.cx_mm is not None: k["cx_mm"] = float(kf.cx_mm)
        if kf.cy_mm is not None: k["cy_mm"] = float(kf.cy_mm)
        if kf.fz_mpa is not None: k["fz_mpa"] = float(kf.fz_mpa)
        if kf.shear_mpa is not None: k["shear_mpa"] = float(kf.shear_mpa)
        if kf.shear_dir_deg is not None: k["shear_dir_deg"] = float(kf.shear_dir_deg) % 360.0
        if kf.torque_mpa is not None: k["torque_mpa"] = float(kf.torque_mpa)
        if kf.sigma_mm is not None: k["sigma_mm"] = float(kf.sigma_mm)
        k["t"] = float(kf.t)
        k["ease_to_next"] = kf.ease_to_next
        flat.append(k)
        cur = k  # carry forward

    # sanity: all fields must be set
    for j, k in enumerate(flat):
        for f in ["cx_mm", "cy_mm", "sigma_mm"]:
            if k[f] is None:
                raise ValueError(f"Keyframe {j} missing required field '{f}' after flattening.")
    return flat


# ---------------------------
# Safe bounds per object (keep gaussian mass inside)
# ---------------------------
def _safe_bounds(Lx: float, Ly: float, sigma_mm: float,
                 factor: float, edge_min: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    m = max(edge_min, factor * sigma_mm)
    m = min(m, min(Lx, Ly) * 0.49)  # keep positive area
    return (m, Lx - m), (m, Ly - m)


# ---------------------------
# Frame sampling
# ---------------------------
def _interp_segment(k0: Dict, k1: Dict, t: float) -> Dict:
    """
    Interpolate between 2 FULL keyframes (all fields present).
    """
    t0, t1 = k0["t"], k1["t"]
    if t1 <= t0 + 1e-12:
        alpha = 0.0
    else:
        alpha = (t - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))

    ease = k0.get("ease_to_next", "smootherstep")
    u = _apply_ease(alpha, ease)

    out = {}
    out["t"] = t
    out["ease_from_k0"] = ease

    # scalars
    out["fz_mpa"]      = _lerp(k0["fz_mpa"],      k1["fz_mpa"],      u)
    out["shear_mpa"]   = _lerp(k0["shear_mpa"],   k1["shear_mpa"],   u)
    out["torque_mpa"]  = _lerp(k0["torque_mpa"],  k1["torque_mpa"],  u)
    out["sigma_mm"]    = _lerp(k0["sigma_mm"],    k1["sigma_mm"],    u)

    # positions & angles
    out["cx_mm"] = _lerp(k0["cx_mm"], k1["cx_mm"], u)
    out["cy_mm"] = _lerp(k0["cy_mm"], k1["cy_mm"], u)
    out["shear_dir_deg"] = _angle_lerp_deg(k0["shear_dir_deg"], k1["shear_dir_deg"], u)
    return out


def _find_segment_idx(kfs: List[Dict], t: float) -> int:
    # return index i such that kfs[i].t <= t <= kfs[i+1].t
    n = len(kfs)
    if t <= kfs[0]["t"]:
        return 0
    for i in range(n - 1):
        if kfs[i]["t"] <= t <= kfs[i + 1]["t"]:
            return i
    return n - 2  # clamp at end


def sample_timeline(spec: TrajectorySpec) -> Dict:
    """
    Returns a dict with:
      - "frames": list[ { "t": float, "objects": [per-object dict], "log": {...} } ]
      - "N": number of frames
      - "dt": timestep

    Each per-object dict has fields required by multi_gaussian_combo() call:
      { cx_mm, cy_mm, sigma_mm, fz_peak_mpa, tau_shear_mpa, shear_dir_deg, tau_torque_mpa }
    """
    # Determine frame count and times
    if spec.include_endpoint:
        N = int(round(spec.duration_s * spec.fps)) + 1
    else:
        N = int(round(spec.duration_s * spec.fps))
    N = max(N, 2)
    dt = 1.0 / float(spec.fps)
    times = [min(k * dt, spec.duration_s) for k in range(N)]

    # Prepare grids once (returned via build_force later if desired)
    frames = []
    capping_events = 0
    clamp_events = 0

    # Pre-flatten all objects
    flat_objs = [ _flatten_object_keyframes(obj) for obj in spec.objects ]

    # Sample
    for t in times:
        objs_for_force = []
        logs = dict(capped_shear=False, clamped_center=False)
        per_obj_logs = []

        for i, kfs in enumerate(flat_objs):
            j = _find_segment_idx(kfs, t)
            k0, k1 = kfs[j], kfs[min(j + 1, len(kfs) - 1)]
            s = _interp_segment(k0, k1, t)

            # Bounds clamp
            (xlo, xhi), (ylo, yhi) = _safe_bounds(
                spec.Lx_mm, spec.Ly_mm, s["sigma_mm"],
                spec.sigma_margin_factor, spec.edge_margin_mm
            )
            cx_raw, cy_raw = s["cx_mm"], s["cy_mm"]
            cx = min(max(cx_raw, xlo), xhi)
            cy = min(max(cy_raw, ylo), yhi)
            clamped = (abs(cx - cx_raw) > 1e-8) or (abs(cy - cy_raw) > 1e-8)
            if clamped:
                logs["clamped_center"] = True
                clamp_events += 1

            # Friction cap: |shear| <= mu * Fz
            fz = max(0.0, s["fz_mpa"])
            shear = max(0.0, s["shear_mpa"])
            cap = spec.mu_friction * fz
            if shear > cap + 1e-12:
                shear = cap
                logs["capped_shear"] = True
                capping_events += 1
            if fz <= 0.0:
                shear = 0.0

            objs_for_force.append(dict(
                cx_mm=cx, cy_mm=cy, sigma_mm=s["sigma_mm"],
                fz_peak_mpa=fz,
                tau_shear_mpa=shear,
                shear_dir_deg=s["shear_dir_deg"],
                tau_torque_mpa=max(0.0, s["torque_mpa"])
            ))
            per_obj_logs.append(dict(
                cx_mm=cx, cy_mm=cy, sigma_mm=s["sigma_mm"],
                fz_mpa=fz, shear_mpa=shear, shear_dir_deg=s["shear_dir_deg"],
                torque_mpa=max(0.0, s["torque_mpa"])
            ))

        frames.append(dict(
            t=t,
            objects=objs_for_force,
            log=dict(**logs, objects=per_obj_logs)
        ))

    stats = dict(
        N=N, dt=dt, duration_s=spec.duration_s,
        capping_events=capping_events,
        clamp_events=clamp_events
    )
    return dict(frames=frames, stats=stats)


# ---------------------------
# Force builder (grid + field)
# ---------------------------
def build_force_field(Lx_mm: float, Ly_mm: float, Nx: int, Ny: int,
                      objects: List[Dict]) -> Dict[str, np.ndarray]:
    """
    objects: list of dicts with keys (cx_mm, cy_mm, sigma_mm, fz_peak_mpa, tau_shear_mpa, shear_dir_deg, tau_torque_mpa)
    returns dict with X_mm, Y_mm, fx_mpa, fy_mpa, fz_mpa
    """
    X, Y = make_surface_grid(Lx_mm, Ly_mm, Nx, Ny)
    fx, fy, fz = multi_gaussian_combo(X, Y, objects)
    return dict(X_mm=X, Y_mm=Y, fx_mpa=fx, fy_mpa=fy, fz_mpa=fz)
