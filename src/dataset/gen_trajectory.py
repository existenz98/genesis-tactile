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
Generate full sequence of data from a keyframe spec:
- Reads spec.yaml to define trajectories.
- For each frame:
    1) builds 3D surface force field (fx, fy, fz),
    2) runs forward FEM (scripts/run_forward.py),
    3) renders the camera image for the gel (scripts/render_sensor_frame.py),
    4) (optional) computes optical flow (current frame vs original I0).,
- Saves per-frame folders and global metadata.


Notes:
- Trajectory is only from spec.yaml
- Friction rule |shear| ≤ μ·Fz and edge clamping are enforced per frame
- With logging in timeline.json and frame_params.json.
- If want σ to breathe with Fz, just need to interpolate sigma_mm in keyframes


Usage example:
python src/dataset/gen_trajectory.py \
  --outdir dataset/sequences/000001 \
  --material src/config/material.yaml \
  --sensor particle_vts \
  --render-config src/config/renderer.yaml \
  --spec src/config/trajectories/press_slide_lift.yaml \
  --save_flow \
  --write_video

This will produce:
outdir/
  I0.png
  meta.json
  timeline.json
  frames/
    000000/{force_top.npz, u.xdmf, u.dofs.npz, I.png, renderer_used.yaml, frame_params.json, ...}
    000001/...
    ...
  flow/ (if --save_flow)
    000001.npz
    000001.png
    000001.quiv.png
  video/ (if --write_video)
    rgb/000000.png ...
    sequence.mp4

Generate gelsight style or tac3d style sequences by changing --sensor and --render-config.
e.g.

gelsight style:

python src/dataset/gen_trajectory.py \
  --outdir dataset/sequences/000001 \
  --material src/config/material.yaml \
  --sensor gelsight_style \
  --render-config src/config/renderer_gelsight.yaml \
  --spec src/config/trajectories/press_slide_lift.yaml \
  --save_flow \
  --write_video

  
tac3d style:

python src/dataset/gen_trajectory.py \
  --outdir dataset/sequences/tac3d_press_slide_lift  \
  --material src/config/material_tac3d.yaml \
  --sensor stereodots  \
  --render-config src/config/renderer_tac3d.yaml  \
  --spec src/config/trajectories/press_slide_lift.yaml \
  --write_video


"""

from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml
import cv2

# Ensure src/ is on path
THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]  # ../src
sys.path.insert(0, str(SRC_ROOT))

from dataset.trajectory_core import (
    load_spec_yaml, sample_timeline, build_force_field,
)

# viz helpers
from synth.optical_flow import FlowConfig, FlowMethod, to_gray_f32_bgr, compute_flow, flow_to_bgr
from viz.quiver import draw_quiver_bgr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True, help="Output sequence directory.")
    p.add_argument("--material", type=Path, required=True, help="YAML for FEM (forward).")
    p.add_argument("--sensor", type=str, default="particle_vts", choices=["particle_vts", "gelsight_style", "stereodots"], help="Registered sensor name (default: particle_vts).")
    p.add_argument("--render-config", type=Path, required=True, help="YAML for sensor renderer.")
    p.add_argument("--spec", type=Path, required=True, help="Keyframe specification YAML.")
    p.add_argument("--save_flow", action="store_true", help="Compute optical flow (current frame vs original I0).")
    p.add_argument("--write_video", action="store_true", help="Assemble MP4 from rendered frames (requires ffmpeg).")
    p.add_argument("--supersample", type=int, default=2, help="Renderer supersample factor (passed to render script).")
    p.add_argument("--debug_show", action="store_true")
    return p.parse_args()


def _render_I0(sensor: str, render_cfg: Path, out_png: Path, supersample: int, seed: int = 0):
    """
    Render undeformed reference frame I0.
    """
    cmd = [
        sys.executable, "src/scripts/render_sensor_frame.py",
        "--sensor", sensor,
        "--config", str(render_cfg),
        "--out", str(out_png),
        "--seed", str(seed),
        "--supersample", str(supersample),
    ]
    subprocess.run(cmd, check=True)


def _render_deformed(sensor: str, base_render_cfg: Path, xdmf: Path, out_png: Path,
                     supersample: int, tmp_yaml: Path, seed: int = 0):
    """
    Render deformed frame given deformation field in XDMF.
    - base_render_cfg: original YAML without deformation section.
    - tmp_yaml: temporary YAML file to write (with deformation section added).
    """
    with open(base_render_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    cfg["deformation"] = {"mode": "xdmf", "xdmf_path": str(xdmf)}
    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(cfg, f)
    cmd = [
        sys.executable, "src/scripts/render_sensor_frame.py",
        "--sensor", sensor,
        "--config", str(tmp_yaml),
        "--out", str(out_png),
        "--seed", str(seed),
        "--supersample", str(supersample),
    ]
    subprocess.run(cmd, check=True)


def _run_forward(force_npz: Path, material_yaml: Path, xdmf_out: Path, dofs_npz: Path,
                 disp_prefix: Path, Nx: int, Ny: int):
    """
    Run forward FEM to compute displacement field given surface force.
    Saves:
      - xdmf_out: XDMF file for displacement field (for renderer).
    """
    cmd = [
        sys.executable, "src/scripts/run_forward.py",
        "--force", str(force_npz),
        "--config", str(material_yaml),
        "--xdmf", str(xdmf_out),
        "--viz_prefix", str(disp_prefix),
        "--sample_Nx", str(Nx),
        "--sample_Ny", str(Ny),
        "--save_dofs_npz", str(dofs_npz),
        "--ksp", "cg", "--pc", "gamg", "--ksp_monitor"
    ]
    subprocess.run(cmd, check=True)


def _compute_flow(prev_png: Path, curr_png: Path, out_npz: Path, out_vis: Path, out_quiver: Path):
    """
    Compute optical flow from prev_png to curr_png.
    Saves:
      - out_npz: numeric flow field (vx, vy)
      - out_vis: color wheel visualization
      - out_quiver: quiver plot visualization
    """
    I0p = cv2.imread(str(prev_png), cv2.IMREAD_COLOR)
    I1p = cv2.imread(str(curr_png), cv2.IMREAD_COLOR)
    if I0p is None or I1p is None:
        raise RuntimeError(f"Failed to read frames for flow: {prev_png}, {curr_png}")

    g0 = to_gray_f32_bgr(I0p)
    g1 = to_gray_f32_bgr(I1p)
    cfg_flow = FlowConfig(method=FlowMethod("dis"))
    vx, vy = compute_flow(g0, g1, cfg_flow)  # From prev -> curr

    # numeric
    np.savez_compressed(out_npz, vx=vx, vy=vy, note=f"flow from {prev_png.name} to {curr_png.name}")

    # color wheel visualize
    vis = flow_to_bgr(vx, vy, clip_mag=None)
    cv2.imwrite(str(out_vis), vis)

    # quiver
    quiv = draw_quiver_bgr(
        vy, vx,
        block=16,
        pool=1,
        scale=4.0,
        thickness=1,
        color=(0, 255, 255),
        bg=(0, 0, 0),
        min_px=0.5,
        draw_centers=False,
        center_color=(255, 255, 255),
    )
    cv2.imwrite(str(out_quiver), quiv)


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load spec (ONLY source of trajectory)
    spec = load_spec_yaml(args.spec)

    # 2) Build per-frame timeline from spec (params per object)
    timeline = sample_timeline(spec)
    frames = timeline["frames"]
    stats = timeline["stats"]

    # 3) Save global metadata
    meta = dict(
        spec_yaml=str(args.spec),
        sensor=args.sensor,
        material=str(args.material),
        render_config=str(args.render_config),
        fps=int(spec.fps),
        duration_s=float(spec.duration_s),
        Nx=int(spec.Nx), Ny=int(spec.Ny),
        Lx_mm=float(spec.Lx_mm), Ly_mm=float(spec.Ly_mm),
        mu_friction=float(spec.mu_friction),
        sigma_margin_factor=float(spec.sigma_margin_factor),
        edge_margin_mm=float(spec.edge_margin_mm),
        timeline_stats=stats
    )
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 4) Render undeformed reference once
    I0 = outdir / "I0.png"
    _render_I0(args.sensor, args.render_config, I0, supersample=args.supersample, seed=0)

    # Prepare dirs
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = outdir / "flow"
    if args.save_flow:
        flow_dir.mkdir(parents=True, exist_ok=True)

    # 5) Iterate frames
    t0_all = time.time()
    prev_png = None
    for k, fr in enumerate(frames):
        t_frame = fr["t"]
        fdir = frames_dir / f"{k:06d}"
        fdir.mkdir(parents=True, exist_ok=True)

        print(f"[starting frame {k:06d}] t={t_frame:.4f}s  objects={len(fr['objects'])}")

        # 5a) Build surface force field
        force = build_force_field(spec.Lx_mm, spec.Ly_mm, spec.Nx, spec.Ny, fr["objects"])
        force_npz = fdir / "force_top.npz"
        np.savez_compressed(force_npz, **force, unit="MPa", t=float(t_frame))

        # 5b) Forward FEM
        xdmf = fdir / "u.xdmf"
        dofs = fdir / "u.dofs.npz"
        disp_prefix = fdir / "disp_top"
        t0 = time.time()
        _run_forward(force_npz, args.material, xdmf, dofs, disp_prefix, spec.Nx, spec.Ny)
        t1 = time.time()

        # 5c) Render deformed frame
        I_png = fdir / "I.png"
        rtmp = fdir / "renderer_used.yaml"
        _render_deformed(args.sensor, args.render_config, xdmf, I_png, args.supersample, rtmp, seed=0)

        # 5d) Flow (I0 -> curr)
        if args.save_flow and I0 is not None:
            out_npz = flow_dir / f"{k:06d}.npz"
            out_vis = flow_dir / f"{k:06d}.png"
            out_quiv = flow_dir / f"{k:06d}.quiv.png"
            _compute_flow(I0, I_png, out_npz, out_vis, out_quiv)
        #I0 = I_png

        # Per-frame param log
        with open(fdir / "frame_params.json", "w") as f:
            json.dump(dict(t=t_frame, objects=fr["objects"], log=fr["log"], fem_time_s=t1-t0), f, indent=2)

        # Debug show
        if args.debug_show:
            img = cv2.imread(str(I_png))
            if img is not None:
                cv2.imshow("I (deformed)", img)
                cv2.waitKey(1)

        print(f"[frame {k:06d}] t={t_frame:.4f}s  FEM {t1-t0:.2f}s  -> {I_png.name}")

    if args.debug_show:
        print("Press any key to close preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 6) Save the whole sampled timeline
    #    Store for later analysis.
    timeline_compact = dict(
        frames=[dict(t=f["t"], objects=f["objects"]) for f in frames],
        stats=stats
    )
    with open(outdir / "timeline.json", "w") as f:
        json.dump(timeline_compact, f, indent=2)

    # 7) Optional: assemble MP4 from frames/*/I.png
    if args.write_video:
        print("[video] assembling MP4 from frames...")
        vid_dir = outdir / "video"
        vid_dir.mkdir(parents=True, exist_ok=True)
        # ffmpeg likes flat patterns; copy into a temp flat folder
        flat = vid_dir / "rgb"
        if flat.exists():
            shutil.rmtree(flat)
        flat.mkdir(parents=True, exist_ok=True)
        # Copy (not symlink for portability)
        for k in range(len(frames)):
            src = frames_dir / f"{k:06d}" / "I.png"
            dst = flat / f"{k:06d}.png"
            shutil.copy2(src, dst)

        if False:
            # standard h264, small file size, but pretty lossy
            mp4_path = vid_dir / "sequence.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(spec.fps),
                "-i", str(flat / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(mp4_path)
            ]
        else:
            # lossless RGB, larger file size
            mp4_path = vid_dir / "sequence.mkv"
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(spec.fps),
                "-i", str(flat / "%06d.png"),
                "-c:v", "ffv1",           # mathematically lossless
                "-level", "3",            # modern bitstream
                "-g", "1",                # intra-only frames (optional)
                str(mp4_path.with_suffix(".mkv"))
            ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[video] wrote {mp4_path}")
        except Exception as e:
            print(f"[video] ffmpeg failed: {e}")

    t1_all = time.time()
    print(f"[OK] sequence saved to {outdir} in {t1_all - t0_all:.1f}s")


if __name__ == "__main__":
    main()
