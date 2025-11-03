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
Run the streaming runtime on camera / video / folder sources with OpenCV optical flow and debug windows.


Usage examples, run python from parent of runtime/

1. Video file as input.
no compensation, no unmix r,g,b,  show debug visualizations

python -m runtime.run_demo \
  --source video \
  --input data/raw/camera_frames.avi \
  --flow dis \
  --compensation skip \
  --unmix skip \
  --vis2d   --vis3d


2. Video file as input, write output to files
with compensation and unmixing

python -m runtime.run_demo \
  --config runtime/config/runtime_config.yaml \
  --source video \
  --input data/raw/camera_frames.avi \
  --vis2d   --vis3d \
  --write_videos \
  --out outputs

3. Gelsight-style photometric sensor demo
python -m runtime.run_demo \
  --source video   --input dataset/sequences/gelsight_press_slide_lift/video/sequence.mp4   --loop \
  --config runtime/config/runtime_config_gelsight.yaml \
  --vis2d --vis3d

3. Image files Folder as input, 30fps playback speed

python -m runtime.run_demo \
  --source folder \
  --input /path/to/frames_folder \
  --fps 30 \
  --write_videos

4. Real camera (usb index 0) as input, but not writing output to file
python -m runtime.run_demo \
  --source camera \
  --device 0


5. Downsampling
python -m runtime.run_demo --source video --input video.mp4 --downscale 0.5

"""

import argparse, os, sys
import threading, signal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from .pipeline import RuntimePipeline
from runtime.config.loader import load_yaml_config
from runtime.config.settings import (
    RuntimeConfig, SourceMode, CompensationMode, UnmixMode, FlowMethod,
    CameraConfig, VirtualSourceConfig, FlowConfig, OutputConfig, DisplayConfig
)
from runtime.core.frame_bus import FrameBus
from runtime.output.vis3d_pyvista import Vis3DLive



def parse_args():
    ap = argparse.ArgumentParser("Streaming RGB Particle Flow Demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    SUPPRESS = argparse.SUPPRESS
    ap.add_argument("--config", default=None, help="Path to runtime_config.yaml")
    ap.add_argument("--source", choices=[m.value for m in SourceMode], default=None, help="camera|video|folder")
    ap.add_argument("--input", default=None, help="Path to video file or folder (for non-camera modes)")
    ap.add_argument("--device", type=int, default=None, help="Camera device index")
    ap.add_argument("--width", type=int, default=None, help="Camera width (None=default)")
    ap.add_argument("--height", type=int, default=None, help="Camera height (None=default)")
    ap.add_argument("--fps", type=float, default=None, help="Virtual source pacing FPS (video/folder)")
    ap.add_argument("--loop", action="store_true", default=SUPPRESS, help="Loop video/folder source") # Boolean toggles as *tri-state*: absent = don't touch; present sets value.
    ap.add_argument("--compensation", choices=[m.value for m in CompensationMode], default=None)
    ap.add_argument("--unmix", choices=[m.value for m in UnmixMode], default=None)
    ap.add_argument("--flow", choices=[m.value for m in FlowMethod], default=None)
    ap.add_argument("--write_videos", action="store_true", default=SUPPRESS, help="Write flow videos to outputs/") # Boolean toggles as *tri-state*: absent = don't touch; present sets value.
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--out_fps", type=float, default=None, help="Output video FPS")
    ap.add_argument("--downscale", type=float, default=None, help="Uniform image downscale in (0,1]")
    ap.add_argument("--max_frames", type=int, default=None, help="Stop after N frames; 0 = unlimited")
    # display
    ap.add_argument("--vis2d", action="store_true", default=SUPPRESS, help="Display 2D Debug Windows") # Boolean toggles as *tri-state*: absent = don't touch; present sets value.
    ap.add_argument("--vis3d", action="store_true", default=SUPPRESS, help="Display 3D Debug Windows") # Boolean toggles as *tri-state*: absent = don't touch; present sets value.
    return ap.parse_args()


def cli_overlay(cfg, args):
    """ Overlay CLI args onto loaded config """

    if args.source is not None: cfg.source_mode = SourceMode(args.source)
    if args.input is not None: cfg.input_path = args.input

    # Camera/virtual
    if args.device is not None: cfg.camera.device_index = args.device
    if args.width is not None: cfg.camera.width = args.width
    if args.height is not None: cfg.camera.height = args.height
    if args.fps is not None: cfg.virtual.fps = args.fps
    if hasattr(args, "loop"): cfg.virtual.loop = args.loop

    # Modes
    if args.compensation is not None: cfg.compensation_mode = CompensationMode(args.compensation)
    if args.unmix is not None: cfg.unmix.mode = UnmixMode(args.unmix)
    if args.flow is not None: cfg.flow.method = FlowMethod(args.flow)

    # Debug display
    if hasattr(args, "vis2d"): cfg.display.enable = args.vis2d
    if hasattr(args, "vis3d"): cfg.vis3d.enable = args.vis3d

    # Output debug video
    if hasattr(args, "write_videos"): cfg.output.write_videos = args.write_videos
    if args.out is not None: cfg.output.dir = args.out
    if args.out_fps is not None: cfg.output.fps = args.out_fps

    if args.downscale is not None: cfg.downscale = args.downscale
    if args.max_frames is not None: cfg.max_frames = args.max_frames


def main():
    a = parse_args()

    # 1) defaults
    cfg = RuntimeConfig()

    # 2) YAML overlay
    if a.config is not None:
        cfg = load_yaml_config(a.config)

    # 3) CLI overlay
    cli_overlay(cfg, a)

    # Create the shared latest-value bus (between algo pipeline and 3D visualization)
    bus = FrameBus()


    # stop signal shared by pipeline 3d viewer
    stop_event = threading.Event()


    # Start the pipeline on a worker thread
    pipe = RuntimePipeline(cfg)
    t = threading.Thread(target=lambda: pipe.run(bus), name="PipelineThread", daemon=True)
    t.start()


    # --- set OS signal handlers to trigger a graceful shutdown ---
    def _handle_signal(signum, frame):
        print("\nShutting down…", flush=True)
        stop_event.set()
        try:
            pipe.request_stop()
        except AttributeError:
            pass
        try:
            viewer.request_close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Start the 3D viewer on the MAIN thread (blocking UI loop)
    try:
        if cfg.vis3d.enable:
            viewer = Vis3DLive(bus=bus, topic=cfg.vis3d.topic, cfg=cfg.vis3d, normal_gain=cfg.physics.normal_gain)
            print("viewer.start_blocking() >>>")
            viewer.start_blocking()
            print("viewer.start_blocking() <<<")
        else:
            # If 3D disabled, just join pipeline or add a CLI loop.  idle here but remain interruptible
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        # ensure clean path even if viewer raised KeyboardInterrupt
        print("\nKeyboardInterrupt: stopping…", flush=True)
        stop_event.set()
        try:
            pipe.request_stop()
        except AttributeError:
            pass
        try:
            viewer.request_close()
        except Exception:
            pass
        
    finally:
        
        # join pipeline (with a short timeout) and exit cleanly
        t.join(timeout=2.0)

        # if the thread didn’t die, avoid ugly aborts by exiting now
        if t.is_alive():
            print("Pipeline did not exit in time; forcing shutdown.", flush=True)

        # TODO bus cleanup

        sys.exit(0)  # prevents 'terminate called without an active exception' messages.

if __name__ == "__main__":
    main()
