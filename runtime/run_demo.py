"""
Run the streaming runtime on camera / video / folder sources with OpenCV optical flow and debug windows.


Usage examples, run python from parent of runtime/

1. Video file as input, write output to files

python -m runtime.run_demo \
  --source video \
  --input data/camera_frames.avi \
  --write_videos \
  --out outputs \
  --flow dis

2. Image files Folder as input, 30fps playback speed

python -m runtime.run_demo \
  --source folder \
  --input /path/to/frames_folder \
  --fps 30 \
  --write_videos

3. Real camera (usb index 0) as input, but not writing output to file
python -m runtime.run_demo \
  --source camera \
  --device 0 \

4. Optical flow using raw image (no compensation, no unmix r,g,b)
python -m runtime.run_demo \
  --source video \
  --input /path/to/video.mp4 \
  --compensation skip \
  --flow dis \
  --unmix skip \
  --raw_flow

5. Downsampling
python -m runtime.run_demo --source video --input video.mp4 --downscale 0.5

"""

import argparse, os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from .pipeline import RuntimePipeline
from runtime.config.loader import load_yaml_config
from runtime.config.settings import (
    RuntimeConfig, SourceMode, CompensationMode, UnmixMode, FlowMethod,
    CameraConfig, VirtualSourceConfig, FlowConfig, OutputConfig, DisplayConfig
)

def parse_args():
    ap = argparse.ArgumentParser("Streaming RGB Particle Flow Demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default=None, help="Path to runtime_config.yaml")
    ap.add_argument("--source", choices=[m.value for m in SourceMode], default=SourceMode.VIDEO.value, help="camera|video|folder")
    ap.add_argument("--input", default="", help="Path to video file or folder (for non-camera modes)")
    ap.add_argument("--device", type=int, default=0, help="Camera device index")
    ap.add_argument("--width", type=int, default=0, help="Camera width (0=default)")
    ap.add_argument("--height", type=int, default=0, help="Camera height (0=default)")
    ap.add_argument("--fps", type=float, default=30.0, help="Virtual source pacing FPS (video/folder)")
    ap.add_argument("--compensation", choices=[m.value for m in CompensationMode], default=CompensationMode.BASELINE.value)
    ap.add_argument("--unmix", choices=[m.value for m in UnmixMode], default=UnmixMode.KMEANS.value)
    ap.add_argument("--flow", choices=[m.value for m in FlowMethod], default=FlowMethod.DIS.value)
    ap.add_argument("--raw_flow", action="store_true", help="Enable raw flow")
    ap.add_argument("--write_videos", action="store_true", help="Write flow videos to outputs/")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--out_fps", type=float, default=20.0, help="Output video FPS")
    ap.add_argument("--downscale", type=float, default=1.0, help="Uniform image downscale in (0,1]")
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after N frames; 0 = unlimited")
    # display
    ap.add_argument("--no_display", action="store_true", help="Disable debug windows")
    return ap.parse_args()


def cli_overlay(cfg, args):
    if args.source is not None:
        cfg.source_mode = SourceMode(args.source)
    if args.input:
        cfg.input_path = args.input
    if args.device is not None:
        cfg.camera.device_index = args.device
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.fps is not None:
        cfg.virtual.fps = args.fps

    if args.compensation:
        cfg.compensation_mode = CompensationMode(args.compensation)
    if args.unmix:
        cfg.unmix_mode = UnmixMode(args.unmix)
    if args.flow:
        cfg.flow.method = FlowMethod(args.flow)

    # tri-state booleans
    if args.raw_flow is not None:
        cfg.do_raw_flow = args.raw_flow
    if args.display_enable is not None:
        cfg.display.enable = args.display_enable

    # optional convenience overrides
    if args.out:
        cfg.output.dir = args.out
    if args.out_fps is not None:
        cfg.output.fps = args.out_fps
    if args.downscale is not None:
        cfg.downscale = args.downscale
    if args.max_frames:
        cfg.max_frames = args.max_frames    


def main():
    a = parse_args()

    ## 1) defaults
    #cfg = RuntimeConfig()
    ## 2) YAML overlay (if provided)
    #if a.config:
    #    cfg = load_yaml_config(a.config)
    ## 3) CLI overlay
    #cli_overlay(cfg, a)

    cfg = RuntimeConfig(
      source_mode=SourceMode(a.source),
      input_path=a.input,
      camera=CameraConfig(device_index=a.device, width=a.width or None, height=a.height or None, fps=None),
      virtual=VirtualSourceConfig(fps=a.fps, loop=False),
      compensation_mode=CompensationMode(a.compensation),
      unmix_mode=UnmixMode(a.unmix),
      flow=FlowConfig(method=FlowMethod(a.flow), incremental=False),
      do_raw_flow=a.raw_flow,
      output=OutputConfig(write_videos=a.write_videos, dir=a.out, fps=a.out_fps),
      display=DisplayConfig(enable=not a.no_display),
      downscale=a.downscale,
      max_frames=(a.max_frames or None),
    )

    pipe = RuntimePipeline(cfg)
    outs = pipe.run()
    
    print("Outputs:")
    for k,v in outs.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
