"""
Run the streaming runtime on camera / video / folder sources with OpenCV optical flow and debug windows.


Usage examples, run python from parent of runtime/

1. Video file as input, write output to files

python -m runtime.run_demo \
  --source video \
  --input data/camera_frames.avi \
  --color_flow \
  --write_videos \
  --out outputs \
  --flow dis

2. Image files Folder as input, 30fps playback speed

python -m runtime.run_demo \
  --source folder \
  --input /path/to/frames_folder \
  --fps 30 \
  --color_flow \
  --write_videos

3. Real camera (usb index 0) as input, but not writing output to file
python -m runtime.run_demo \
  --source camera \
  --device 0 \
  --color_flow

4. Optical flow using raw image (instead of unmixed r,g,b), no debug window (good for headless)
python -m runtime.run_demo \
  --source video \
  --input /path/to/video.mp4 \
  --compensation skip \
  --unmix skip \
  --raw_flow \
  --no_display \
  --write_videos

5. Downsampling
python -m runtime.run_demo --source video --input video.mp4 --downscale 0.5 --color_flow

"""

import argparse, os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from .pipeline import RuntimePipeline
from .config.settings import (
    RuntimeConfig, SourceMode, CompensationMode, UnmixMode, FlowMethod,
    CameraConfig, VirtualSourceConfig, FlowConfig, OutputConfig, DisplayConfig
)

def parse_args():
    ap = argparse.ArgumentParser("Streaming RGB Particle Flow Demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", choices=[m.value for m in SourceMode], default=SourceMode.VIDEO.value, help="camera|video|folder")
    ap.add_argument("--input", default="", help="Path to video file or folder (for non-camera modes)")
    ap.add_argument("--device", type=int, default=0, help="Camera device index")
    ap.add_argument("--width", type=int, default=0, help="Camera width (0=default)")
    ap.add_argument("--height", type=int, default=0, help="Camera height (0=default)")
    ap.add_argument("--fps", type=float, default=30.0, help="Virtual source pacing FPS (video/folder)")
    ap.add_argument("--compensation", choices=[m.value for m in CompensationMode], default=CompensationMode.BASELINE.value)
    ap.add_argument("--unmix", choices=[m.value for m in UnmixMode], default=UnmixMode.KMEANS.value)
    ap.add_argument("--flow", choices=[m.value for m in FlowMethod], default=FlowMethod.DIS.value)
    ap.add_argument("--color_flow", action="store_true", help="Enable per-color flow")
    ap.add_argument("--raw_flow", action="store_true", help="Enable raw flow")
    ap.add_argument("--write_videos", action="store_true", help="Write flow videos to outputs/")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--out_fps", type=float, default=20.0, help="Output video FPS")
    ap.add_argument("--downscale", type=float, default=1.0, help="Uniform image downscale in (0,1]")
    ap.add_argument("--max_frames", type=int, default=0, help="Stop after N frames; 0 = unlimited")
    # display
    ap.add_argument("--no_display", action="store_true", help="Disable debug windows")
    return ap.parse_args()

def main():
    a = parse_args()

    cfg = RuntimeConfig(
      source_mode=SourceMode(a.source),
      input_path=a.input,
      camera=CameraConfig(device_index=a.device, width=a.width or None, height=a.height or None, fps=None),
      virtual=VirtualSourceConfig(fps=a.fps, loop=False),
      compensation_mode=CompensationMode(a.compensation),
      unmix_mode=UnmixMode(a.unmix),
      flow=FlowConfig(method=FlowMethod(a.flow), vis_flow_max=None, incremental=False),
      do_color_flow=a.color_flow,
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
