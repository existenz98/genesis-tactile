from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SourceMode(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    FOLDER = "folder"

class CompensationMode(str, Enum):
    SKIP = "skip"
    BASELINE = "baseline"
    PER_FRAME = "per_frame"

class UnmixMode(str, Enum):
    SKIP = "skip"
    KMEANS = "kmeans"

class FlowMethod(str, Enum):
    FARNEBACK = "farneback"
    DIS = "dis"          # requires opencv-contrib-python (cv2.optflow)
    TVL1 = "tvl1"        # requires opencv-contrib-python (cv2.optflow)

@dataclass
class CameraConfig:
    device_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None  # desired capture fps (may be ignored by driver)

@dataclass
class VirtualSourceConfig:
    fps: float = 30.0   # playback pacing for video/folder
    loop: bool = False  # loop when reach the end

@dataclass
class PreprocConfig:
    patch: int = 24
    sat_thr: float = 0.25
    val_thr: float = 0.86
    clip_min: float = 0.5
    clip_max: float = 2.0
    smooth_kernel: int = 3
    smooth_iters: int = 1

@dataclass
class UnmixConfig:
    mode: UnmixMode = UnmixMode.KMEANS
    kmeans_samples: int = 150_000
    kmeans_iters: int = 40
    reg: float = 1e-6

@dataclass
class FlowConfig:
    method: FlowMethod = FlowMethod.DIS
    # Farneback parameters are internal defaults; DIS/TVL1 use OpenCV presets
    vis_flow_max: Optional[float] = None  # None => auto by 95th percentile
    incremental: bool = False             # False: compare to first frame; True: accumulate incremental

@dataclass
class DisplayConfig:
    enable: bool = True
    window_scale: float = 0.7     # resize display windows
    wait_key_ms: int = 0          # cv2.waitKey delay per frame, 0 will block until key press.
    show_input: bool = True
    show_compensated: bool = True
    show_seg_color: bool = True   # color segmentation (argmax of components)
    show_seg_R: bool = False      # grayscale component maps
    show_seg_G: bool = False
    show_seg_B: bool = False
    show_flow_R: bool = True      # color-coded optical flow per component
    show_flow_G: bool = True
    show_flow_B: bool = True
    show_flow_raw: bool = False   # raw grayscale flow

@dataclass
class OutputConfig:
    write_videos: bool = True
    dir: str = "outputs"
    fps: float = 20.0             # writer fps (if not deriving from source)

@dataclass
class RuntimeConfig:
    # Source selection
    source_mode: SourceMode = SourceMode.VIDEO
    input_path: str = ""          # for VIDEO/FOLDER modes
    is_folder: bool = False       # deprecated; use source_mode
    camera: CameraConfig = field(default_factory=CameraConfig)
    virtual: VirtualSourceConfig = field(default_factory=VirtualSourceConfig)

    # Core modules
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    compensation_mode: CompensationMode = CompensationMode.BASELINE
    unmix: UnmixConfig = field(default_factory=UnmixConfig)
    unmix_mode: UnmixMode = UnmixMode.KMEANS
    flow: FlowConfig = field(default_factory=FlowConfig)

    # Options
    do_color_flow: bool = True
    do_raw_flow: bool = False

    # Output & display
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # Misc
    downscale: float = 1.0
    max_frames: Optional[int] = None   # stop after N frames if set
