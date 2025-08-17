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
    incremental: bool = False             # False: compare to first frame; True: accumulate incremental

    # Downsample, block-pooled
    ds_block: int = 16            # tile size (e.g., 8 / 16 / 24 / 32)
    ds_pool: str = "median"       # "median" (robust) or "mean"

    # Visualization
    vis_flow_max: Optional[float] = 20  # Max flow strength unit pixels.  None => auto by 95th percentile


@dataclass
class DisplayConfig:
    enable: bool = True
    window_scale: float = 0.7     # resize display windows
    wait_key_ms: int = 1          # cv2.waitKey delay per frame, 0 will block until key press.
    show_input: bool = True
    show_compensated: bool = True
    show_seg_color: bool = True   # color segmentation (argmax of components)
    show_seg_R: bool = False      # grayscale component maps
    show_seg_G: bool = False
    show_seg_B: bool = False

   
    # Color visualization of flow
    show_flow_color_R: bool = True
    show_flow_color_G: bool = True
    show_flow_color_B: bool = True
    show_flow_color_raw: bool = True

    # Quiver visualization of flow
    show_flow_quiver_R: bool = True
    show_flow_quiver_G: bool = True
    show_flow_quiver_B: bool = True
    show_flow_quiver_raw: bool = True

    # quiver rendering parameters
    quiver_block: int = 32
    quiver_pool: str = "median"
    quiver_scale: float = 2.0
    quiver_thickness: int = 1
    quiver_min_px: float = 0.6              # skip arrows shorter than this (in px)
    quiver_draw_centers: bool = True       # draw center dots for reference
    quiver_color: tuple = (255, 255, 255)   # BGR
    quiver_bg: str = "black"                # "black" | "white"

@dataclass
class OutputConfig:
    write_videos: bool = True
    dir: str = "outputs"
    fps: float = 20.0             # writer fps (if not deriving from source)


@dataclass
class PhysicsDisplayConfig:
    show_pressure_map: bool = True
    show_tau_quiver: bool = True
    # visualization tuning
    p_vis_min: float = 0.0           # min value for colormap (0 for non-negative pressure)
    p_vis_max: float = 0.0           # 0 => auto by 98th percentile
    tau_quiver_scale: float = 10.0      # arrow length multiplier (in grid units)
    tau_quiver_min: float = 0.6        # do not draw tiny shear arrows (< this magnitude, in same units as tau)
    tau_quiver_color: tuple = (255,255,255)
    tau_quiver_thickness: int = 1
    tau_quiver_bg: str = "black"

@dataclass
class PhysicsConfig:
    # Physical / calibration parameters
    thickness_mm: float = 3.0          # gel thickness h
    mm_per_px: float = 0.05            # image scale (set from calibration)
    normal_gain: float = 1.0           # k_n, maps w -> pressure: p = normal_gain * w
    shear_gain: float = 1.0            # b = kappa_s * G / h ; tau = shear_gain * u_hat + slope_gain * ∇w
    slope_gain: float = 1.0            # c_slope * ∇w term; set >0 to enable slope correction

    # Downsampling (from dense flow to coarse field for solving)
    ds_block: int = 32                 # tile size in pixels
    ds_pool: str = "mean"             # "median" | "mean" | "huber"
    huber_sigma_px: float = 1.0        # robust scale (pixels) for huber pooling

    # Pre-smoothing and derivative (work on the coarse grid)
    smooth_sigma_cells: float = 0.8    # Gaussian sigma in "grid cells" before taking derivatives
    min_flow_px: float = 0.2           # zero-out tiny flows (on dense field) before pooling

    # Visualization helpers (optional fixed ranges)
    vis_p_min: Optional[float] = 0.0  # pressure heatmap min (None => auto)
    vis_p_max: Optional[float] = 1.5  # pressure heatmap max (None => auto)



@dataclass
class RuntimeConfig:
    # Source selection
    source_mode: SourceMode = SourceMode.VIDEO
    input_path: str = ""          # for VIDEO/FOLDER modes
    is_folder: bool = False       # deprecated; use source_mode
    camera: CameraConfig = field(default_factory=CameraConfig)
    virtual: VirtualSourceConfig = field(default_factory=VirtualSourceConfig)

    # Options
    do_raw_flow: bool = True

    # Core modules
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    compensation_mode: CompensationMode = CompensationMode.BASELINE
    unmix: UnmixConfig = field(default_factory=UnmixConfig)
    unmix_mode: UnmixMode = UnmixMode.KMEANS
    flow: FlowConfig = field(default_factory=FlowConfig)

    # Physics solver
    physics: PhysicsConfig = PhysicsConfig()
    physics_display: PhysicsDisplayConfig = PhysicsDisplayConfig()

    # Output & display
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # Misc
    downscale: float = 2.0
    max_frames: Optional[int] = None   # stop after N frames if set
