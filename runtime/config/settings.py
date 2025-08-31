# -----------------------------------------------------------------------------
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE file in the root directory of this source tree.
#
# Patent Notice:
#   This software is provided under copyright only.
#   No license to any patents is granted or implied.
#   Users are responsible for ensuring that their use of this software,
#   especially in commercial applications, does not infringe on any
#   third-party patents (e.g., tactile sensor hardware, methods).
#
# Citation:
#   If you use this code in academic work, please cite the associated
#   publication(s) when available.
# -----------------------------------------------------------------------------


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple


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
    quiver_draw_centers: bool = True        # draw center dots for reference
    quiver_color: tuple = (255, 255, 255)   # BGR
    quiver_bg: str = "black"                # "black" | "white"


@dataclass
class Vis3DConfig:
    enable: bool = True             # enable live 3D window
    topic: str = "physics"          # which bus topic to visualize: "physics" or "cnn"
    
    # surface (pressure)
    show_height: bool = True        # extrude surface z by w ≈ p / normal_gain
    height_gain: float = 1.0         # extra gain applied to w if show_height = True
    surface_opacity: float = 1.0
    colormap: str = "turbo"
    p_vmin: Optional[float] = None   # fixed pressure colorbar min; None = auto percentiles
    p_vmax: Optional[float] = None   # fixed pressure colorbar max; None = auto percentiles

    # traction arrows
    arrow_enable: bool = True
    arrow_stride: int = 1            # draw every Nth grid point
    arrow_min_len: float = 0.6       # min vector length to draw, in grid-cell units
    lift_z_mm: float = 0.2           # lift arrow bases above the surface to avoid z-fighting

    # axis scaling
    scale_t_auto: bool = True        # auto scale tangential components (xy)
    scale_n_auto: bool = True        # auto scale normal component (z)
    scale_t: float = 2.0             # used only if scale_t_auto is False
    scale_n: float = 2.0             # used only if scale_n_auto is False

    # update cadence
    update_ms: int = 33              # ~30 FPS update cadence



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
class OutputConfig:
    write_videos: bool = True
    dir: str = "outputs"
    fps: float = 20.0             # writer fps (if not deriving from source)



@dataclass
class PhysicsConfig:
    # Physical solver's / calibration parameters
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
class CnnConfig:
    enable: bool = True
    # Paths relative to project root (or absolute)
    model_cfg: str = "src/nn/train_unet.yaml"
    checkpoint: str = "outputs/ckpt/best.pth"
    device: str = "cuda:0"         # or "cpu"
    use_half: bool = False         # fp16 inference (CUDA only)
    input_is_flow: bool = True     # this model expects (vx, vy)
    # If None, use flip_y from nn config; otherwise override here.
    flip_y_override: Optional[bool] = None

    # Force a specific model input size; if None use nn cfg.dataset.resize_to
    #force_input_size: Optional[Tuple[int, int]] = None  # (W, H)
    force_input_size: Optional[Tuple[int, int]] = (80, 60)  # resize input to this res
    out_grid_wh: Optional[Tuple[int, int]] = (20, 15)       # resize output to this res


# ----- Server Config -----

@dataclass
class CtrlConfig:
    enable: bool = True
    bind: str = "ipc:///tmp/tacto6d.ctrl"

@dataclass
class IpcNotifyConfig:
    # local-only
    enable: bool = True
    bind: str = "ipc:///tmp/tacto6d.frame"
    topic: str = "frame/ready"

@dataclass
class ShmFrameConfig:
    name: str = "tacto6d_frame0"
    n_slots: int = 8
    cam_wh: Tuple[int, int] = (640, 480)      # (Wc, Hc)
    flow_wh: Tuple[int, int] = (640, 480)     # (Wf, Hf)
    force_wh: Tuple[int, int] = (20, 15)      # (Wp, Hp)
    # fixed formats
    cam_format: str = "BGR8"  # 3 channels
    flow_order: str = "[vy,vx]"
    force_order: str = "[p,tx,ty]"
    # physical scales (informational)
    mm_per_px: float = 0.05   # at flow resolution
    cell_mm: float = 2.0      # at force resolution
    schema: int = 1

    def to_header_dict(self) -> Dict[str, Any]:
        Wc, Hc = self.cam_wh
        Wf, Hf = self.flow_wh
        Wp, Hp = self.force_wh
        return {
            "schema": self.schema,
            "name": self.name,
            "n_slots": self.n_slots,
            "cam": {"W": Wc, "H": Hc, "format": self.cam_format, "channels": 3},
            "flow": {"W": Wf, "H": Hf, "dtype": "float32", "order": self.flow_order, "units": "px/frame"},
            "force": {"W": Wp, "H": Hp, "dtype": "float32", "order": self.force_order, "units": "MPa"},
            "scales": {"mm_per_px": self.mm_per_px, "cell_mm": self.cell_mm},
            "coords": { "x_right": True, "y_down": True, "z_into": True },
        }





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
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    physics_display: PhysicsDisplayConfig = field(default_factory=PhysicsDisplayConfig)

    # CNN solver
    cnn: CnnConfig = field(default_factory=CnnConfig)

    # 3D Visualization
    vis3d: Vis3DConfig = field(default_factory=Vis3DConfig)

    # Output & display
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # Server
    shm: ShmFrameConfig = field(default_factory=ShmFrameConfig)
    notify: IpcNotifyConfig = field(default_factory=IpcNotifyConfig)
    control: CtrlConfig = field(default_factory=CtrlConfig)

    # Misc
    downscale: float = 2.0
    max_frames: Optional[int] = None   # stop after N frames if set


    def __post_init__(self):
        # run after basic field initialized
        if isinstance(self.cnn, dict):
            self.cnn = CnnConfig(**self.cnn)
        if isinstance(self.vis3d, dict):
            self.vis3d = Vis3DConfig(**self.vis3d)
