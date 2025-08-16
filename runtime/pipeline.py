import os, time
from typing import Dict, Any, Iterator, Optional, List
import numpy as np
import cv2

from .config.settings import (
    RuntimeConfig, SourceMode, CompensationMode, UnmixMode
)
from .input_sources.camera_reader import CameraSource
from .input_sources.video_reader import VideoFileSource, FolderSource
from .preprocessing.photo_compensator import BaselineCompensator, PerFrameCompensator, bgr_to_rgb
from .preprocessing.feature_extraction import UnmixModel
from .preprocessing.optical_flow import compute_flow, to_gray_f32_bgr, to_gray_f32_linear_component
from .output.visualizer import flow_to_color_bgr, VideoWriters
from .output.display import DebugDisplay


def _resize_bgr(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0: return img
    h,w = img.shape[:2]
    nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw,nh), interpolation=cv2.INTER_AREA)

class RuntimePipeline:
    """Streaming pipeline: source -> (optional) compensation -> (optional) unmix -> flows -> outputs & display."""
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg

    def _make_source(self):
        sm = self.cfg.source_mode
        if sm == SourceMode.CAMERA:
            c = self.cfg.camera
            return CameraSource(c.device_index, c.width, c.height, c.fps)
        elif sm == SourceMode.VIDEO:
            v = self.cfg.virtual
            return VideoFileSource(self.cfg.input_path, fps=v.fps, loop=v.loop)
        elif sm == SourceMode.FOLDER:
            v = self.cfg.virtual
            return FolderSource(self.cfg.input_path, fps=v.fps, loop=v.loop)
        else:
            raise ValueError(f"Unknown source mode: {sm}")

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        src = self._make_source()
        disp = DebugDisplay(cfg.display)
        key = disp.tick()

        writers = VideoWriters(cfg.output.dir, cfg.output.fps) if cfg.output.write_videos else None
        outputs = {}

        # State for first frame references
        ref_bal_bgr: Optional[np.ndarray] = None
        ref_comp: Optional[np.ndarray] = None  # HxWx3 linear components
        ref_raw_gray: Optional[np.ndarray] = None

        # Modules
        compensator = None
        if cfg.compensation_mode == CompensationMode.BASELINE:
            compensator = BaselineCompensator(cfg.preproc)
        elif cfg.compensation_mode == CompensationMode.PER_FRAME:
            compensator = PerFrameCompensator(cfg.preproc)

        unmix = None
        if cfg.unmix_mode != UnmixMode.SKIP:
            unmix = UnmixModel(cfg.unmix)

        # Stream
        count = 0
        src.start()
        try:
            for bgr in src.frames():
                bgr = _resize_bgr(bgr, cfg.downscale)
                if cfg.display.show_input:
                    disp.show("input", bgr)
                    key = disp.tick()

                # Compensation
                if isinstance(compensator, BaselineCompensator) and ref_bal_bgr is None:
                    # fit on first frame
                    compensator.fit(bgr)
                if compensator is not None:
                    bal_bgr = compensator.apply(bgr)
                else:
                    bal_bgr = bgr.copy()

                if cfg.display.show_compensated:
                    disp.show("compensated", bal_bgr)

                # Initialize references on first frame
                if ref_bal_bgr is None:
                    ref_bal_bgr = bal_bgr.copy()
                    if cfg.unmix_mode != UnmixMode.SKIP and unmix is not None:
                        unmix.fit(ref_bal_bgr)
                        ref_comp = unmix.transform(ref_bal_bgr)
                    ref_raw_gray = to_gray_f32_bgr(ref_bal_bgr if cfg.unmix_mode != UnmixMode.SKIP else bgr)

                # Unmix per frame if enabled
                if cfg.unmix_mode != UnmixMode.SKIP and unmix is not None:
                    comp_lin = unmix.transform(bal_bgr)  # HxWx3 linear
                    # Visualization: segmentation color & component maps
                    if cfg.display.show_seg_color or cfg.display.show_seg_R or cfg.display.show_seg_G or cfg.display.show_seg_B:
                        seg_idx = np.argmax(comp_lin, axis=-1)
                        seg_bgr = np.zeros_like(bal_bgr)
                        # R,G,B colors in BGR
                        palette = np.array([[0,0,255],[0,255,0],[255,0,0]], dtype=np.uint8)
                        for k in range(3):
                            seg_bgr[seg_idx==k] = palette[k]
                        if cfg.display.show_seg_color:
                            disp.show("seg_rgb", seg_bgr)
                        def to_vis(comp):
                            g = to_gray_f32_linear_component(comp)
                            return (np.stack([g,g,g], axis=-1)*255).astype(np.uint8)
                        if cfg.display.show_seg_R: disp.show("comp_R", to_vis(comp_lin[...,0]))
                        if cfg.display.show_seg_G: disp.show("comp_G", to_vis(comp_lin[...,1]))
                        if cfg.display.show_seg_B: disp.show("comp_B", to_vis(comp_lin[...,2]))

                    # Per-color flows vs reference
                    if cfg.do_color_flow and ref_comp is not None:
                        for k, name in enumerate(["R","G","B"]):
                            refg = to_gray_f32_linear_component(ref_comp[...,k])
                            curg = to_gray_f32_linear_component(comp_lin[...,k])
                            vy, vx = compute_flow(refg, curg, cfg.flow)  # current -> reference
                            flow_bgr = flow_to_color_bgr(vy, vx, cfg.flow.vis_flow_max)
                            win = f"flow_{name}"
                            if getattr(cfg.display, f"show_flow_{name}"):
                                disp.show(win, flow_bgr)
                            if writers is not None:
                                writers.write(win, flow_bgr)

                # Raw flow vs reference
                if cfg.do_raw_flow:
                    cur_gray = to_gray_f32_bgr(bal_bgr if cfg.unmix_mode != UnmixMode.SKIP else bgr)
                    vy, vx = compute_flow(ref_raw_gray, cur_gray, cfg.flow)
                    flow_bgr = flow_to_color_bgr(vy, vx, cfg.flow.vis_flow_max)
                    if cfg.display.show_flow_raw:
                        disp.show("flow_raw", flow_bgr)
                    if writers is not None:
                        writers.write("flow_raw", flow_bgr)

                key = disp.tick()
                if key == ord('q'):
                    break

                count += 1
                if cfg.max_frames is not None and count >= cfg.max_frames:
                    break
        finally:
            src.stop()
            disp.close()
            if writers is not None:
                writers.close()
                outputs.update(writers.paths())

        return outputs
