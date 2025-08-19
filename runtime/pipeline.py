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
from .output.visualizer import flow_to_color_bgr, draw_quiver_bgr, flow_block_reduce, scalar_to_color_bgr, draw_quiver_grid_bgr, VideoWriters
from .output.display import DebugDisplay
from .core.frame_bus import FrameBus
from .output.vis3d_pyvista import Vis3DLive
from .algorithms.physics_solver import PhysicsSolver
from .config.settings import PhysicsConfig
from .algorithms.cnn_model import CnnForceSolver


def _resize_bgr(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0: return img
    h,w = img.shape[:2]
    nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw,nh), interpolation=cv2.INTER_AREA)

class RuntimePipeline:
    """
    Streaming pipeline:
      source -> (compensation) -> (unmix) -> flows -> windows/videos + algorithm handoff.
    Exposes latest dense and downsampled flows via `self.latest_flows` for the next-stage solvers.
    """

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.latest_flows: Dict[str, Any] = {}  # updated every frame
        self.bus: Optional[FrameBus] = None     # but's topics will be created later by runner
        self.latest_physics = None

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

    def run(self, bus: Optional[FrameBus] = None) -> Dict[str, Any]:
        cfg = self.cfg
        self.bus = bus

        # camera source
        src = self._make_source()

        # debug 2D display
        disp = DebugDisplay(cfg.display)

        writers = VideoWriters(cfg.output.dir, cfg.output.fps) if cfg.output.write_videos else None
        outputs: Dict[str, Any] = {}

        # First-frame references
        ref_bal_bgr: Optional[np.ndarray] = None
        ref_comp: Optional[np.ndarray] = None  # HxWx3 linear components (R/G/B)
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

        # Physics Solver
        phys_cfg = self.cfg.physics
        physics_solver = PhysicsSolver(phys_cfg)

        # CNN Solver
        cnn_solver = None
        cnn_cfg = self.cfg.cnn
        if self.cfg.cnn.enable:
            cnn_solver = CnnForceSolver(self.cfg.cnn)        

        # Stream
        count = 0
        src.start()
        try:
            for bgr in src.frames():
                #print("pipeline - got new frame")

                bgr = _resize_bgr(bgr, cfg.downscale)
                if cfg.display.show_input:
                    disp.show("input", bgr)

                # Compensation
                #print("pipeline - compensation")
                if isinstance(compensator, BaselineCompensator) and ref_bal_bgr is None:
                    # fit on first frame
                    compensator.fit(bgr)
                if compensator is not None:
                    bal_bgr = compensator.apply(bgr)
                    if cfg.display.show_compensated:
                        disp.show("compensated", bal_bgr)
                else:
                    bal_bgr = bgr.copy()

                # Initialize refs from the first balanced frame
                if ref_bal_bgr is None:
                    ref_bal_bgr = bal_bgr.copy()
                    if cfg.unmix_mode != UnmixMode.SKIP and unmix is not None:
                        unmix.fit(ref_bal_bgr)
                        ref_comp = unmix.transform(ref_bal_bgr)     # linear components
                    ref_raw_gray = to_gray_f32_bgr(ref_bal_bgr if cfg.unmix_mode != UnmixMode.SKIP else bgr)

                # Per-frame containers for output to solver
                dense: Dict[str, Any] = {}
                down: Dict[str, Any] = {}

                # Color unmix per frame, and per-color flows
                if cfg.unmix_mode != UnmixMode.SKIP and unmix is not None:
                    #print("pipeline - color unmix")
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
                    for k, name in enumerate(["R","G","B"]):
                        refg = to_gray_f32_linear_component(ref_comp[...,k])
                        curg = to_gray_f32_linear_component(comp_lin[...,k])
                        vy, vx = compute_flow(refg, curg, cfg.flow)
                        dense[name] = (vy, vx)

                        # COLOR window + (optional) writing
                        if getattr(cfg.display, f"show_flow_color_{name}"):
                            color_bgr = flow_to_color_bgr(vy, vx, cfg.flow.vis_flow_max)
                            disp.show(f"flow_color_{name}", color_bgr)
                            if writers is not None:
                                writers.write(f"flow_color_{name}", color_bgr)

                        # QUIVER window
                        if getattr(cfg.display, f"show_flow_quiver_{name}"):
                            quiv_bgr = draw_quiver_bgr(
                                vy, vx,
                                block=cfg.display.quiver_block,
                                pool=cfg.display.quiver_pool,
                                scale=cfg.display.quiver_scale,
                                thickness=cfg.display.quiver_thickness,
                                color=cfg.display.quiver_color,
                                bg=cfg.display.quiver_bg,
                                min_px=cfg.display.quiver_min_px,
                                draw_centers=cfg.display.quiver_draw_centers,
                                center_color=cfg.display.quiver_color,
                            )
                            disp.show(f"flow_quiver_{name}", quiv_bgr)

                        # downsampled arrays for output (to algorithms)
                        vy_ds, vx_ds, _ = flow_block_reduce(vy, vx, block=cfg.flow.ds_block, pool=cfg.flow.ds_pool)
                        down[name] = (vy_ds, vx_ds)

                # Raw flow vs reference
                if cfg.do_raw_flow:
                    #print("pipeline - compute flow")
                    cur_gray = to_gray_f32_bgr(bal_bgr if unmix is not None else bgr)
                    vy, vx = compute_flow(ref_raw_gray, cur_gray, cfg.flow)
                    dense["raw"] = (vy, vx)

                    #print("pipeline - show flow")
                    if cfg.display.show_flow_color_raw:
                        color_bgr = flow_to_color_bgr(vy, vx, cfg.flow.vis_flow_max)
                        disp.show("flow_color_raw", color_bgr)
                        if writers is not None:
                            writers.write("flow_color_raw", color_bgr)

                    if cfg.display.show_flow_quiver_raw:
                        quiv_bgr = draw_quiver_bgr(
                            vy, vx,
                            block=cfg.display.quiver_block,
                            pool=cfg.display.quiver_pool,
                            scale=cfg.display.quiver_scale,
                            thickness=cfg.display.quiver_thickness,
                            color=cfg.display.quiver_color,
                            bg=cfg.display.quiver_bg,
                            min_px=cfg.display.quiver_min_px,
                            draw_centers=cfg.display.quiver_draw_centers,
                            center_color=cfg.display.quiver_color,
                        )
                        disp.show("flow_quiver_raw", quiv_bgr)

                    # Solver
                    phys = None
                    #phys = physics_solver.solve_from_dense(vy, vx)  # vy/vx are in pixels
                    #print(f"physics solver result = {phys['p'].shape}")

                    if self.cfg.cnn.enable and cnn_solver is not None:
                        phys = cnn_solver.solve_from_flow(vy, vx, mm_per_px=self.cfg.physics.mm_per_px)
                        #print(f"cnn solver result = {phys['p'].shape}")

                    self.latest_physics = phys  # expose to downstream (SDK / Visualization)

                    # 2D Visualization
                    #print("pipeline - show 2d view")
                    pd = self.cfg.physics_display
                    Hf, Wf = vy.shape

                    if pd.show_pressure_map:
                        p = phys["p"]
                        p_bgr = scalar_to_color_bgr(
                            p,
                            vmin=(physics_solver.cfg.vis_p_min if physics_solver.cfg.vis_p_min is not None else None),
                            vmax=(physics_solver.cfg.vis_p_max if physics_solver.cfg.vis_p_max is not None else None),
                        )
                        # Resize to full frame for easy comparison
                        p_bgr = cv2.resize(p_bgr, (Wf, Hf), interpolation=cv2.INTER_CUBIC)
                        disp.show("physics_pressure", p_bgr)

                    if pd.show_tau_quiver:
                        tx = phys["tau"]["tx"]; ty = phys["tau"]["ty"]
                        cell_px = phys["grid"]["cell_px"]
                        quiv_bgr = draw_quiver_grid_bgr(
                            vy_grid=ty, vx_grid=tx,        # vy = ty (down), vx = tx (right)
                            cell_px=cell_px,
                            out_H=Hf, out_W=Wf,
                            scale=pd.tau_quiver_scale,
                            thickness=pd.tau_quiver_thickness,
                            color=pd.tau_quiver_color,
                            bg=pd.tau_quiver_bg,
                            min_len=pd.tau_quiver_min,
                        )
                        disp.show("physics_shear", quiv_bgr)

                    # Publish message   (receiver is 3D visualization thread)
                    #print("publish >>>")
                    if self.bus is not None:
                        self.bus.publish("physics", phys)
                    #print("publish <<<")

                    vy_ds, vx_ds, _ = flow_block_reduce(vy, vx, block=cfg.flow.ds_block, pool=cfg.flow.ds_pool)
                    down["raw"] = (vy_ds, vx_ds)


                # Output: make latest flows available to solvers (iFEM/physics/CNN)
                #print("pipeline - self.latest_flows")
                self.latest_flows = {
                    "frame_index": count,
                    "dense": dense,         # dict: { 'R': (vy, vx), 'G':..., 'B':..., 'raw':(...) }
                    "downsampled": down,    # dict: same keys but pooled arrays
                }

                #print("pipeline - cv disp.tick")
                key = disp.tick()
                if key == ord('q'):
                    break

                count += 1
                if cfg.max_frames is not None and count >= cfg.max_frames:
                    break

                #print("pipeline - loop")

        finally:
            src.stop()
            disp.close()
            if writers is not None:
                writers.close()
                outputs.update(writers.paths())

        return outputs
