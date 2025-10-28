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
from .tir.types import TirMeta, Deformation, to_flow2d_pixels
from .output.vis3d_pyvista import Vis3DLive
from .algorithms.physics_solver import PhysicsSolver
from .config.settings import PhysicsConfig
from .algorithms.cnn_model import CnnForceSolver

from runtime.utils.prof import Prof

# sdk related
from .server.shm_frame_ring import ShmFrameRing, ShmFrameConfig, ALGO_PHYSICS, ALGO_CNN, ALGO_IFEM
from .server.ipc_notify import IpcNotifier, IpcNotifyConfig
from .server.ctrl_server import CtrlServer, CtrlConfig




def _resize_bgr(img: np.ndarray, scale: float) -> np.ndarray:

    # no change
    if scale == 1.0:
        return img

    # 1/2 scale
    #if scale == 0.5:
    #    return cv2.pyrDown(img)

    # other scale
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
        """
        Initialize pipeline with configuration.
        """

        self.cfg = cfg

        # Algorithm, 1=hard coded Physics, 2=CNN, 3=iFEM
        self._algo_choice = 2

        # IPC bus for publishing frames to 3D viz / SDK
        self.bus: Optional[FrameBus] = None     # but's topics will be created later by runner

        # latest physics output (for sending to SDK / visualization)
        self.latest_physics = None

    # callable by CtrlServer
    def get_algo(self) -> int: return self._algo_choice
    def set_algo(self, a: int) -> None:
        a = int(a)
        if a in (1,2,3): self._algo_choice = a

    def _make_camera_source(self):
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

        # Config
        cfg = self.cfg

        # message out
        self.bus = bus

        # profiler
        prof = Prof(enable=True, report_every=20)

        # debug video writers
        writers = VideoWriters(cfg.output.dir, cfg.output.fps) if cfg.output.write_videos else None
        print(f"[pipeline] Video writers enabled: {cfg.output.write_videos}, writers={writers}")

        outputs: Dict[str, Any] = {}

        # debug 2D display
        disp = DebugDisplay(cfg.display)

        # ----- Create Cameras -----
        src = self._make_camera_source()

        # ----- Create SHM ring buffer -----
        # get shapes by warmup: read first frame, compute initial flow & force to determine SHM shapes
        first_bgr = next(src.frames())
        H_camera, W_camera = first_bgr.shape[:2]
        #vy, vx = self._compute_initial_flow(first_bgr)
        H_flow, W_flow = (int(H_camera*cfg.downscale), int(W_camera*cfg.downscale))
        #phys = self._compute_force(vy, vx)  # dict with grid/p/tau
        #H_force, W_force = int(phys["grid"]["H"]), int(phys["grid"]["W"])
        H_force, W_force = (15, 20)

        shm_cfg = ShmFrameConfig(
            cam_wh=(W_camera, H_camera),
            flow_wh=(W_flow, H_flow),
            force_wh=(W_force, H_force),
            mm_per_px=float(self.cfg.physics.mm_per_px),
            #cell_mm=float(phys["grid"]["cell_mm"]),    #TODO need to compute once and get the value.
        )
        ring = ShmFrameRing.create(shm_cfg)

        # ----- Notifier (IPC PUB) and Control server (IPC REP) -----
        notifier = IpcNotifier(IpcNotifyConfig(enable=self.cfg.notify.enable, bind=self.cfg.notify.bind))
        notifier.start()
        ctrl = CtrlServer(CtrlConfig(enable=self.cfg.control.enable, bind=self.cfg.control.bind),
                          get_algo=self.get_algo, set_algo=self.set_algo,
                          get_info=lambda: {"version":"0.1","shm":self.cfg.shm.name})
        ctrl.start()


        # First-frame references
        ref_bal_bgr: Optional[np.ndarray] = None
        ref_comp: Optional[np.ndarray] = None  # HxWx3 linear components (R/G/B)
        ref_raw_gray: Optional[np.ndarray] = None

        # ----- Modules -----

        # Create Sensor specific adapter
        print(f"[pipeline] Creating sensor adapter for type '{cfg.sensor.type}'")
        self.adapter = None
        if cfg.sensor.type in ("particles"):
            from .sensors.particles.adapter import ParticlesAdapter as Adapter
            self.adapter = Adapter(cfg, disp, writers)  # pass existing flags: unmix, compensation, flow, downscale
            print("[pipeline] Using ParticlesAdapter")
        elif cfg.sensor.type in ("particles_layered"):
            from .sensors.particles_layered.adapter import ParticlesLayeredAdapter as Adapter
            self.adapter = Adapter(cfg, disp, writers)  # pass existing flags: unmix, compensation, flow, downscale
            print("[pipeline] Using ParticlesLayeredAdapter")
        elif cfg.sensor.type in ("photometric", "gelsight", "gelsight_style"):
            from .sensors.photometric.adapter import PhotometricAdapter as Adapter
            self.adapter = Adapter(cfg, disp, writers)
            print("[pipeline] Using PhotometricAdapter")
        elif cfg.sensor.type in ("tac3d"):
            from .sensors.stereo_markers.adapter import Tac3DAdapter as Adapter
            self.adapter = Adapter(cfg, disp, writers)
            print("[pipeline] Using Tac3DAdapter")
        else:
            raise ValueError("Unknown sensor.type")


        # ----- Create Solvers -----

        # physics solver
        phys_cfg = self.cfg.physics
        physics_solver = PhysicsSolver(phys_cfg)

        # cnn solver
        cnn_solver = None
        cnn_cfg = self.cfg.cnn
        if self.cfg.cnn.enable:
            cnn_solver = CnnForceSolver(self.cfg.cnn)
            print(f"[pipeline] Initialized CNN solver: {cnn_solver}")

        # TODO: iFEM solver


        # ----- Main Loop -----
        count = 0
        src.start()
        try:
            for camera_frame in src.frames():
                #print("pipeline - got new frame")

                # 0) downsampling input
                with prof("0) resize"):
                    bgr = _resize_bgr(camera_frame, cfg.downscale)
                if cfg.display.show_input:
                    disp.show("input", bgr)

                # 1) Sensor addaptor computes Tactile Intermediate Representation (TIR)
                if count == 0:
                    self.adapter.prepare(bgr)

                deformation = self.adapter.process(bgr)
                if deformation is None:
                    print("[pipeline] Warning: deformation is None; skipping frame.")
                    continue


                # 2) Solver
                algo_id = self.get_algo()
                phys = None

                vy, vx = to_flow2d_pixels(deformation)

                if algo_id == 1:
                    with prof("2) solver - phys"):
                        phys = physics_solver.solve_from_dense(vy, vx)  # vy/vx are in pixels
                elif algo_id == 2:
                    with prof("2) solver - cnn"):
                        if self.cfg.cnn.enable and cnn_solver is not None:
                            phys = cnn_solver.solve_from_flow(vy, vx, mm_per_px=self.cfg.physics.mm_per_px)

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

                # 3) Output: write one Frame into SHM
                slot, views = ring.begin_frame(seq=count)               # 1. prepare to write
                # camera
                views["camera"][:] = camera_frame  # must be BGR8
                # flow (resize if flow size differs; omitted here assuming same as header)
                views["vy"][:] = vy
                views["vx"][:] = vx
                # force (assumed already at (Hp,Wp))
                views["p"][:]  = phys["p"]
                views["tx"][:] = phys["tau"]["tx"]
                views["ty"][:] = phys["tau"]["ty"]
                # commit & notify
                t_usec = int(time.time() * 1e6)
                ring.commit_frame(slot, algo=algo_id, t_usec=t_usec)    # 2. actual writing
                                                                        
                notifier.announce_ready(name=shm_cfg.name, slot=slot,   # 3. notify clients
                                        seq=count, t_usec=t_usec)


                # Kick Debug Display
                #print("pipeline - cv disp.tick")
                key = disp.tick()
                if key == ord('q'):
                    break

                count += 1
                if cfg.max_frames is not None and count >= cfg.max_frames:
                    break

                #print("pipeline - loop")
                prof.tick()

        finally:
            # SDK output related
            ctrl.stop()
            notifier.stop()
            ring.close()
            try: ring.unlink()   # commenting out to keep debugging...
            except: pass

            # algo pipeline related
            src.stop()
            disp.close()
            if writers is not None:
                writers.close()
                outputs.update(writers.paths())

        return outputs
