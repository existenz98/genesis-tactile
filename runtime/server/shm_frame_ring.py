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
Local, zero-copy "Frame" ring buffer using Python's shared_memory.

A Frame packs, atomically:
  - camera: BGR8 uint8 [Hc, Wc, 3]
  - flow  : float32 [2, Hf, Wf] order [vy, vx]  (units: px/frame)
  - force : float32 [3, Hp, Wp] order [p, tx, ty] (units: MPa)

Layout (single shared segment):
  [ GlobalHeader (JSON, padded to 4096) ]
  [ Slot 0: SlotHeader (binary, 64B-aligned) | cam_payload | flow_payload | force_payload ]
  [ Slot 1: ... ]
  ...
  ...

Header defines capacities and shapes. Writer fills a slot, flips status READY, then notifies (via ZMQ IPC).
Readers mmap once and index directly to view numpy arrays without per-frame allocations.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from multiprocessing import shared_memory
import json, struct, time
import numpy as np

# ---- constants ----
_HDR_BYTES = 4096  # Global header region, JSON text padded with zeros.
_SLOT_HDR_FMT = "<QQIIIIII"  # seq:u64, t_usec:u64, algo:u32, status:u32, n_cam:u32, n_flow:u32, n_force:u32, reserved:u32
_SLOT_HDR_SIZE = struct.calcsize(_SLOT_HDR_FMT)

STATUS_FREE = 0
STATUS_WRITING = 1
STATUS_READY = 2

ALGO_PHYSICS = 1
ALGO_CNN     = 2
ALGO_IFEM    = 3


from ..config.settings import ShmFrameConfig

class ShmFrameRing:
    """
    Owner/Writer API (runtime):
      ring = ShmFrameRing.create(cfg)  # allocates
      slot = ring.begin_frame()        # returns slot index and writable views
      ring.commit_frame(slot, algo, t_usec, cam_used, flow_used, force_used)  # atomically marks READY

    Reader API (SDK):
      ring = ShmFrameRing.attach(name)  # opens existing
      views = ring.map_slot(slot)       # returns numpy views for camera/flow/force (read-only)
    """
    def __init__(self, shm: shared_memory.SharedMemory, cfg: ShmFrameConfig, create: bool):
        self.shm = shm
        self.cfg = cfg
        self._buf = shm.buf  # memoryview
        self._n = cfg.n_slots
        # precompute capacities and offsets
        Wc, Hc = cfg.cam_wh
        self._cam_stride = Wc * 3  # BGR8 tight
        self._cap_cam = self._cam_stride * Hc

        Wf, Hf = cfg.flow_wh
        self._cap_flow = 2 * Wf * Hf * 4  # 2 channels float32

        Wp, Hp = cfg.force_wh
        self._cap_force = 3 * Wp * Hp * 4

        self._slot_bytes = self._align64(_SLOT_HDR_SIZE + self._cap_cam + self._cap_flow + self._cap_force)
        self._base_slots = _HDR_BYTES

        if create:
            self._write_header_json()
            # zero slot headers
            for s in range(self._n):
                self._write_slot_hdr(s, seq=0, t_usec=0, algo=0, status=STATUS_FREE,
                                     n_cam=0, n_flow=0, n_force=0)

    # ---- creation/attach ----

    @classmethod
    def create(cls, cfg: ShmFrameConfig) -> "ShmFrameRing":
        size = _HDR_BYTES + cfg.n_slots * cls._est_slot_bytes(cfg)
        shm = shared_memory.SharedMemory(create=True, size=size, name=cfg.name)
        return cls(shm=shm, cfg=cfg, create=True)

    @classmethod
    def attach(cls, name: str) -> "ShmFrameRing":
        shm = shared_memory.SharedMemory(name=name, create=False)
        # read JSON header
        hdr_bytes = bytes(shm.buf[:_HDR_BYTES])
        j = hdr_bytes.split(b"\x00", 1)[0]
        meta = json.loads(j.decode("utf-8"))
        cfg = ShmFrameConfig(
            name=meta["name"],
            n_slots=int(meta["n_slots"]),
            cam_wh=(int(meta["cam"]["W"]), int(meta["cam"]["H"])),
            flow_wh=(int(meta["flow"]["W"]), int(meta["flow"]["H"])),
            force_wh=(int(meta["force"]["W"]), int(meta["force"]["H"])),
            cam_format=meta["cam"]["format"],
            mm_per_px=float(meta["scales"]["mm_per_px"]),
            cell_mm=float(meta["scales"]["cell_mm"]),
            schema=int(meta.get("schema", 1)),
        )
        return cls(shm=shm, cfg=cfg, create=False)

    @staticmethod
    def _est_slot_bytes(cfg: ShmFrameConfig) -> int:
        Wc, Hc = cfg.cam_wh
        cap_cam = Wc * 3 * Hc
        Wf, Hf = cfg.flow_wh
        cap_flow = 2 * Wf * Hf * 4
        Wp, Hp = cfg.force_wh
        cap_force = 3 * Wp * Hp * 4
        raw = _SLOT_HDR_SIZE + cap_cam + cap_flow + cap_force
        return ((raw + 63) // 64) * 64

    @staticmethod
    def _align64(n: int) -> int:
        return ((n + 63) // 64) * 64

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except Exception:
            pass

    # ---- internal helpers ----

    def _write_header_json(self) -> None:
        meta = self.cfg.to_header_dict()
        jb = json.dumps(meta).encode("utf-8")
        if len(jb) > _HDR_BYTES:
            raise RuntimeError("Header JSON too large")
        self._buf[:len(jb)] = jb
        # zero the remainder
        self._buf[len(jb):_HDR_BYTES] = b"\x00" * (_HDR_BYTES - len(jb))

    def _slot_base(self, slot: int) -> int:
        return self._base_slots + slot * self._slot_bytes

    def _write_slot_hdr(self, slot: int, seq: int, t_usec: int, algo: int, status: int,
                        n_cam: int, n_flow: int, n_force: int) -> None:
        off = self._slot_base(slot)
        hdr = struct.pack(_SLOT_HDR_FMT, int(seq), int(t_usec), int(algo), int(status),
                          int(n_cam), int(n_flow), int(n_force), 0)
        self._buf[off:off+_SLOT_HDR_SIZE] = hdr

    def _read_slot_hdr(self, slot: int) -> Tuple[int, int, int, int, int, int, int]:
        off = self._slot_base(slot)
        seq, t_usec, algo, status, n_cam, n_flow, n_force, _ = struct.unpack(
            _SLOT_HDR_FMT, self._buf[off:off+_SLOT_HDR_SIZE])
        return seq, t_usec, algo, status, n_cam, n_flow, n_force

    def _payload_offsets(self, slot: int) -> Tuple[int, int, int]:
        base = self._slot_base(slot) + _SLOT_HDR_SIZE
        cam_off = base
        flow_off = cam_off + self._cap_cam
        force_off = flow_off + self._cap_flow
        return cam_off, flow_off, force_off

    # ---- writer API ----

    def begin_frame(self, seq: int, slot_hint: Optional[int] = None) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Reserve a slot (round-robin or slot_hint) and return writable numpy views:
          views["camera"]: uint8 [Hc,Wc,3]
          views["vy"], views["vx"]: float32 [Hf,Wf]
          views["p"],  views["tx"], views["ty"]: float32 [Hp,Wp]
        Caller must fill them, then call commit_frame().
        """
        # simple RR: slot = seq % n_slots unless slot_hint provided
        slot = int(seq % self._n) if slot_hint is None else int(slot_hint) % self._n
        # mark WRITING
        self._write_slot_hdr(slot, seq=seq, t_usec=0, algo=0, status=STATUS_WRITING,
                             n_cam=0, n_flow=0, n_force=0)
        cam_off, flow_off, force_off = self._payload_offsets(slot)

        # camera view
        Wc, Hc = self.cfg.cam_wh
        cam_mv = self._buf[cam_off: cam_off + self._cap_cam]
        cam = np.frombuffer(cam_mv, dtype=np.uint8, count=self._cap_cam)
        cam = cam.reshape(Hc, Wc, 3)

        # flow views
        Wf, Hf = self.cfg.flow_wh
        flow_mv = self._buf[flow_off: flow_off + self._cap_flow]
        flow = np.frombuffer(flow_mv, dtype=np.float32, count=2*Wf*Hf).reshape(2, Hf, Wf)
        vy = flow[0]; vx = flow[1]

        # force views
        Wp, Hp = self.cfg.force_wh
        force_mv = self._buf[force_off: force_off + self._cap_force]
        force = np.frombuffer(force_mv, dtype=np.float32, count=3*Wp*Hp).reshape(3, Hp, Wp)
        p = force[0]; tx = force[1]; ty = force[2]

        return slot, {"camera": cam, "vy": vy, "vx": vx, "p": p, "tx": tx, "ty": ty}

    def commit_frame(self, slot: int, algo: int, t_usec: Optional[int],
                     cam_bytes: Optional[int] = None,
                     flow_bytes: Optional[int] = None,
                     force_bytes: Optional[int] = None) -> None:
        """
        Atomically marks the slot READY. Byte sizes are optional (for diagnostics).
        """
        if t_usec is None:
            t_usec = int(time.time() * 1e6)
        # sizes default to caps
        if cam_bytes is None:
            cam_bytes = self._cap_cam
        if flow_bytes is None:
            flow_bytes = self._cap_flow
        if force_bytes is None:
            force_bytes = self._cap_force
        seq, _, _, _, _, _, _ = self._read_slot_hdr(slot)
        # memory barrier is implicit after Python writes; we just flip status to READY
        self._write_slot_hdr(slot, seq=seq, t_usec=t_usec, algo=int(algo), status=STATUS_READY,
                             n_cam=int(cam_bytes), n_flow=int(flow_bytes), n_force=int(force_bytes))

    # ---- reader helpers (SDK side) ----

    def map_slot(self, slot: int) -> Dict[str, np.ndarray]:
        """Return *read-only* views for a given slot."""
        cam_off, flow_off, force_off = self._payload_offsets(slot)
        Wc, Hc = self.cfg.cam_wh
        cam_mv = self._buf[cam_off: cam_off + self._cap_cam]
        cam = np.frombuffer(cam_mv, dtype=np.uint8, count=self._cap_cam).reshape(Hc, Wc, 3)

        Wf, Hf = self.cfg.flow_wh
        flow_mv = self._buf[flow_off: flow_off + self._cap_flow]
        flow = np.frombuffer(flow_mv, dtype=np.float32, count=2*Wf*Hf).reshape(2, Hf, Wf)
        vy = flow[0]; vx = flow[1]

        Wp, Hp = self.cfg.force_wh
        force_mv = self._buf[force_off: force_off + self._cap_force]
        force = np.frombuffer(force_mv, dtype=np.float32, count=3*Wp*Hp).reshape(3, Hp, Wp)
        p = force[0]; tx = force[1]; ty = force[2]
        return {"camera": cam, "vy": vy, "vx": vx, "p": p, "tx": tx, "ty": ty}

    def read_slot_header(self, slot: int) -> Dict[str, Any]:
        seq, t_usec, algo, status, n_cam, n_flow, n_force = self._read_slot_hdr(slot)
        return {"seq": seq, "t_usec": t_usec, "algo": algo, "status": status,
                "n_cam": n_cam, "n_flow": n_flow, "n_force": n_force}
