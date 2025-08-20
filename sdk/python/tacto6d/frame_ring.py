"""
Read-only SHM Frame Ring for local computer zero-copy access to synchronized frames.

A Frame packs:
  camera: BGR8 uint8 [Hc, Wc, 3]
  flow  : float32 [2, Hf, Wf] order [vy, vx]
  force : float32 [3, Hp, Wp] order [p, tx, ty]

This reader relies ONLY on the header JSON in the first 4096 bytes of the SHM
segment to reconstruct shapes and offsets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from multiprocessing import shared_memory
import json, struct
import numpy as np

_HDR_BYTES = 4096
_SLOT_HDR_FMT = "<QQIIIIII"  # must match writer
_SLOT_HDR_SIZE = struct.calcsize(_SLOT_HDR_FMT)

@dataclass
class FrameRingMeta:
    name: str
    n_slots: int
    cam_wh: Tuple[int, int]    # (Wc, Hc)
    flow_wh: Tuple[int, int]   # (Wf, Hf)
    force_wh: Tuple[int, int]  # (Wp, Hp)
    cam_format: str = "BGR8"
    mm_per_px: float = 1.0
    cell_mm: float = 1.0
    schema: int = 1

class FrameRingReader:
    """Attach to a shared-memory Frame ring and map slots to numpy views (read-only)."""

    def __init__(self, shm: shared_memory.SharedMemory, meta: FrameRingMeta):
        self.shm = shm
        self.meta = meta
        self._buf = shm.buf
        # capacities & strides (must match writer)
        Wc, Hc = meta.cam_wh
        self._cap_cam = Wc * 3 * Hc

        Wf, Hf = meta.flow_wh
        self._cap_flow = 2 * Wf * Hf * 4  # 2 channels float32

        Wp, Hp = meta.force_wh
        self._cap_force = 3 * Wp * Hp * 4

        self._slot_bytes = self._align64(_SLOT_HDR_SIZE + self._cap_cam + self._cap_flow + self._cap_force)
        self._base_slots = _HDR_BYTES

    @classmethod
    def attach(cls, name: str) -> "FrameRingReader":
        shm = shared_memory.SharedMemory(name=name, create=False)
        # parse JSON header
        header_b = bytes(shm.buf[:_HDR_BYTES])
        j = header_b.split(b"\x00", 1)[0]
        meta = json.loads(j.decode("utf-8"))
        # build meta
        fr = FrameRingMeta(
            name=meta["name"],
            n_slots=int(meta["n_slots"]),
            cam_wh=(int(meta["cam"]["W"]),  int(meta["cam"]["H"])),
            flow_wh=(int(meta["flow"]["W"]), int(meta["flow"]["H"])),
            force_wh=(int(meta["force"]["W"]), int(meta["force"]["H"])),
            cam_format=meta["cam"]["format"],
            mm_per_px=float(meta["scales"]["mm_per_px"]),
            cell_mm=float(meta["scales"]["cell_mm"]),
            schema=int(meta.get("schema", 1)),
        )
        return cls(shm=shm, meta=fr)

    @staticmethod
    def _align64(n: int) -> int:
        return ((n + 63) // 64) * 64

    def _slot_base(self, slot: int) -> int:
        return self._base_slots + slot * self._slot_bytes

    def _payload_offsets(self, slot: int) -> Tuple[int, int, int]:
        base = self._slot_base(slot) + _SLOT_HDR_SIZE
        cam_off = base
        flow_off = cam_off + self._cap_cam
        force_off = flow_off + self._cap_flow
        return cam_off, flow_off, force_off

    def read_slot_header(self, slot: int) -> Dict[str, Any]:
        off = self._slot_base(slot)
        seq, t_usec, algo, status, n_cam, n_flow, n_force, _ = struct.unpack(
            _SLOT_HDR_FMT, self._buf[off:off+_SLOT_HDR_SIZE]
        )
        return {"seq": seq, "t_usec": t_usec, "algo": algo, "status": status,
                "n_cam": n_cam, "n_flow": n_flow, "n_force": n_force}

    def map_slot(self, slot: int) -> Dict[str, np.ndarray]:
        """Return numpy views for this slot (read-only; zero-copy)."""
        cam_off, flow_off, force_off = self._payload_offsets(slot)

        # camera
        Wc, Hc = self.meta.cam_wh
        cam_mv = self._buf[cam_off: cam_off + self._cap_cam]
        cam = np.frombuffer(cam_mv, dtype=np.uint8, count=self._cap_cam).reshape(Hc, Wc, 3)

        # flow
        Wf, Hf = self.meta.flow_wh
        flow_mv = self._buf[flow_off: flow_off + self._cap_flow]
        flow = np.frombuffer(flow_mv, dtype=np.float32, count=2*Wf*Hf).reshape(2, Hf, Wf)
        vy = flow[0]; vx = flow[1]

        # force
        Wp, Hp = self.meta.force_wh
        force_mv = self._buf[force_off: force_off + self._cap_force]
        force = np.frombuffer(force_mv, dtype=np.float32, count=3*Wp*Hp).reshape(3, Hp, Wp)
        p = force[0]; tx = force[1]; ty = force[2]

        return {"camera": cam, "vy": vy, "vx": vx, "p": p, "tx": tx, "ty": ty}

    def close(self) -> None:
        try: self.shm.close()
        except: pass
