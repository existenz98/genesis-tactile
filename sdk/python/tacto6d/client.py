# -----------------------------------------------------------------------------
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025 Yue Fei <feiyuefy@gmail.com>
#
# This SDK is licensed under the GNU Lesser General Public License v3.0 or later.
# You may link this library with proprietary applications under the terms of the LGPL.
# Modifications to the SDK itself must be released under the same license when distributed.
# See LICENSE-SDK-LGPL for details.
#
# Patent Notice:
#   Except for rights granted under the LGPL, no patent license is granted or implied.
#   Users are responsible for third-party patent clearance.
#
# Citation:
#   Please cite associated publications when available.
# -----------------------------------------------------------------------------

"""
Tacto6D: SDK for synchronized frames + control, local computer.

- Subscribe to IPC notify (ZMQ) on "frame/ready"
- Attach to the SHM ring by name
- Map slots to numpy views (zero-copy) and expose latest_frame()
- Control RPC (IPC REQ): set_algo(1|2|3), get_algo()

This module is self-contained; no imports from the runtime tree.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading, json
import zmq
import numpy as np

from .frame_ring import FrameRingReader


@dataclass
class ClientConfig:
    notify_ep: str = "ipc:///tmp/tacto6d.frame"
    ctrl_ep:   str = "ipc:///tmp/tacto6d.ctrl"

# Note: Do NOT set CONFLATE on SUB when using multipart
# conflation + SUB + multipart isn’t reliable across platforms/builds.


class Tacto6DClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self._ctx = zmq.Context.instance()

        # notify SUB
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, 10)                # small queue
        # Do NOT set CONFLATE on SUB when using multipart
        self._sub.connect(self.cfg.notify_ep)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"frame/ready") # specify topic
        #self._sub.setsockopt(zmq.SUBSCRIBE, b"")           # receive everything


        # control REQ
        self._req = self._ctx.socket(zmq.REQ)
        self._req.connect(self.cfg.ctrl_ep)

        # SHM ring buffer
        self._ring: Optional[FrameRingReader] = None
        self._ring_name: Optional[str] = None

        # thread
        self._thr = threading.Thread(target=self._recv_loop, name="Tacto6DClient", daemon=True)
        self._stop = threading.Event()

        self._latest: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    # --- lifecycle ---
    def start(self) -> None:
        self._stop.clear()
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=1.0)
        try:
            if self._ring: self._ring.close()
        except: pass

    # --- data access ---
    def latest_frame(self, copy: bool = True) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            if not copy:
                return self._latest
            fr = dict(self._latest)
            for k in ("camera","vy","vx","p","tx","ty"):
                fr[k] = fr[k].copy()
            return fr

    # --- control API ---
    def set_algo(self, algo: int) -> bool:
        self._req.send_string(json.dumps({"cmd": "set_algo", "algo": int(algo)}))
        rep = json.loads(self._req.recv_string())
        return bool(rep.get("ok", False))

    def get_algo(self) -> int:
        self._req.send_string(json.dumps({"cmd": "get_algo"}))
        rep = json.loads(self._req.recv_string())
        return int(rep.get("algo", 0))

    # --- notify loop ---
    def _recv_loop(self) -> None:

        print("[Tacto6DClient] >>> _recv_loop()")

        # Use a poller and drain in each loop
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)

        while not self._stop.is_set():
            evts = dict(poller.poll(50))  # 50 ms
            if self._sub in evts:
                # Drain everything available and keep only the last
                last = None
                while True:
                    try:
                        parts = self._sub.recv_multipart(flags=zmq.NOBLOCK)
                        last = parts
                    except zmq.Again:
                        break
                if last is not None:
                    topic, header_b = last

                    try:
                        # get shm name
                        hdr = json.loads(header_b.decode("utf-8"))
                        name = hdr["name"]; slot = int(hdr["slot"]); seq = int(hdr["seq"]); t_usec = int(hdr["t_usec"])
                        if self._ring is None or self._ring_name != name:
                            # attach shm
                            self._ring = FrameRingReader.attach(name)
                            self._ring_name = name

                        # get frame from shm
                        views = self._ring.map_slot(slot)
                        frame = {**views, "seq": seq, "t_usec": t_usec,
                                "mm_per_px": self._ring.meta.mm_per_px,
                                "cell_mm": self._ring.meta.cell_mm}
                        
                        # store frame
                        with self._lock:
                            self._latest = frame

                    except Exception as e:
                        print("[Tacto6DClient] notify error:", e)
                        continue

        print("[Tacto6DClient] <<< _recv_loop()")
