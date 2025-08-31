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


"""
Local control RPC (IPC) using ZeroMQ REP:
Simple JSON RPC over ZeroMQ REP (IPC).

  - set_algo {algo: 1|2|3}
  - get_algo {}
  - get_info {}
  - endpoint default: ipc:///tmp/tacto6d.ctrl
  - Thread-safe via getter/setter callbacks.
"""

from __future__ import annotations
import json, threading
from dataclasses import dataclass
from typing import Callable, Optional
import zmq

from ..config.settings import CtrlConfig

class CtrlServer:
    def __init__(self, cfg: CtrlConfig,
                 get_algo: Callable[[], int],
                 set_algo: Callable[[int], None],
                 get_info: Callable[[], dict]):
        self.cfg = cfg
        self._get_algo = get_algo
        self._set_algo = set_algo
        self._get_info = get_info
        self._ctx = None
        self._sock = None
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.cfg.enable: return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(self.cfg.bind)
        self._stop.clear()
        self._thr = threading.Thread(target=self._serve, name="CtrlServer", daemon=True)
        self._thr.start()
        print(f"[ctrl] REP at {self.cfg.bind}")

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)
        try:
            if self._sock: self._sock.close(0)
        finally:
            self._sock = None

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                req = json.loads(msg.decode("utf-8"))
                cmd = req.get("cmd", "")
                if cmd == "set_algo":
                    algo = int(req["algo"])
                    self._set_algo(algo)
                    self._sock.send_string(json.dumps({"ok": True}))
                elif cmd == "get_algo":
                    self._sock.send_string(json.dumps({"algo": self._get_algo()}))
                elif cmd == "get_info":
                    self._sock.send_string(json.dumps(self._get_info()))
                else:
                    self._sock.send_string(json.dumps({"ok": False, "err": "unknown_cmd"}))
            except Exception as e:
                self._sock.send_string(json.dumps({"ok": False, "err": str(e)}))
