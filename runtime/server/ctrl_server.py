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
