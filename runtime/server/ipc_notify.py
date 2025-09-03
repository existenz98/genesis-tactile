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
IPC (using ZeroMQ) notify publisher:
Publish only small JSON “frame ready” messages.

  - topic: b"frame/ready"
  - payload: header JSON (shm name, slot, seq, t_usec)
  - endpoint default: ipc:///tmp/tacto6d.frame
"""

from __future__ import annotations
import json, threading
from dataclasses import dataclass
from typing import Optional
import zmq

from ..config.settings import IpcNotifyConfig

class IpcNotifier:
    def __init__(self, cfg: IpcNotifyConfig):
        self.cfg = cfg
        self._ctx = None
        self._sock = None
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.cfg.enable: return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.sndhwm = 10
        self._sock.bind(self.cfg.bind)
        print(f"[ipc_notify] PUB at {self.cfg.bind}, topic={self.cfg.topic}")

    def stop(self) -> None:
        try:
            if self._sock: self._sock.close(0)
        finally:
            self._sock = None

    def announce_ready(self, name: str, slot: int, seq: int, t_usec: int) -> None:
        if not self._sock:
            print("[IpcNotifier] announce_ready() error: sock is None")
            return

        header = {"name": name, "slot": int(slot), "seq": int(seq), "t_usec": int(t_usec)}
        self._sock.send_multipart([self.cfg.topic.encode("ascii"),
                                   json.dumps(header).encode("utf-8")], zmq.DONTWAIT)
        print(f"[IpcNotifier] announce_ready() send_multipart() topic={self.cfg.topic}, header={header}")
