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
