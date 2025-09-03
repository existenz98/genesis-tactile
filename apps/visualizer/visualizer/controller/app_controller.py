# -----------------------------------------------------------------------------
# 3D Visualizer — Application Source
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Yue Fei <feiyuefy@gmail.com>
#
# License (Code):
#   This source file is MIT-licensed. See LICENSE-APPS-MIT.
#
# Brand & UI Assets (Not MIT-licensed):
#   Project name, logo(s), icons, UI themes/skins, and other brand assets are
#   NOT licensed under MIT. Use requires a separate license. See:
#     - LICENSE-APPS-ASSETS
#     - TRADEMARKS.md
#
# Patents:
#   Except as may be granted under the applicable open-source license, no patent
#   rights are granted or implied. Users are responsible for third-party patent
#   clearances.
#
# Notes:
#   - Redistribution must retain this header and the referenced license files.
#   - Forks may not imply endorsement or use the original branding without
#     permission (see TRADEMARKS.md).
# -----------------------------------------------------------------------------

from __future__ import annotations
import time
from ..utils.qt import QtCore
from tacto6d import Tacto6DClient, ClientConfig

class AppController(QtCore.QObject):
    frame_arrived = QtCore.Signal(dict)   # emits the latest frame dict

    def __init__(self, notify_ep: str, ctrl_ep: str, update_hz: float = 30.0):
        super().__init__()
        self.tacto6d = Tacto6DClient(ClientConfig(notify_ep=notify_ep, ctrl_ep=ctrl_ep))
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000.0 / max(1.0, update_hz)))
        self.timer.timeout.connect(self._tick)
        self._last_seq = None
        self._fps = 0.0
        self._fps_acc = 0
        self._fps_t0 = time.time()

    def start(self):
        self.tacto6d.start()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.tacto6d.stop()

    def _tick(self):
        # get latest frame
        fr = self.tacto6d.latest_frame(copy=True)
        if fr is None:
            return
        
        # TODO if frame is the same as last seq, should return
        
        # compute simple FPS by seq delta
        seq = fr.get("seq", 0)
        self._fps_acc += 1
        t = time.time()
        if t - self._fps_t0 >= 1.0:
            self._fps = self._fps_acc / (t - self._fps_t0)
            self._fps_acc = 0
            self._fps_t0 = t
        fr["_ui_fps"] = self._fps

        # send event to update ui
        self.frame_arrived.emit(fr)

    # Control API passthrough
    def set_algo(self, algo: int) -> bool:
        return self.tacto6d.set_algo(algo)

    def get_algo(self) -> int:
        return self.tacto6d.get_algo()
