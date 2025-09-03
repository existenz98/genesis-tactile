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
Real camera reader adapter.
Currently uses OpenCV for implementation.
"""


from typing import Iterator, Optional
import numpy as np

from .base import FrameSource

class CameraSource(FrameSource):
    """
    Real camera via OpenCV VideoCapture (BGR frames).
    """
    def __init__(self, device_index: int = 0, width: Optional[int]=None, height: Optional[int]=None, fps: Optional[float]=None):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.device_index)
        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frames(self) -> Iterator[np.ndarray]:
        import cv2
        if self.cap is None:
            self.start()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame  # BGR uint8
