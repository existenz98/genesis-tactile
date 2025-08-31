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
