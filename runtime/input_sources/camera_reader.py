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
