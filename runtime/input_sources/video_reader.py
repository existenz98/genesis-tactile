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
Video reader adapter.
Supports both video file, or folder with frames as individual files.
"""


import time, os, glob
from typing import Iterator
import numpy as np
import cv2

from .base import FrameSource

class VideoFileSource(FrameSource):
    """Video file streaming with pacing (BGR frames)."""
    def __init__(self, path: str, fps: float = 30.0, loop: bool = False):
        self.path = path
        self.fps = fps
        self.loop = loop
        self.cap = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.path)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frames(self) -> Iterator[np.ndarray]:
        if self.cap is None:
            self.start()
        delay = 1.0 / max(self.fps, 1e-6)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            t0 = time.time()
            yield frame
            # pace
            t_used = time.time() - t0
            sleep = delay - t_used
            if sleep > 0:
                time.sleep(sleep)

class FolderSource(FrameSource):
    """Folder of images streaming with pacing (BGR frames)."""
    def __init__(self, folder: str, fps: float = 30.0, loop: bool = False):
        self.folder = folder
        self.fps = fps
        self.loop = loop
        self.files = []

    def start(self) -> None:
        exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.folder, e)))
        self.files = sorted(files)

    def stop(self) -> None:
        pass

    def frames(self) -> Iterator[np.ndarray]:
        if not self.files:
            self.start()
        delay = 1.0 / max(self.fps, 1e-6)
        idx = 0
        import cv2
        while True:
            if idx >= len(self.files):
                if self.loop:
                    idx = 0
                else:
                    break
            path = self.files[idx]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                idx += 1
                continue
            t0 = time.time()
            yield img
            t_used = time.time() - t0
            sleep = delay - t_used
            if sleep > 0:
                time.sleep(sleep)
            idx += 1
