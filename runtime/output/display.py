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


import cv2
import numpy as np
from ..config.settings import DisplayConfig

class DebugDisplay:
    def __init__(self, cfg: DisplayConfig):
        self.cfg = cfg
        self._created = set()

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        s = self.cfg.window_scale
        if s == 1.0: return img
        h,w = img.shape[:2]
        nh, nw = max(1,int(h*s)), max(1,int(w*s))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def show(self, name: str, img_bgr: np.ndarray):
        if not self.cfg.enable: return
        if name not in self._created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            self._created.add(name)
        cv2.imshow(name, self._maybe_resize(img_bgr))

    def tick(self) -> int:
        """Call once per frame after all windows are shown. Returns key code."""
        if not self.cfg.enable: return -1
        return cv2.waitKey(self.cfg.wait_key_ms) & 0xFF

    def close(self):
        if self.cfg.enable:
            cv2.destroyAllWindows()
