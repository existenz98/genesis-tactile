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


import cv2
import numpy as np
from ..config.settings import DisplayConfig

class DebugDisplay:
    def __init__(self, cfg: DisplayConfig):
        self.cfg = cfg

        # Track all created windows
        self._created = set()

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image according to config."""
        s = self.cfg.window_scale

        # No resizing
        if s == 1.0: return img

        # Resize
        h,w = img.shape[:2]
        nh, nw = max(1,int(h*s)), max(1,int(w*s))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def show(self, name: str, img_bgr: np.ndarray):
        if not self.cfg.enable:
            return

        # Create window only if not exists
        if name not in self._created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            self._created.add(name)

        cv2.imshow(name, self._maybe_resize(img_bgr))

    def tick(self) -> int:
        """
        Call once for all windows to refresh.
        Returns key code.
        """
        if not self.cfg.enable: return -1
        return cv2.waitKey(self.cfg.wait_key_ms) & 0xFF

    def close(self):
        if self.cfg.enable:
            cv2.destroyAllWindows()
