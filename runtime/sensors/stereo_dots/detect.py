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
Dot detection

robust bright-blob centers
"""


from __future__ import annotations
import numpy as np
import cv2



def detect_bright_blobs(gray: np.ndarray,
                        min_area: float = 6.0,
                        max_area: float = 1e6,
                        min_circ: float = 0.3) -> np.ndarray:
    """
    Detect bright blobs in the grayscale image.

    Uses adaptive threshold + contour moments, with simple filters.
    Return Nx2 centers (u,v) in pixel coords.    
    """
    g = gray
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    # mild blur to stabilize
    g = cv2.GaussianBlur(g, (3,3), 0.8)

    # adaptive threshold (bright dots on dark bg)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, -5)
    # cleanup
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # find contours
    cnts,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        # size check   too small or too large
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        # circularity check
        per = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(per*per + 1e-9)
        if circ < min_circ:
            # too elliptical
            continue

        # compute center via moments
        M = cv2.moments(c)
        if abs(M['m00']) < 1e-6: 
            continue
        u = float(M['m10']/M['m00']); v = float(M['m01']/M['m00'])
        centers.append((u,v))

    if len(centers)==0:
        return np.zeros((0,2), dtype=float)
    centers = np.array(centers, dtype=float)
    return centers


