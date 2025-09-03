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


import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..config.settings import UnmixConfig, UnmixMode
from .photo_compensator import bgr_to_rgb, srgb_to_linear_uint8

def _chromaticity(img_lin: np.ndarray) -> np.ndarray:
    L = img_lin.sum(axis=-1, keepdims=True) + 1e-8
    return img_lin / L

def _kmeans_pp(X: np.ndarray, k: int, iters: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n,d = X.shape
    centers = np.empty((k,d), dtype=np.float32)
    centers[0] = X[rng.integers(0,n)]
    cd2 = np.sum((X - centers[0])**2, axis=1)
    for i in range(1,k):
        probs = cd2 / (cd2.sum()+1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        d2 = np.sum((X - centers[i])**2, axis=1)
        cd2 = np.minimum(cd2, d2)
    for _ in range(iters):
        dists = np.stack([np.sum((X-c)**2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        new = centers.copy()
        for j in range(k):
            m = labels==j
            if np.any(m):
                new[j] = X[m].mean(axis=0)
            else:
                new[j] = X[rng.integers(0,n)]
        if np.linalg.norm(new-centers) < 1e-6:
            centers = new
            break
        centers = new
    return centers

@dataclass
class UnmixModel:
    cfg: UnmixConfig
    B: Optional[np.ndarray] = None  # 3x3 chromaticity basis (columns=R,G,B)

    def fit(self, balanced_bgr_uint8: np.ndarray, mask_valid: Optional[np.ndarray]=None) -> "UnmixModel":
        assert self.cfg.mode == UnmixMode.KMEANS, "Only KMEANS mode implemented in fit"
        img_lin = srgb_to_linear_uint8(bgr_to_rgb(balanced_bgr_uint8))
        X = _chromaticity(img_lin).reshape(-1,3)
        if mask_valid is not None:
            X = X[mask_valid.reshape(-1)]
        # subsample
        if X.shape[0] > self.cfg.kmeans_samples:
            idx = np.random.default_rng(0).choice(X.shape[0], size=self.cfg.kmeans_samples, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        centers = _kmeans_pp(Xs.astype(np.float32), 3, self.cfg.kmeans_iters, seed=1)
        mapping = np.argsort(np.argmax(centers, axis=1))
        ordered = centers[mapping]
        self.B = ordered.T.astype(np.float32)
        return self

    def transform(self, balanced_bgr_uint8: np.ndarray) -> np.ndarray:
        assert self.B is not None, "call fit() first"
        img_lin = srgb_to_linear_uint8(bgr_to_rgb(balanced_bgr_uint8))
        L = img_lin.sum(axis=-1, keepdims=True) + 1e-8
        X = img_lin / L
        reg = self.cfg.reg * np.eye(3, dtype=np.float32)
        B = self.B
        B_inv = np.linalg.inv(B.T @ B + reg) @ B.T
        alpha = (X.reshape(-1,3) @ B_inv.T)
        alpha = np.maximum(alpha, 0.0)
        alpha /= (alpha.sum(axis=1, keepdims=True)+1e-8)
        C = (alpha * L.reshape(-1,1)).reshape(balanced_bgr_uint8.shape)  # linear intensities
        return C.astype(np.float32)
