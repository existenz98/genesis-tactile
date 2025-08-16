import numpy as np
import cv2
from typing import Iterator

def flow_to_color_bgr(vy: np.ndarray, vx: np.ndarray, max_flow: float = None) -> np.ndarray:
    mag = np.sqrt(vy**2 + vx**2)
    ang = np.arctan2(-vy, vx)  # invert y for image coords
    if max_flow is None:
        max_flow = np.percentile(mag, 95)
        if max_flow < 1e-6: max_flow = 1e-6
    H = (ang + np.pi) / (2*np.pi)        # [0,1)
    S = np.ones_like(H, dtype=np.float32)
    V = np.clip(mag / max_flow, 0, 1)
    hsv = np.stack([H*179.0, S*255.0, V*255.0], axis=-1).astype(np.uint8)  # OpenCV HSV scales
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

class VideoWriters:
    def __init__(self, out_dir: str, fps: float):
        self.out_dir = out_dir
        self.fps = fps
        self.writers = {}

    def _open(self, key: str, frame_shape):
        import os
        os.makedirs(self.out_dir, exist_ok=True)
        h,w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = os.path.join(self.out_dir, f"{key}.mp4")
        wr = cv2.VideoWriter(path, fourcc, self.fps, (w,h))
        self.writers[key] = (wr, path)
        return wr

    def write(self, key: str, frame_bgr: np.ndarray):
        if key not in self.writers:
            self._open(key, frame_bgr.shape)
        self.writers[key][0].write(frame_bgr)

    def close(self):
        for wr, _ in self.writers.values():
            wr.release()

    def paths(self):
        return {k: p for k, (_, p) in self.writers.items()}
