from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class HeuristicResult:
    ok: bool
    score: float
    reason: str


def sharpness_heuristic(frame_bgr: np.ndarray, bbox: BBox, max_var: float = 900.0) -> HeuristicResult:
    """
    Optional cheap heuristic:
    - Very high sharpness variance sometimes indicates phone-screen replay.
    - Keep conservative; don't rely solely on this.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return HeuristicResult(False, 0.0, "empty_crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if var > max_var:
        return HeuristicResult(False, var, "too_sharp_screen_like")

    return HeuristicResult(True, var, "ok")


def close_face_block(
    frame_bgr: np.ndarray,
    bbox: BBox,
    *,
    max_face_area_ratio: float = 0.18,
    max_face_width_ratio: float = 0.60,
) -> HeuristicResult:
    """
    STRONG anti-phone-near-camera filter.

    Blocks when face bbox is "too large" relative to frame.
    This commonly happens when a phone screen is held very close to the camera.

    - max_face_area_ratio: (bbox_area / frame_area)
    - max_face_width_ratio: (bbox_width / frame_width)

    Tune carefully:
      - If your real faces are close to camera normally, increase ratios slightly.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    face_area = float(bw * bh)
    frame_area = float(w * h)
    area_ratio = face_area / max(1.0, frame_area)

    width_ratio = float(bw) / max(1.0, float(w))

    # If face is extremely large -> likely phone close
    if area_ratio > float(max_face_area_ratio):
        return HeuristicResult(False, area_ratio, "face_too_close_area_ratio")

    if width_ratio > float(max_face_width_ratio):
        return HeuristicResult(False, width_ratio, "face_too_close_width_ratio")

    return HeuristicResult(True, max(area_ratio, width_ratio), "ok")
