from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
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
