from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import numpy as np


@dataclass
class PoseState:
    last_ts: float
    yaw_min: float
    yaw_max: float


class MotionGate:
    """
    Stronger than center-drift:
    Require small change in a yaw proxy derived from landmarks.

    Why this blocks phone replay better:
      - Moving a phone changes the whole image position
      - But a flat photo/video often keeps landmark geometry (yaw proxy) nearly stable
      - Real faces naturally vary yaw slightly over 1–2 seconds
    """

    def __init__(self, window_sec: float = 1.5, min_yaw_range: float = 0.035):
        self.window_sec = float(window_sec)
        self.min_yaw_range = float(min_yaw_range)
        self._st: Dict[str, PoseState] = {}

    @staticmethod
    def _yaw_proxy_from_kps(kps: Optional[np.ndarray]) -> Optional[float]:
        """
        InsightFace kps usually: (5,2) = [L_eye, R_eye, Nose, L_mouth, R_mouth]
        yaw proxy = (nose_x - mid_eye_x) / eye_distance
        """
        if kps is None:
            return None
        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim != 2 or kps.shape[1] != 2 or kps.shape[0] < 3:
            return None

        le = kps[0]
        re = kps[1]
        nose = kps[2]

        mid = (le + re) / 2.0
        eye_dist = float(np.linalg.norm(re - le))
        if eye_dist < 1e-6:
            return None

        return float((nose[0] - mid[0]) / eye_dist)

    def update_and_check(self, key: str, kps: Optional[np.ndarray]) -> Tuple[bool, float]:
        now = time.time()
        yaw = self._yaw_proxy_from_kps(kps)
        if yaw is None:
            return False, 0.0

        st = self._st.get(key)
        if st is None:
            self._st[key] = PoseState(last_ts=now, yaw_min=yaw, yaw_max=yaw)
            return False, 0.0

        if (now - st.last_ts) > self.window_sec:
            st.yaw_min = yaw
            st.yaw_max = yaw
        else:
            st.yaw_min = min(st.yaw_min, yaw)
            st.yaw_max = max(st.yaw_max, yaw)

        st.last_ts = now

        yaw_range = float(st.yaw_max - st.yaw_min)
        ok = yaw_range >= self.min_yaw_range
        return ok, yaw_range

    def cleanup(self, max_age_sec: float = 5.0) -> None:
        now = time.time()
        dead = [k for k, st in self._st.items() if (now - st.last_ts) > max_age_sec]
        for k in dead:
            self._st.pop(k, None)
