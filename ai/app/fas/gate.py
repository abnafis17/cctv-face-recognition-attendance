from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import time
import numpy as np

from .antispoof import AntiSpoofONNX
from .motion import MotionGate
from .heuristics import sharpness_heuristic, close_face_block

BBox = Tuple[int, int, int, int]


@dataclass
class GateConfig:
    enabled: bool = True
    fas_threshold: float = 0.55

    motion_window_sec: float = 2.0
    min_yaw_range: float = 0.02

    use_heuristics: bool = True
    heuristics_max_var: float = 900.0

    # ✅ NEW: strict anti-phone-near-camera rules
    close_face_max_area_ratio: float = 0.18
    close_face_max_width_ratio: float = 0.60

    cooldown_sec: float = 2.0


class FASGate:
    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Sequence[str]] = None,
        default_cfg: Optional[GateConfig] = None,
        input_size: Tuple[int, int] = (112, 112),
    ):
        self.default_cfg = default_cfg or GateConfig()
        self.cfg_by_camera: Dict[str, GateConfig] = {}

        self.model = AntiSpoofONNX(
            onnx_path=onnx_path,
            threshold=self.default_cfg.fas_threshold,
            input_size=input_size,
            providers=providers or ["CPUExecutionProvider"],
        )

        self.motion = MotionGate(
            window_sec=self.default_cfg.motion_window_sec,
            min_yaw_range=self.default_cfg.min_yaw_range,
        )

        self._last_pass: Dict[str, float] = {}

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        cid = str(camera_id)
        cfg = self.cfg_by_camera.get(cid, GateConfig(**self.default_cfg.__dict__))
        cfg.enabled = bool(enabled)
        self.cfg_by_camera[cid] = cfg

    def get_camera_cfg(self, camera_id: str) -> GateConfig:
        cid = str(camera_id)
        return self.cfg_by_camera.get(cid, self.default_cfg)

    def set_camera_params(
        self,
        camera_id: str,
        *,
        fas_threshold: Optional[float] = None,
        min_yaw_range: Optional[float] = None,
        motion_window_sec: Optional[float] = None,
        use_heuristics: Optional[bool] = None,
        close_face_max_area_ratio: Optional[float] = None,
        close_face_max_width_ratio: Optional[float] = None,
    ) -> GateConfig:
        cid = str(camera_id)
        base = self.cfg_by_camera.get(cid, self.default_cfg)
        cfg = GateConfig(**base.__dict__)

        if fas_threshold is not None:
            cfg.fas_threshold = float(fas_threshold)
        if min_yaw_range is not None:
            cfg.min_yaw_range = float(min_yaw_range)
        if motion_window_sec is not None:
            cfg.motion_window_sec = float(motion_window_sec)
        if use_heuristics is not None:
            cfg.use_heuristics = bool(use_heuristics)
        if close_face_max_area_ratio is not None:
            cfg.close_face_max_area_ratio = float(close_face_max_area_ratio)
        if close_face_max_width_ratio is not None:
            cfg.close_face_max_width_ratio = float(close_face_max_width_ratio)

        self.cfg_by_camera[cid] = cfg
        return cfg

    def check(
        self,
        camera_id: str,
        person_key: str,
        frame_bgr: np.ndarray,
        bbox: BBox,
        kps: Optional[np.ndarray],
    ) -> Tuple[bool, Dict[str, Any]]:
        cfg = self.get_camera_cfg(camera_id)

        if not cfg.enabled:
            return True, {"fas": "disabled"}

        now = time.time()
        key = f"{camera_id}:{person_key}"

        last = self._last_pass.get(key, 0.0)
        if (now - last) < cfg.cooldown_sec:
            return False, {"fas": "cooldown"}

        if cfg.use_heuristics:
            # ✅ NEW strict rule: block "face too close" (phone near camera)
            cr = close_face_block(
                frame_bgr,
                bbox,
                max_face_area_ratio=cfg.close_face_max_area_ratio,
                max_face_width_ratio=cfg.close_face_max_width_ratio,
            )
            if not cr.ok:
                return False, {"fas": "heuristic_block", "h_score": cr.score, "h_reason": cr.reason}

            # Existing sharpness rule
            hr = sharpness_heuristic(frame_bgr, bbox, max_var=cfg.heuristics_max_var)
            if not hr.ok:
                return False, {"fas": "heuristic_block", "h_score": hr.score, "h_reason": hr.reason}

        # model
        self.model.threshold = cfg.fas_threshold
        r = self.model.predict(frame_bgr, bbox)
        if not r.is_live:
            return False, {"fas": "model_spoof", "score": r.score}

        # yaw motion
        self.motion.window_sec = cfg.motion_window_sec
        self.motion.min_yaw_range = cfg.min_yaw_range
        m_ok, m_score = self.motion.update_and_check(key, kps)
        if not m_ok:
            return False, {"fas": "need_pose_change", "score": r.score, "yaw_range": m_score}

        self._last_pass[key] = now
        return True, {"fas": "ok", "score": r.score, "yaw_range": m_score}
