from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque

import cv2
import numpy as np

from .backend_client import BackendClient
from .recognizer import FaceRecognizer, match_gallery
from .tracker import SimpleTracker
from .utils import now_iso, l2_normalize

from .fas.gate import FASGate, GateConfig


@dataclass
class CameraScanState:
    tracker: SimpleTracker = field(default_factory=lambda: SimpleTracker(iou_threshold=0.35, max_age_frames=30))
    last_mark: Dict[str, float] = field(default_factory=dict)  # employee_id(str) -> last_mark_ts
    frame_idx: int = 0


@dataclass
class LivenessState:
    ema: float = 0.5
    last_ts: float = 0.0
    human_hits: int = 0
    spoof_hits: int = 0
    # sticky state (prevents flicker)
    is_human: bool = False


def _put_text_white(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.8) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _nearest_kps(
    track_bbox: Tuple[int, int, int, int],
    det_kps_map: Dict[Tuple[int, int, int, int], Optional[np.ndarray]],
    max_center_dist: float = 60.0,
) -> Optional[np.ndarray]:
    tx1, ty1, tx2, ty2 = track_bbox
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0

    best_kps = None
    best_d = 1e18

    for (x1, y1, x2, y2), kps in det_kps_map.items():
        if kps is None:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        d = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_kps = kps

    if best_kps is None or best_d > max_center_dist:
        return None
    return best_kps


def _clip_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, b)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _face_blur_score(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    Laplacian variance blur metric:
      - low value => blurry (movement)
      - high value => sharp
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class AttendanceRuntime:
    def __init__(
        self,
        use_gpu: bool = False,
        model_name: str = "buffalo_l",
        min_face_size: int = 40,
        similarity_threshold: float = 0.35,
        gallery_refresh_s: float = 5.0,
        cooldown_s: int = 10,
        stable_hits_required: int = 3,
    ):
        self.client = BackendClient()
        self.rec = FaceRecognizer(model_name=model_name, use_gpu=use_gpu, min_face_size=min_face_size)

        self.similarity_threshold = float(similarity_threshold)
        self.gallery_refresh_s = float(gallery_refresh_s)
        self.cooldown_s = int(cooldown_s)
        self.stable_hits_required = int(stable_hits_required)

        self._gallery_last_load = 0.0
        self._gallery_matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self._gallery_meta: List[Tuple[int, str, str]] = []  # (emp_int, emp_id_str, name)

        self._cam_state: Dict[str, CameraScanState] = {}
        self._enabled_for_attendance: Dict[str, bool] = {}

        self._emp_id_to_int: Dict[str, int] = {}
        self._int_to_emp_id: Dict[int, str] = {}
        self._next_emp_int: int = 1_000_000

        # per-camera per-track liveness smoothing
        self._live_state: Dict[str, Dict[int, LivenessState]] = {}

        # -------------
        # FAS
        # -------------
        fas_enabled = os.getenv("FAS_ENABLED", "1") == "1"
        fas_onnx_path = os.getenv("FAS_ONNX_PATH", "app/fas/models/fas.onnx")

        min_yaw_range = os.getenv("FAS_MIN_YAW_RANGE")
        if min_yaw_range is None:
            min_yaw_range = os.getenv("FAS_MIN_MOTION_PX", "0.02")

        self.fas_gate = FASGate(
            onnx_path=fas_onnx_path,
            providers=["CPUExecutionProvider"],
            default_cfg=GateConfig(
                enabled=fas_enabled,
                fas_threshold=float(os.getenv("FAS_THRESHOLD", "0.65")),
                motion_window_sec=float(os.getenv("FAS_MOTION_WINDOW", "2.0")),
                min_yaw_range=float(min_yaw_range),
                use_heuristics=(os.getenv("FAS_USE_HEURISTICS", "1") == "1"),
                cooldown_sec=float(os.getenv("FAS_COOLDOWN_SEC", "2.0")),
            ),
            input_size=(112, 112),
        )

        # ✅ Overlay stability tuning (does NOT affect attendance security)
        self._ema_alpha = float(os.getenv("FAS_OVERLAY_EMA_ALPHA", "0.20"))

        # Make HUMAN easier; SPOOF harder (reduces false NOT HUMAN)
        self._human_on = float(os.getenv("FAS_HUMAN_ON", "0.60"))
        self._human_off = float(os.getenv("FAS_HUMAN_OFF", "0.35"))

        # Need fewer hits to become HUMAN, more hits to become SPOOF
        self._need_human_hits = int(os.getenv("FAS_HUMAN_NEED_HITS", "2"))
        self._need_spoof_hits = int(os.getenv("FAS_SPOOF_NEED_HITS", "6"))

        # Blur guard: if face is blurry, do NOT flip to spoof quickly
        self._blur_hold_var = float(os.getenv("FAS_BLUR_HOLD_VAR", "70"))

    def set_attendance_enabled(self, camera_id: str, enabled: bool) -> None:
        self._enabled_for_attendance[str(camera_id)] = bool(enabled)

    def is_attendance_enabled(self, camera_id: str) -> bool:
        return bool(self._enabled_for_attendance.get(str(camera_id), True))

    def _emp_str_to_int(self, emp_id_str: str) -> int:
        emp_id_str = str(emp_id_str)
        if emp_id_str.isdigit():
            v = int(emp_id_str)
            self._int_to_emp_id[v] = emp_id_str
            return v

        if emp_id_str in self._emp_id_to_int:
            return self._emp_id_to_int[emp_id_str]

        v = self._next_emp_int
        self._next_emp_int += 1
        self._emp_id_to_int[emp_id_str] = v
        self._int_to_emp_id[v] = emp_id_str
        return v

    def _emp_int_to_str(self, emp_int: int) -> str:
        return self._int_to_emp_id.get(int(emp_int), str(emp_int))

    def _ensure_gallery(self) -> None:
        now = time.time()
        if now - self._gallery_last_load < self.gallery_refresh_s:
            return

        templates = self.client.list_templates()
        embs: List[np.ndarray] = []
        meta: List[Tuple[int, str, str]] = []

        for t in templates:
            emp_id_str = str(t.get("employeeId") or t.get("employee_id") or "").strip()
            if not emp_id_str:
                continue

            emb_list = t.get("embedding") or []
            if not isinstance(emb_list, list) or len(emb_list) < 10:
                continue

            emb = np.asarray(emb_list, dtype=np.float32)
            emb = l2_normalize(emb)

            name = str(t.get("employeeName") or t.get("employee_name") or t.get("name") or emp_id_str)
            emp_int = self._emp_str_to_int(emp_id_str)

            embs.append(emb)
            meta.append((emp_int, emp_id_str, name))

        self._gallery_matrix = np.stack(embs, axis=0) if embs else np.zeros((0, 512), dtype=np.float32)
        self._gallery_meta = meta
        self._gallery_last_load = now

    def _get_state(self, camera_id: str) -> CameraScanState:
        cid = str(camera_id)
        if cid not in self._cam_state:
            self._cam_state[cid] = CameraScanState()
        return self._cam_state[cid]

    def process_frame(self, frame_bgr: np.ndarray, camera_id: str) -> np.ndarray:
        self._ensure_gallery()
        cid = str(camera_id)

        state = self._get_state(cid)
        state.frame_idx += 1

        enable_attendance = self.is_attendance_enabled(cid)
        annotated = frame_bgr.copy()

        _put_text_white(annotated, f"cam={cid} frame={state.frame_idx}", 10, 28, scale=0.9)
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")

        dets = self.rec.detect_and_embed(frame_bgr)

        det_list: List[Tuple[np.ndarray, str, int, float]] = []
        det_kps_by_bbox: Dict[Tuple[int, int, int, int], Optional[np.ndarray]] = {}

        for d in dets:
            bbox_key = tuple(int(v) for v in d.bbox)
            det_kps_by_bbox[bbox_key] = d.kps

            idx, sim = match_gallery(d.emb, self._gallery_matrix) if self._gallery_matrix.size else (-1, -1.0)
            if idx != -1 and sim >= self.similarity_threshold:
                emp_int, emp_id_str, name = self._gallery_meta[idx]
                det_list.append((d.bbox, name, int(emp_int), float(sim)))
            else:
                det_list.append((d.bbox, "Unknown", -1, float(sim)))

        tracks = state.tracker.update(
            frame_idx=int(time.time() * 30),
            dets=[(bbox, name, emp_int, sim) for (bbox, name, emp_int, sim) in det_list],
        )

        cam_ls = self._live_state.setdefault(cid, {})

        # cleanup liveness states for removed tracks
        live_ids = set(tr.track_id for tr in tracks)
        for tid in list(cam_ls.keys()):
            if tid not in live_ids:
                cam_ls.pop(tid, None)

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            bbox_key = (x1, y1, x2, y2)

            ls = cam_ls.get(tr.track_id)
            if ls is None:
                ls = LivenessState()
                cam_ls[tr.track_id] = ls

            # raw liveness score (overlay only)
            raw_score, _ = self.fas_gate.score_only(camera_id=cid, frame_bgr=frame_bgr, bbox=bbox_key)

            # EMA smoothing
            a = self._ema_alpha
            ls.ema = (1.0 - a) * ls.ema + a * float(raw_score)
            ls.last_ts = time.time()

            # blur guard
            blur_var = _face_blur_score(frame_bgr, bbox_key)
            is_blurry = blur_var < self._blur_hold_var

            # Update hysteresis counters
            if ls.ema >= self._human_on:
                ls.human_hits += 1
                ls.spoof_hits = 0
            elif ls.ema <= self._human_off:
                # If blurry, DON'T count spoof hits aggressively (prevents false NOT HUMAN)
                if not is_blurry:
                    ls.spoof_hits += 1
                else:
                    ls.spoof_hits = max(0, ls.spoof_hits - 1)
                ls.human_hits = 0
            else:
                # uncertain: decay slowly
                ls.human_hits = max(0, ls.human_hits - 1)
                ls.spoof_hits = max(0, ls.spoof_hits - 1)

            # sticky state transitions
            if ls.human_hits >= self._need_human_hits:
                ls.is_human = True
            if ls.spoof_hits >= self._need_spoof_hits:
                ls.is_human = False

            # ---- draw
            if not ls.is_human:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                _put_text_white(
                    annotated,
                    f"NOT HUMAN (SPOOF) | live={ls.ema:.2f} | blur={blur_var:.0f}",
                    x1,
                    max(24, y1 - 10),
                    scale=0.8,
                )
                continue

            known = (tr.employee_id != -1)
            color = (0, 255, 0) if known else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            emp_id_str = self._emp_int_to_str(tr.employee_id) if known else "-1"
            name = tr.name if known else "Unknown"

            label = f"{name} | sim={tr.similarity:.2f} | live={ls.ema:.2f} | {ts_now}"
            _put_text_white(annotated, label, x1, max(24, y1 - 10), scale=0.8)

            # attendance (still strict)
            if not enable_attendance:
                continue
            if not known:
                continue
            if tr.stable_name_hits < self.stable_hits_required:
                continue

            last = state.last_mark.get(emp_id_str, 0.0)
            now = time.time()
            if now - last < self.cooldown_s:
                continue

            face_kps = _nearest_kps(bbox_key, det_kps_by_bbox)

            fas_ok, _ = self.fas_gate.check(
                camera_id=cid,
                person_key=emp_id_str,
                frame_bgr=frame_bgr,
                bbox=bbox_key,
                kps=face_kps,
            )
            if not fas_ok:
                continue

            try:
                self.client.create_attendance(
                    employee_id=emp_id_str,
                    timestamp=now_iso(),
                    camera_id=cid,
                    confidence=float(tr.similarity),
                    snapshot_path=None,
                )
                state.last_mark[emp_id_str] = now
            except Exception as e:
                print(f"[ATTENDANCE] Failed to mark emp={emp_id_str} cam={cid}: {e}")

        return annotated
