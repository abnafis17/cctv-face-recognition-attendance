from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .backend_client import BackendClient
from .recognizer import FaceRecognizer, match_gallery
from .tracker import SimpleTracker
from .utils import now_iso, l2_normalize


@dataclass
class CameraScanState:
    tracker: SimpleTracker = field(default_factory=lambda: SimpleTracker(iou_threshold=0.35, max_age_frames=30))
    last_mark: Dict[str, float] = field(default_factory=dict)  # employee_id -> last_mark_ts
    frame_idx: int = 0


def _put_text_white(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.8) -> None:
    """
    OpenCV imshow style: white text with black border for readability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    # black outline
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # white text
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


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
        self._gallery_meta: List[Tuple[str, str]] = []  # (employee_id, name)

        self._cam_state: Dict[str, CameraScanState] = {}
        self._enabled_for_attendance: Dict[str, bool] = {}

    def set_attendance_enabled(self, camera_id: str, enabled: bool) -> None:
        self._enabled_for_attendance[str(camera_id)] = bool(enabled)

    def is_attendance_enabled(self, camera_id: str) -> bool:
        return bool(self._enabled_for_attendance.get(str(camera_id), True))

    def _ensure_gallery(self) -> None:
        now = time.time()
        if now - self._gallery_last_load < self.gallery_refresh_s:
            return

        templates = self.client.list_templates()
        embs: List[np.ndarray] = []
        meta: List[Tuple[str, str]] = []

        for t in templates:
            emp_id = str(t.get("employeeId") or t.get("employee_id") or "").strip()
            if not emp_id:
                continue

            emb_list = t.get("embedding") or []
            if not isinstance(emb_list, list) or len(emb_list) < 10:
                continue

            emb = np.asarray(emb_list, dtype=np.float32)
            emb = l2_normalize(emb)

            name = str(t.get("employeeName") or t.get("employee_name") or t.get("name") or emp_id)

            embs.append(emb)
            meta.append((emp_id, name))

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

        # ---- top-left info like OpenCV demo ----
        _put_text_white(annotated, f"cam={cid} frame={state.frame_idx}", 10, 28, scale=0.9)

        dets = self.rec.detect_and_embed(frame_bgr)

        det_list = []
        for d in dets:
            idx, sim = match_gallery(d.emb, self._gallery_matrix) if self._gallery_matrix.size else (-1, -1.0)
            if idx != -1 and sim >= self.similarity_threshold:
                emp_id, name = self._gallery_meta[idx]
                emp_id_int = int(emp_id) if emp_id.isdigit() else -1
                det_list.append((d.bbox, name, emp_id_int, float(sim)))
            else:
                det_list.append((d.bbox, "Unknown", -1, float(sim)))

        tracks = state.tracker.update(
            frame_idx=int(time.time() * 30),
            dets=[(bbox, name, emp_id_int, sim) for (bbox, name, emp_id_int, sim) in det_list],
        )

        # timestamp format like screenshot
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]

            known = (tr.employee_id != -1)

            # ✅ EXACT color rule you requested:
            # known => green, unknown => red
            color = (0, 255, 0) if known else (0, 0, 255)

            # bbox (thickness 3 like OpenCV)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # label: Name | sim=0.83 | datetime
            name = tr.name if known else "Unknown"
            label = f"{name} | sim={tr.similarity:.2f} | {ts_now}"

            # place label above bbox
            text_y = max(24, y1 - 10)
            _put_text_white(annotated, label, x1, text_y, scale=0.8)

            # ---- attendance logic ----
            if not enable_attendance:
                continue
            if not known:
                continue
            if tr.stable_name_hits < self.stable_hits_required:
                continue

            emp_id_str = str(tr.employee_id)
            last = state.last_mark.get(emp_id_str, 0.0)
            now = time.time()
            if now - last < self.cooldown_s:
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
