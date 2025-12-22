from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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


def _put_text_white(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.8) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _nearest_kps(track_bbox: Tuple[int, int, int, int],
                 det_kps_map: Dict[Tuple[int, int, int, int], Optional[np.ndarray]],
                 max_center_dist: float = 50.0) -> Optional[np.ndarray]:
    """
    Tracker bbox is often slightly different from detector bbox.
    This finds the nearest detector bbox center and returns its kps.
    """
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

        # ---------------------------
        # Face Anti-Spoofing (FAS)
        # ---------------------------
        fas_enabled = os.getenv("FAS_ENABLED", "1") == "1"
        fas_onnx_path = os.getenv("FAS_ONNX_PATH", "app/fas/models/fas.onnx")

        # Backward compatible:
        # - new: FAS_MIN_YAW_RANGE
        # - old: FAS_MIN_MOTION_PX (we map it to yaw range if new one not set)
        min_yaw_range = os.getenv("FAS_MIN_YAW_RANGE")
        if min_yaw_range is None:
            min_yaw_range = os.getenv("FAS_MIN_MOTION_PX", "0.035")

        self.fas_gate = FASGate(
            onnx_path=fas_onnx_path,
            providers=["CPUExecutionProvider"],
            default_cfg=GateConfig(
                enabled=fas_enabled,
                fas_threshold=float(os.getenv("FAS_THRESHOLD", "0.55")),
                motion_window_sec=float(os.getenv("FAS_MOTION_WINDOW", "1.5")),
                min_yaw_range=float(min_yaw_range),
                use_heuristics=(os.getenv("FAS_USE_HEURISTICS", "1") == "1"),
                cooldown_sec=float(os.getenv("FAS_COOLDOWN_SEC", "2.0")),
            ),
            input_size=(112, 112),
        )

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

        det_list = []
        det_kps_by_bbox: Dict[Tuple[int, int, int, int], Optional[np.ndarray]] = {}

        for d in dets:
            idx, sim = match_gallery(d.emb, self._gallery_matrix) if self._gallery_matrix.size else (-1, -1.0)

            bbox_key = tuple(int(v) for v in d.bbox)
            det_kps_by_bbox[bbox_key] = d.kps

            if idx != -1 and sim >= self.similarity_threshold:
                emp_int, emp_id_str, name = self._gallery_meta[idx]
                det_list.append((d.bbox, name, int(emp_int), float(sim)))
            else:
                det_list.append((d.bbox, "Unknown", -1, float(sim)))

        tracks = state.tracker.update(
            frame_idx=int(time.time() * 30),
            dets=[(bbox, name, emp_int, sim) for (bbox, name, emp_int, sim) in det_list],
        )

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]

            known = (tr.employee_id != -1)
            color = (0, 255, 0) if known else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            if known:
                emp_id_str = self._emp_int_to_str(tr.employee_id)
                name = tr.name
            else:
                emp_id_str = "-1"
                name = "Unknown"

            label = f"{name} | sim={tr.similarity:.2f} | {ts_now}"
            _put_text_white(annotated, label, x1, max(24, y1 - 10), scale=0.8)

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

            bbox_key = (x1, y1, x2, y2)

            # ✅ IMPORTANT: nearest kps match (tracker bbox != detector bbox)
            face_kps = _nearest_kps(bbox_key, det_kps_by_bbox)

            fas_ok, fas_dbg = self.fas_gate.check(
                camera_id=cid,
                person_key=emp_id_str,
                frame_bgr=frame_bgr,
                bbox=bbox_key,
                kps=face_kps,
            )

            print(
                "[FAS DEBUG]",
                "emp=", emp_id_str,
                "ok=", fas_ok,
                "dbg=", fas_dbg,
                "kps_none=", face_kps is None
            )

            if not fas_ok:
                # Optional debug overlay:
                # _put_text_white(annotated, f"FAS BLOCK: {fas_dbg.get('fas')}", x1, y2 + 22, scale=0.7)
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
