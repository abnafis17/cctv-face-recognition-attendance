from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from .recognizer import FaceRecognizer
from .utils import quality_score, now_iso, estimate_head_pose_deg, pose_label
from .backend_client import BackendClient
from .camera_runtime import CameraRuntime


# -----------------------------
# Data models
# -----------------------------
@dataclass
class EnrollConfig:
    # UI-driven angles (production)
    angles: List[str] = field(default_factory=lambda: ["front", "left", "right", "up", "down"])

    # Minimum quality gate (keep same default behavior)
    min_quality_score: float = 35.0

    # Require pose match for angle
    pose_required: bool = True

    # Pose thresholds (matching your utils.pose_label logic)
    yaw_left_deg: float = -18.0
    yaw_right_deg: float = 18.0
    pitch_up_deg: float = -12.0
    pitch_down_deg: float = 12.0
    tolerance_deg: float = 15.0


@dataclass
class EnrollSession:
    session_id: str
    employee_id: str
    name: str
    camera_id: str
    started_at: str

    status: str = "running"  # running|done|error|stopped
    error: Optional[str] = None

    # UI state
    current_angle: str = "front"

    # Angle -> count collected (staged)
    collected: Dict[str, int] = field(default_factory=dict)

    # last feedback
    last_quality: float = 0.0
    last_pose: Optional[str] = None
    last_message: Optional[str] = None
    last_update_at: str = field(default_factory=now_iso)


# -----------------------------
# Enrollment service (UI-driven)
# -----------------------------
class EnrollmentService:
    """
    UI-driven enrollment (NO OpenCV window, NO auto background collecting):
    - start(employee_id, name, camera_id): start a session
    - set_angle(angle): change current angle
    - capture(): capture ONE embedding for current angle (staged in memory)
    - save(): average per angle and upsert FaceTemplate to backend DB
    - cancel(): clear staged captures
    - stop(): stop session
    """

    def __init__(
        self,
        camera_rt: CameraRuntime,
        use_gpu: bool = True,
        model_name: str = "buffalo_l",
        min_face_size: int = 40,
    ):
        self.camera_rt = camera_rt
        self.rec = FaceRecognizer(model_name=model_name, use_gpu=use_gpu, min_face_size=min_face_size)
        self.client = BackendClient()
        self.cfg = EnrollConfig()

        self._lock = threading.Lock()
        self._session: Optional[EnrollSession] = None

        # staged embeddings: angle -> list[np.ndarray]
        self._embs: Dict[str, List[np.ndarray]] = {}

    # -------- Session controls --------
    def start(self, employee_id: str, name: str, camera_id: str) -> EnrollSession:
        employee_id = str(employee_id).strip()
        name = str(name).strip()
        camera_id = str(camera_id).strip()

        if not employee_id or not name or not camera_id:
            raise ValueError("employee_id, name, camera_id are required")

        with self._lock:
            sid = f"enroll_{int(time.time())}"
            self._session = EnrollSession(
                session_id=sid,
                employee_id=employee_id,
                name=name,
                camera_id=camera_id,
                started_at=now_iso(),
                status="running",
                current_angle="front",
                collected={a: 0 for a in self.cfg.angles},
            )
            self._embs = {a: [] for a in self.cfg.angles}

        # Ensure employee exists in backend (outside lock)
        self.client.upsert_employee(name, employee_id)

        return self.status()  # type: ignore

    def stop(self) -> bool:
        with self._lock:
            if not self._session or self._session.status != "running":
                return False
            self._session.status = "stopped"
            self._session.last_message = "Stopped"
            self._session.last_update_at = now_iso()
            return True

    def cancel(self) -> Dict[str, Any]:
        """
        Clears staged embeddings (undo captures) but keeps session running.
        """
        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("No running enrollment session")
            self._embs = {a: [] for a in self.cfg.angles}
            self._session.collected = {a: 0 for a in self.cfg.angles}
            self._session.last_message = "Canceled staged captures"
            self._session.last_update_at = now_iso()
            return {"cleared": True, "angles": list(self.cfg.angles)}

    def status(self) -> Optional[EnrollSession]:
        with self._lock:
            return self._session

    # -------- UI actions --------
    def set_angle(self, angle: str) -> Optional[EnrollSession]:
        angle = str(angle).strip().lower()
        if angle not in set(self.cfg.angles):
            raise ValueError(f"Invalid angle: {angle}. Allowed: {self.cfg.angles}")

        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("No running enrollment session")
            self._session.current_angle = angle
            self._session.last_message = f"Angle set to {angle}"
            self._session.last_update_at = now_iso()
            return self._session

    def capture(self) -> Dict[str, Any]:
        """
        Captures ONE embedding for the current angle and stages it in memory.
        Applies quality + (optional) pose checks.
        """
        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("No running enrollment session")
            session = self._session
            angle = session.current_angle

        frame = self.camera_rt.get_frame(session.camera_id)
        if frame is None:
            return {"ok": False, "error": "No frame yet from camera", "angle": angle}

        dets = self.rec.detect_and_embed(frame)
        if not dets:
            self._update_last(q=0.0, pose=None, msg="No face detected")
            return {"ok": False, "error": "No face detected", "angle": angle}

        # Pick largest face
        det = max(dets, key=lambda d: float((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])))

        q = float(quality_score(det.bbox, frame))

        pose_name: Optional[str] = None
        pose_ok = True
        if self.cfg.pose_required and det.kps is not None:
            pose = estimate_head_pose_deg(det.kps, frame.shape)
            if pose:
                yaw, pitch, _roll = pose
                pose_name = pose_label(
                    yaw,
                    pitch,
                    {
                        "yaw_left_deg": self.cfg.yaw_left_deg,
                        "yaw_right_deg": self.cfg.yaw_right_deg,
                        "pitch_up_deg": self.cfg.pitch_up_deg,
                        "pitch_down_deg": self.cfg.pitch_down_deg,
                        "tolerance_deg": self.cfg.tolerance_deg,
                    },
                )
                pose_ok = (pose_name == angle)

        # Update live feedback
        self._update_last(q=q, pose=pose_name, msg="")

        if q < self.cfg.min_quality_score:
            self._update_last(q=q, pose=pose_name, msg=f"Low quality ({q:.1f})")
            return {"ok": False, "error": f"Low quality ({q:.1f})", "angle": angle, "quality": q, "pose": pose_name}

        if self.cfg.pose_required and not pose_ok:
            self._update_last(q=q, pose=pose_name, msg=f"Pose mismatch: saw '{pose_name}', need '{angle}'")
            return {
                "ok": False,
                "error": f"Pose mismatch: saw '{pose_name}', need '{angle}'",
                "angle": angle,
                "quality": q,
                "pose": pose_name,
            }

        # Stage embedding
        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("Enrollment session ended")

            self._embs[angle].append(det.emb)
            self._session.collected[angle] = len(self._embs[angle])
            self._session.last_message = f"Captured {angle} ({self._session.collected[angle]})"
            self._session.last_update_at = now_iso()

            return {
                "ok": True,
                "angle": angle,
                "quality": q,
                "pose": pose_name,
                "count_for_angle": self._session.collected[angle],
                "staged": dict(self._session.collected),
            }

    def save(self) -> Dict[str, Any]:
        """
        Averages embeddings per angle (if any) and upserts FaceTemplate to backend DB.
        Clears staged captures after successful save.
        """
        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("No running enrollment session")
            session = self._session
            embs_copy = {a: list(v) for a, v in self._embs.items()}

        saved_angles: List[str] = []
        skipped_angles: List[str] = []

        for angle, vecs in embs_copy.items():
            if not vecs:
                skipped_angles.append(angle)
                continue

            arr = np.stack(vecs, axis=0)  # (N, D)
            mean = arr.mean(axis=0).astype(np.float32)
            mean = mean / (np.linalg.norm(mean) + 1e-12)

            self.client.upsert_template(
                employee_id=session.employee_id,
                angle=angle,
                embedding=mean.tolist(),
                model_name="insightface",
            )
            saved_angles.append(angle)

        with self._lock:
            if not self._session:
                return {"ok": True, "saved_angles": saved_angles, "skipped_angles": skipped_angles}

            # Clear staged data after save
            self._embs = {a: [] for a in self.cfg.angles}
            self._session.collected = {a: 0 for a in self.cfg.angles}
            self._session.last_message = f"Saved angles: {saved_angles}" if saved_angles else "Nothing to save"
            self._session.last_update_at = now_iso()

        return {"ok": True, "saved_angles": saved_angles, "skipped_angles": skipped_angles}

    # -------- internal helpers --------
    def _update_last(self, q: float, pose: Optional[str], msg: str):
        with self._lock:
            if not self._session:
                return
            self._session.last_quality = float(q)
            self._session.last_pose = pose
            if msg:
                self._session.last_message = msg
            self._session.last_update_at = now_iso()


    # -------- Clear Angle --------
    def clear_angle(self, angle: str) -> Dict[str, Any]:
        angle = str(angle).strip().lower()
        if angle not in set(self.cfg.angles):
            raise ValueError(f"Invalid angle: {angle}")
        with self._lock:
            if not self._session or self._session.status != "running":
                raise RuntimeError("No running enrollment session")
            self._embs[angle] = []
            self._session.collected[angle] = 0
            self._session.last_message = f"Cleared angle: {angle}"
            self._session.last_update_at = now_iso()
            return {"cleared": True, "angle": angle}
