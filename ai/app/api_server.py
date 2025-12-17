from __future__ import annotations

import time
import cv2
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, Response

from .camera_runtime import CameraRuntime
from .enroll_service import EnrollmentService
from .attendance_runtime import AttendanceRuntime

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="AI Camera API", version="1.3")

# --------------------------------------------------
# Runtimes (order matters)
# --------------------------------------------------
camera_rt = CameraRuntime()

enroller = EnrollmentService(
    camera_rt=camera_rt,
    use_gpu=False,   # set True if GPU available
)

attendance_rt = AttendanceRuntime(
    use_gpu=False,               # GPU optional
    similarity_threshold=0.35,   # tune if needed
    cooldown_s=10,               # seconds between marks
    stable_hits_required=3,      # frames needed
)

# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Camera control
# --------------------------------------------------
@app.api_route("/camera/start", methods=["GET", "POST"])
def start_camera(camera_id: str, rtsp_url: str):
    camera_rt.start(camera_id, rtsp_url)
    return {"ok": True, "camera_id": camera_id, "rtsp_url": rtsp_url}


@app.api_route("/camera/stop", methods=["GET", "POST"])
def stop_camera(camera_id: str):
    camera_rt.stop(camera_id)
    return {"ok": True, "camera_id": camera_id}

# --------------------------------------------------
# Snapshot (debug / fallback)
# --------------------------------------------------
@app.get("/camera/snapshot/{camera_id}")
def camera_snapshot(camera_id: str):
    frame = camera_rt.get_frame(camera_id)
    if frame is None:
        return Response(content=b"No frame yet", status_code=503)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return Response(content=b"Encode failed", status_code=500)

    return Response(
        content=jpg.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

# --------------------------------------------------
# RAW MJPEG stream (no recognition)
# --------------------------------------------------
def mjpeg_generator_raw(camera_id: str):
    for _ in range(60):
        if camera_rt.get_frame(camera_id) is not None:
            break
        time.sleep(0.05)

    try:
        while True:
            frame = camera_rt.get_frame(camera_id)
            if frame is None:
                time.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )
            time.sleep(0.01)

    except GeneratorExit:
        return


@app.get("/camera/stream/{camera_id}")
def camera_stream(camera_id: str):
    return StreamingResponse(
        mjpeg_generator_raw(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
        },
    )

# --------------------------------------------------
# RECOGNITION + ATTENDANCE STREAM
# --------------------------------------------------
def mjpeg_generator_recognition(camera_id: str):
    for _ in range(60):
        if camera_rt.get_frame(camera_id) is not None:
            break
        time.sleep(0.05)

    try:
        while True:
            frame = camera_rt.get_frame(camera_id)
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = attendance_rt.process_frame(
                frame_bgr=frame,
                camera_id=camera_id,
            )

            ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )
            time.sleep(0.01)

    except GeneratorExit:
        return


@app.get("/camera/recognition/stream/{camera_id}")
def camera_recognition_stream(camera_id: str):
    """
    Browser-visible stream with:
    - face boxes
    - name + similarity
    - automatic attendance logging
    """
    return StreamingResponse(
        mjpeg_generator_recognition(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
        },
    )

# --------------------------------------------------
# Attendance enable / disable per camera (optional)
# --------------------------------------------------
@app.post("/attendance/enable")
def attendance_enable(camera_id: str):
    attendance_rt.set_attendance_enabled(camera_id, True)
    return {"ok": True, "camera_id": camera_id, "enabled": True}


@app.post("/attendance/disable")
def attendance_disable(camera_id: str):
    attendance_rt.set_attendance_enabled(camera_id, False)
    return {"ok": True, "camera_id": camera_id, "enabled": False}


@app.get("/attendance/enabled")
def attendance_enabled(camera_id: str):
    return {
        "ok": True,
        "camera_id": camera_id,
        "enabled": attendance_rt.is_attendance_enabled(camera_id),
    }

# --------------------------------------------------
# Enrollment (Browser-based)
# --------------------------------------------------
@app.post("/enroll/session/start")
def enroll_session_start(payload: dict = Body(...)):
    employee_id = str(payload.get("employeeId") or "").strip()
    name = str(payload.get("name") or "").strip()
    camera_id = str(payload.get("cameraId") or "").strip()

    if not employee_id or not name or not camera_id:
        return {"ok": False, "error": "employeeId, name, cameraId are required"}

    s = enroller.start(employee_id=employee_id, name=name, camera_id=camera_id)
    return {"ok": True, "session": s.__dict__}


@app.post("/enroll/session/stop")
def enroll_session_stop():
    stopped = enroller.stop()
    s = enroller.status()
    return {
        "ok": True,
        "stopped": stopped,
        "session": (s.__dict__ if s else None),
    }


@app.get("/enroll/session/status")
def enroll_session_status():
    s = enroller.status()
    return {"ok": True, "session": (s.__dict__ if s else None)}
