from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None

BBox = Tuple[int, int, int, int]


@dataclass
class FASResult:
    is_live: bool
    score: float
    reason: str = ""


class AntiSpoofONNX:
    """
    Configured for your model:
      - Input:  [1, 3, 112, 112] float
      - Output: [1, 2] logits => [spoof_logit, live_logit]
      - We apply softmax and use probs[1] as LIVE probability.
    """

    def __init__(
        self,
        onnx_path: str,
        threshold: float = 0.55,
        input_size: Tuple[int, int] = (112, 112),
        providers: Optional[Sequence[str]] = None,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime (or onnxruntime-gpu)")

        self.onnx_path = onnx_path
        self.threshold = float(threshold)
        self.input_size = (int(input_size[0]), int(input_size[1]))  # (W,H) for cv2.resize

        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, providers=self.providers)
        self.in_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _clip_bbox(b: BBox, w: int, h: int) -> BBox:
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

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Correct preprocessing for modelrgb.onnx:
        - resize to 112x112
        - BGR -> RGB
        - float32
        - ImageNet normalization
        - NCHW
        """
        img = cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))[None, ...]  # 1x3x112x112
        return img


    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _parse_output_score(self, out: list) -> float:
        logits = np.array(out[0]).reshape(-1).astype(np.float32)
        if logits.size < 2:
            return float(logits[0]) if logits.size == 1 else 0.0

        probs = self._softmax(logits)
        return float(probs[0])  # ✅ index 0 = LIVE


    def predict(self, frame_bgr: np.ndarray, bbox: BBox) -> FASResult:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._clip_bbox(bbox, w, h)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return FASResult(False, 0.0, "empty_crop")

        inp = self._preprocess(crop)
        out = self.sess.run(None, {self.in_name: inp})
        score = float(self._parse_output_score(out))

        ok = score >= self.threshold
        return FASResult(ok, score, "live" if ok else "spoof")
