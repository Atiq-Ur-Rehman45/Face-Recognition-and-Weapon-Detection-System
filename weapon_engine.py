import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    WEAPON_MODEL_PATH,
    WEAPON_INPUT_SIZE,
    WEAPON_CONFIDENCE_THRESHOLD,
    WEAPON_NMS_THRESHOLD,
    WEAPON_MIN_BOX_AREA,
    WEAPON_RESULT_TTL_SECONDS,
    WEAPON_WORKER_SLEEP_SECONDS,
    WEAPON_CLASSES,
    WEAPON_THREAT_LEVELS,
)


class AsyncWeaponDetector:
    """Asynchronous weapon detector that always processes the latest frame."""

    def __init__(self, model_path: str = WEAPON_MODEL_PATH):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = []

        self._running = False
        self._thread = None

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_id = -1

        self._result_lock = threading.Lock()
        self._latest_result = {
            "frame_id": -1,
            "timestamp": 0.0,
            "detections": [],
            "model_loaded": False,
        }

        self.model_loaded = False
        self.last_error = None
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.model_loaded = True

            with self._result_lock:
                self._latest_result["model_loaded"] = True
        except Exception as exc:
            self.last_error = str(exc)
            self.model_loaded = False

    def start(self):
        if not self.model_loaded or self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        if not self.model_loaded:
            return

        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_frame_id = frame_id

    def get_latest_result(self) -> Dict:
        with self._result_lock:
            result = {
                "frame_id": self._latest_result["frame_id"],
                "timestamp": self._latest_result["timestamp"],
                "detections": list(self._latest_result["detections"]),
                "model_loaded": self._latest_result["model_loaded"],
            }

        age = time.time() - result["timestamp"]
        if age > WEAPON_RESULT_TTL_SECONDS:
            result["detections"] = []
        return result

    def _run(self):
        last_processed_id = -1
        while self._running:
            frame = None
            frame_id = -1

            with self._frame_lock:
                if self._latest_frame is not None and self._latest_frame_id != last_processed_id:
                    frame = self._latest_frame
                    frame_id = self._latest_frame_id

            if frame is None:
                time.sleep(WEAPON_WORKER_SLEEP_SECONDS)
                continue

            detections = self._detect_sync(frame)
            last_processed_id = frame_id

            with self._result_lock:
                self._latest_result = {
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "detections": detections,
                    "model_loaded": True,
                }

    def _detect_sync(self, frame: np.ndarray) -> List[Dict]:
        if not self.model_loaded:
            return []

        input_tensor = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self._postprocess(outputs, frame.shape[:2])

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        target_w, target_h = WEAPON_INPUT_SIZE
        img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _postprocess(self, outputs, orig_hw) -> List[Dict]:
        pred = outputs[0]

        if len(pred.shape) == 3:
            if pred.shape[1] < pred.shape[2]:
                pred = np.transpose(pred, (0, 2, 1))
            pred = pred[0]

        if pred.ndim != 2 or pred.shape[1] < 6:
            return []

        # Support both common YOLO ONNX layouts:
        # 1) [x, y, w, h, obj, c1..cN]
        # 2) [x, y, w, h, c1..cN] (no explicit objectness)
        known_class_count = (max(WEAPON_CLASSES.keys()) + 1) if WEAPON_CLASSES else 0
        has_objectness = pred.shape[1] >= (5 + known_class_count) if known_class_count > 0 else True

        orig_h, orig_w = orig_hw
        in_w, in_h = WEAPON_INPUT_SIZE
        scale_x = float(orig_w) / float(in_w)
        scale_y = float(orig_h) / float(in_h)

        boxes = []
        scores = []
        classes = []

        for row in pred:
            x_c, y_c, w, h = row[:4]

            if has_objectness and row.shape[0] > 5:
                obj_conf = float(row[4])
                class_scores = row[5:]
            else:
                obj_conf = 1.0
                class_scores = row[4:]

            if class_scores.size == 0:
                continue

            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            confidence = obj_conf * class_conf

            if confidence < WEAPON_CONFIDENCE_THRESHOLD:
                continue

            x1 = int((x_c - (w / 2.0)) * scale_x)
            y1 = int((y_c - (h / 2.0)) * scale_y)
            bw = int(w * scale_x)
            bh = int(h * scale_y)

            if bw <= 0 or bh <= 0:
                continue
            if bw * bh < WEAPON_MIN_BOX_AREA:
                continue

            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            bw = min(bw, orig_w - x1)
            bh = min(bh, orig_h - y1)

            boxes.append([x1, y1, bw, bh])
            scores.append(confidence)
            classes.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, WEAPON_CONFIDENCE_THRESHOLD, WEAPON_NMS_THRESHOLD)
        if len(indices) == 0:
            return []

        detections = []
        for idx in np.array(indices).flatten():
            x1, y1, bw, bh = boxes[idx]
            class_id = classes[idx]
            weapon_type = WEAPON_CLASSES.get(class_id, f"weapon_{class_id}")
            detections.append(
                {
                    "type": weapon_type,
                    "confidence": float(scores[idx]),
                    "threat_level": WEAPON_THREAT_LEVELS.get(weapon_type, "UNKNOWN"),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x1 + bw),
                        "y2": int(y1 + bh),
                    },
                }
            )

        return detections


def draw_weapon_detections(frame: np.ndarray, detections: List[Dict]):
    """Draw weapon boxes onto a frame in-place."""
    threat_colors = {
        "CRITICAL": (0, 0, 255),
        "HIGH": (0, 140, 255),
        "MEDIUM": (0, 220, 220),
        "UNKNOWN": (160, 160, 160),
    }

    for det in detections:
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        threat = det.get("threat_level", "UNKNOWN")
        color = threat_colors.get(threat, (160, 160, 160))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.get('type', 'weapon')} {det.get('confidence', 0.0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
