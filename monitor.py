import cv2
import numpy as np
import os
import queue
import threading
import time
import logging
from datetime import datetime
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
    COLOR_BLACK, COLOR_ORANGE, COLOR_BLUE,
    ALERT_COOLDOWN_SECONDS, SNAPSHOT_ON_DETECTION,
    CAPTURED_DIR, RECOGNITION_CONFIDENCE_THRESHOLD,
    SFACE_MATCH_THRESHOLD, RECOGNITION_ENGINE, UNKNOWN_LABEL,
    CAMERA_FPS_TARGET, CAMERA_AUTOFOCUS, CAMERA_BUFFER_SIZE,
    WARMUP_FRAMES, ASYNC_CAMERA_CAPTURE, ASYNC_ALERT_PROCESSING,
    ALERT_WORKER_QUEUE_SIZE
)

logger = logging.getLogger(__name__)


class LatestFrameCamera:
    """Continuously reads from the camera and exposes only the newest frame."""

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self._frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(CAMERA_AUTOFOCUS))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"[MONITOR] Cannot open camera index {self.camera_index}")

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def frame_count(self):
        with self._lock:
            return self._frame_count

    def stop(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()


class DirectCamera:
    """Synchronous camera wrapper with the same interface as LatestFrameCamera."""

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self._frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(CAMERA_AUTOFOCUS))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"[MONITOR] Cannot open camera index {self.camera_index}")

        return self

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def frame_count(self):
        return self._frame_count

    def stop(self):
        if self.cap is not None:
            self.cap.release()


class AlertWorker:
    """Processes slow alert side-effects away from the recognition loop."""

    def __init__(self, processor, maxsize):
        self.processor = processor
        self.queue = queue.Queue(maxsize=maxsize)
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def submit(self, payload):
        try:
            self.queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("[MONITOR] Alert queue full; processing alert inline")
            return False

    def _run(self):
        while self._running or not self.queue.empty():
            try:
                payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.processor(payload)
            except Exception as exc:
                logger.warning(f"[MONITOR] Alert worker error: {exc}")
            finally:
                self.queue.task_done()

    def stop(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

class LiveMonitor:
    """
    Real-time face recognition monitor.
    Draws HUD and dynamically calculates match confidence based on active engine.
    """

    def __init__(self, face_engine, db_manager):
        self.engine      = face_engine
        self.db          = db_manager
        self.alert_log   = {}   # {criminal_id: last_alert_timestamp}
        self.fps_history = []
        self.detection_history = []   # Recent detections for HUD display
        self.max_history = 5
        self.camera_stream = None
        self.alert_worker = None
        self.engine_text = f"ENGINE: {RECOGNITION_ENGINE}"
        self.ctrl_text = "[Q/ESC] Quit  [S] Snapshot  [P] Pause"
        self.engine_text_width = cv2.getTextSize(
            self.engine_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )[0][0]
        self.ctrl_text_width = cv2.getTextSize(
            self.ctrl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )[0][0]

    # ── Main Loop ─────────────────────────────────────────────────────────────
    def run(self, camera_index=CAMERA_INDEX):
        if not self.engine.model_loaded:
            print("\n[MONITOR] ⚠ No trained model found.")
            print("[MONITOR]   → Enroll faces first (Option 1 in menu), then train (Option 2).\n")
            print("[MONITOR]   Running in DETECTION-ONLY mode (faces detected but not recognized).\n")

        camera_cls = LatestFrameCamera if ASYNC_CAMERA_CAPTURE else DirectCamera
        self.camera_stream = camera_cls(camera_index).start()
        if ASYNC_ALERT_PROCESSING:
            self.alert_worker = AlertWorker(self._process_alert_payload, ALERT_WORKER_QUEUE_SIZE).start()

        # Camera warmup (stabilization)
        print("\n[MONITOR] Live feed starting...")
        print("[MONITOR] Warming up camera...")
        warmup_count = 0
        while warmup_count < WARMUP_FRAMES:
            if ASYNC_CAMERA_CAPTURE:
                current_count = self.camera_stream.frame_count()
                if current_count > warmup_count:
                    warmup_count = current_count
                else:
                    time.sleep(0.01)
            else:
                ret, _ = self.camera_stream.read()
                if ret:
                    warmup_count += 1
        print(f"[MONITOR] ✓ Camera ready ({warmup_count} frames discarded)")
        
        print("[MONITOR] Controls:")
        print("  Q / ESC → Quit")
        print("  S       → Save snapshot manually")
        print("  P       → Pause / Resume")
        print("  F       → Show FPS stats\n")

        paused = False
        frame_num = 0
        prev_time = time.time()
        fps_history = []
        show_fps_stats = False

        try:
            while True:
                ret, frame = self.camera_stream.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame, 1)

                if not paused:
                    frame_num += 1
                    current_time = time.time()
                    fps = 1.0 / max(current_time - prev_time, 0.001)
                    prev_time = current_time
                    
                    fps_history.append(fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = sum(fps_history) / len(fps_history)

                    # Recognition
                    results, _ = self.engine.recognize_all_faces(frame)

                    for r in results:
                        self._draw_face_box(frame, r)
                        if r["is_known"]:
                            self._handle_alert(r, frame)

                    # HUD
                    self._draw_hud(frame, avg_fps, len(results), frame_num)
                    
                    # Optional: FPS stats overlay
                    if show_fps_stats and len(fps_history) > 0:
                        stats_y = 80
                        cv2.putText(frame, f"FPS Stats:", (10, stats_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
                        cv2.putText(frame, f"  Current: {fps:.1f}", (10, stats_y + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
                        cv2.putText(frame, f"  Average: {avg_fps:.1f}", (10, stats_y + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
                        cv2.putText(frame, f"  Min: {min(fps_history):.1f}", (10, stats_y + 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
                        cv2.putText(frame, f"  Max: {max(fps_history):.1f}", (10, stats_y + 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

                else:
                    self._draw_paused_overlay(frame)

                cv2.imshow("AI Face Recognition System — BSCS Project", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord('s') or key == ord('S'):
                    self._save_snapshot(frame, tag="manual")
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
                    print(f"[MONITOR] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('f') or key == ord('F'):
                    show_fps_stats = not show_fps_stats
                    print(f"[MONITOR] FPS stats: {'ON' if show_fps_stats else 'OFF'}")
        finally:
            if self.camera_stream is not None:
                self.camera_stream.stop()
                self.camera_stream = None

            if self.alert_worker is not None:
                self.alert_worker.stop()
                self.alert_worker = None

            cv2.destroyAllWindows()
        
        # Final statistics
        if fps_history:
            print(f"\n[MONITOR] ═══════════════════════════════════════")
            print(f"[MONITOR]   Session Statistics")
            print(f"[MONITOR] ═══════════════════════════════════════")
            print(f"[MONITOR]   Frames: {frame_num}")
            print(f"[MONITOR]   Avg FPS: {sum(fps_history)/len(fps_history):.2f}")
            print(f"[MONITOR]   Min FPS: {min(fps_history):.2f}")
            print(f"[MONITOR]   Max FPS: {max(fps_history):.2f}")
            print(f"[MONITOR]   Duration: {frame_num / (sum(fps_history)/len(fps_history)):.1f}s")
            print(f"[MONITOR] ═══════════════════════════════════════\n")

    # ── Drawing Helpers ───────────────────────────────────────────────────────
    def _draw_face_box(self, frame, r):
        """Draw bounding box, name label, and dynamically calculated confidence bar."""
        x, y, w, h    = r["x"], r["y"], r["w"], r["h"]
        name          = r["name"]
        confidence    = r["confidence"]
        is_known      = r["is_known"]

        # Box color: RED for known criminal, GREEN for unknown
        box_color = COLOR_RED if is_known else COLOR_GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Corner accents (stylized box corners)
        corner_len = 15
        corner_t   = 3
        for (cx, cy) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = 1 if cx == x else -1
            dy = 1 if cy == y else -1
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), box_color, corner_t)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), box_color, corner_t)

        # ── HYBRID MATH FOR CONFIDENCE METERS ──
        if is_known:
            if RECOGNITION_ENGINE == "SFACE":
                match_pct = min(1.0, max(0.0, confidence))
            else:
                match_pct = max(0, 1.0 - confidence / RECOGNITION_CONFIDENCE_THRESHOLD)
        else:
            match_pct = 0.0

        label_text = f"{name}"
        if is_known:
            label_text += f"  [{match_pct * 100:.1f}% match]"
            criminal = r.get("criminal", {})
            if criminal:
                crime = criminal.get("crime_type", "")
                if crime:
                    label_text += f"  Crime: {crime}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = y - 10 if y > 40 else y + h + 20

        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 8, label_y + 4), box_color, -1)
        cv2.putText(frame, label_text, (x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 1)

        if is_known:
            meter_y    = y + h + 8
            bar_width  = int(w * match_pct)
            cv2.rectangle(frame, (x, meter_y), (x + w, meter_y + 5), (60, 60, 60), -1)
            meter_color = COLOR_GREEN if match_pct > 0.6 else COLOR_YELLOW if match_pct > 0.3 else COLOR_RED
            cv2.rectangle(frame, (x, meter_y), (x + bar_width, meter_y + 5), meter_color, -1)
            cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), COLOR_RED, 3)

    def _draw_hud(self, frame, fps, face_count, frame_num):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)

        cv2.putText(frame, "AI FACE RECOGNITION SYSTEM", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

        faces_text = f"Faces: {face_count}"
        cv2.putText(frame, faces_text, (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        cv2.putText(frame, self.engine_text, (220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, ts, (w - tw - 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        status  = "MODEL: ACTIVE" if self.engine.model_loaded else "MODEL: NOT TRAINED"
        s_color = COLOR_GREEN if self.engine.model_loaded else COLOR_RED
        (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, status, (w - sw - 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, s_color, 1)

        if self.detection_history:
            log_y = h - 10
            for log_entry in reversed(self.detection_history[-3:]):
                cv2.putText(frame, log_entry, (10, log_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_RED, 1)
                log_y -= 18

        cv2.putText(frame, self.ctrl_text, (w // 2 - self.ctrl_text_width // 2, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    def _draw_paused_overlay(self, frame):
        box_x = 40
        box_y = frame.shape[0] // 2 - 35
        box_w = frame.shape[1] - 80
        box_h = 70
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_BLACK, -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_YELLOW, 2)
        cv2.putText(frame, "PAUSED — Press P to resume", (80, frame.shape[0] // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

    # ── Alert / Logging ───────────────────────────────────────────────────────
    def _handle_alert(self, result, frame):
        criminal  = result.get("criminal")
        if not criminal:
            return

        criminal_id = criminal["id"]
        now         = time.time()

        last_alert = self.alert_log.get(criminal_id, 0)
        if now - last_alert < ALERT_COOLDOWN_SECONDS:
            return

        self.alert_log[criminal_id] = now

        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"⚠ ALERT [{ts}] {result['name']} detected (conf: {result['confidence']:.1f})"
        self.detection_history.append(msg)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        payload = {
            "criminal": criminal,
            "name": result["name"],
            "confidence": result["confidence"],
            "camera_id": f"webcam_{CAMERA_INDEX}",
            "snapshot_tag": f"criminal_{criminal_id}",
            "snapshot_frame": frame.copy() if SNAPSHOT_ON_DETECTION else None,
        }

        if self.alert_worker is not None and self.alert_worker.submit(payload):
            return

        self._process_alert_payload(payload)

    def _process_alert_payload(self, payload):
        snapshot_path = None

        if payload["snapshot_frame"] is not None:
            snapshot_path = self._save_snapshot(
                payload["snapshot_frame"],
                tag=payload["snapshot_tag"],
                announce=False
            )

        self.db.log_detection(
            criminal_id=payload["criminal"]["id"],
            detected_name=payload["name"],
            confidence=payload["confidence"],
            snapshot_path=snapshot_path,
            camera_id=payload["camera_id"]
        )
        self._print_alert(payload["criminal"], payload["confidence"])

    def _print_alert(self, criminal, confidence):
        print("\n" + "!" * 60)
        print(f"  ⚠  CRIMINAL DETECTED! ({RECOGNITION_ENGINE} ENGINE)")
        print(f"  Name      : {criminal['name']}")
        print(f"  Crime     : {criminal.get('crime_type', 'N/A')}")
        print(f"  Status    : {criminal.get('status', 'N/A')}")
        
        if RECOGNITION_ENGINE == "SFACE":
            print(f"  Confidence: {confidence:.2f} (Cosine Similarity: Higher = Better)")
        else:
            print(f"  Confidence: {confidence:.2f} (Euclidean Distance: Lower = Better)")
            
        print(f"  Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("!" * 60 + "\n")

    def _save_snapshot(self, frame, tag="snapshot", announce=True):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{ts}.jpg"
        path     = os.path.join(CAPTURED_DIR, filename)
        cv2.imwrite(path, frame)
        if announce:
            print(f"[MONITOR] Snapshot saved → {path}")
        return path