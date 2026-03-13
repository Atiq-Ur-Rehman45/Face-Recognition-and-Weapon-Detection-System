
import cv2
import numpy as np
import pickle
import os
import time
import logging
from collections import defaultdict, deque

from config import *

logger = logging.getLogger(__name__)

class FaceEngine:
    """
    Production-grade hybrid engine supporting both:
    - SFace (YuNet + SFace DNN) - Modern, accurate
    - LBPH (Haar + LBPH) - Legacy, fast
    """

    def __init__(self):
        self.engine_type = RECOGNITION_ENGINE
        self.label_map = {}
        self.model_loaded = False
        self._sface_input_size = None
        self._sface_labels = np.empty((0,), dtype=np.int32)
        self._sface_embedding_matrix = np.empty((0, 0), dtype=np.float32)
        self._sface_frame_index = 0
        self._sface_track_cache = []
        
        print(f"\n[ENGINE] ═══════════════════════════════════════")
        print(f"[ENGINE] Booting: {self.engine_type} Recognition Engine")
        print(f"[ENGINE] ═══════════════════════════════════════")
        
        if self.engine_type == "SFACE":
            self._init_sface()
        else:
            self._init_lbph()
        
        print(f"[ENGINE] ✓ Engine ready\n")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _init_sface(self):
        """Initialize Deep Learning YuNet + SFace."""
        try:
            self.detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH, "", YUNET_INPUT_SIZE,
                score_threshold=YUNET_SCORE_THRESHOLD,
                nms_threshold=YUNET_NMS_THRESHOLD, 
                top_k=YUNET_TOP_K
            )
            self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
            self.embeddings_db = {}
            self._set_sface_input_size((FRAME_WIDTH, FRAME_HEIGHT))
            
            if os.path.exists(SFACE_DB_PATH):
                with open(SFACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings_db = data.get('embeddings', {})
                    self.model_loaded = len(self.embeddings_db) > 0
                    self._refresh_sface_index()
                    print(f"[ENGINE] ✓ Loaded {len(self.embeddings_db)} SFace profiles")
            else:
                self._refresh_sface_index()
                print("[ENGINE] No SFace database found (will create on first enrollment)")
                
        except Exception as e:
            print(f"[ENGINE] ✗ SFace initialization failed: {e}")
            print("[ENGINE] → Falling back to LBPH mode")
            self.engine_type = "LBPH"
            self._init_lbph()

    def _set_sface_input_size(self, size):
        """Update the detector input size only when dimensions change."""
        if self._sface_input_size != size:
            self.detector.setInputSize(size)
            self._sface_input_size = size

    def _normalize_sface_embedding(self, feature):
        """Normalize embeddings so vectorized dot products match cosine similarity."""
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _refresh_sface_index(self):
        """Build a contiguous embedding matrix for fast full-database matching."""
        labels = []
        vectors = []

        for label, stored_feature in self.embeddings_db.items():
            labels.append(int(label))
            vectors.append(self._normalize_sface_embedding(stored_feature))

        if vectors:
            self._sface_labels = np.asarray(labels, dtype=np.int32)
            self._sface_embedding_matrix = np.vstack(vectors).astype(np.float32)
        else:
            self._sface_labels = np.empty((0,), dtype=np.int32)
            self._sface_embedding_matrix = np.empty((0, 0), dtype=np.float32)


    def _bbox_iou(self, box_a, box_b):
        """Compute IoU between two (x, y, w, h) boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        a_x2 = ax + aw
        a_y2 = ay + ah
        b_x2 = bx + bw
        b_y2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(a_x2, b_x2)
        iy2 = min(a_y2, b_y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return inter / union

    def _prune_sface_cache(self, now_ts):
        """Keep only fresh tracks and cap cache size."""
        self._sface_track_cache = [
            track for track in self._sface_track_cache
            if now_ts - track["last_seen_ts"] <= SFACE_CACHE_TTL_SECONDS
        ]

        if len(self._sface_track_cache) > SFACE_CACHE_MAX_TRACKS:
            self._sface_track_cache.sort(key=lambda t: t["last_seen_ts"], reverse=True)
            self._sface_track_cache = self._sface_track_cache[:SFACE_CACHE_MAX_TRACKS]

    def _find_cache_track(self, box):
        """Find the best cached track for a detected box."""
        best_idx = -1
        best_iou = 0.0

        for idx, track in enumerate(self._sface_track_cache):
            iou = self._bbox_iou(box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= SFACE_CACHE_IOU_THRESHOLD:
            return best_idx
        return -1

    def _run_sface_match(self, frame, face):
        """Compute recognition for a single detected face."""
        best_match_label = -1
        best_score = 0.0
        name = UNKNOWN_LABEL

        if self.model_loaded and self._sface_embedding_matrix.size > 0:
            try:
                aligned_face = self.recognizer.alignCrop(frame, face)
                feature = self._normalize_sface_embedding(
                    self.recognizer.feature(aligned_face)
                )
                scores = self._sface_embedding_matrix @ feature

                if scores.size > 0:
                    best_index = int(np.argmax(scores))
                    best_score = float(scores[best_index])
                    best_match_label = int(self._sface_labels[best_index])

                if best_score > SFACE_MATCH_THRESHOLD:
                    criminal = self.label_map.get(best_match_label)
                    name = criminal["name"] if criminal else f"ID_{best_match_label}"
                else:
                    best_match_label = -1

            except Exception as e:
                logger.warning(f"[ENGINE] SFace recognition error: {e}")

        return best_match_label, best_score, name

    def _init_lbph(self):
        """Initialize Legacy Haar Cascades + LBPH."""
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.profile_cascade = cv2.CascadeClassifier(FACE_PROFILE_CASCADE)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8,
            threshold=float(RECOGNITION_CONFIDENCE_THRESHOLD)
        )
        
        # Smoothing for stable recognition
        self.recognition_memory = defaultdict(lambda: {
            'label_votes': deque(maxlen=5),
            'last_seen': 0
        })
        
        if os.path.exists(LBPH_MODEL_PATH):
            try:
                self.recognizer.read(LBPH_MODEL_PATH)
                self.model_loaded = True
                print("[ENGINE] ✓ LBPH model loaded")
            except Exception as e:
                print(f"[ENGINE] Could not load LBPH model: {e}")
                self.model_loaded = False
        else:
            print("[ENGINE] No LBPH model found (will create on first training)")

    # ══════════════════════════════════════════════════════════════════════════
    # DETECTION & RECOGNITION (SFACE)
    # ══════════════════════════════════════════════════════════════════════════

    def _recognize_sface(self, frame):
        """SFace detection and recognition with proper landmark alignment."""
        self._sface_frame_index += 1
        height, width = frame.shape[:2]
        self._set_sface_input_size((width, height))
        
        _, faces = self.detector.detect(frame)
        
        results = []
        if faces is None:
            return results, frame

        now_ts = time.time()
        self._prune_sface_cache(now_ts)
        
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            box = (int(x), int(y), int(w), int(h))

            cache_idx = self._find_cache_track(box)
            reuse_cached = False

            if cache_idx != -1:
                cached = self._sface_track_cache[cache_idx]
                frame_gap = self._sface_frame_index - cached["last_recognition_frame"]
                if frame_gap <= SFACE_CACHE_REUSE_FRAMES:
                    best_match_label = cached["label"]
                    best_score = cached["confidence"]
                    name = cached["name"]
                    reuse_cached = True
                else:
                    best_match_label, best_score, name = self._run_sface_match(frame, face)
            else:
                best_match_label, best_score, name = self._run_sface_match(frame, face)

            criminal_data = self.label_map.get(best_match_label) if best_match_label != -1 else None

            if not reuse_cached:
                track_payload = {
                    "box": box,
                    "label": int(best_match_label),
                    "confidence": float(best_score),
                    "name": name,
                    "last_seen_ts": now_ts,
                    "last_recognition_frame": self._sface_frame_index,
                }
                if cache_idx != -1:
                    self._sface_track_cache[cache_idx] = track_payload
                else:
                    self._sface_track_cache.append(track_payload)
            elif cache_idx != -1:
                self._sface_track_cache[cache_idx]["box"] = box
                self._sface_track_cache[cache_idx]["last_seen_ts"] = now_ts
            
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": best_match_label,
                "confidence": float(best_score),
                "name": name,
                "criminal": criminal_data,
                "is_known": best_match_label != -1
            })
        
        return results, frame

    # ══════════════════════════════════════════════════════════════════════════
    # DETECTION & RECOGNITION (LBPH)
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_single_best_face_lbph(self, gray):
        """Detect only the largest face (prevents multiple overlapping boxes)."""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=DETECTION_MIN_NEIGHBORS,
            minSize=DETECTION_MIN_SIZE
        )
        
        if len(faces) == 0:
            return []
        
        # Filter and return only largest face
        img_h, img_w = gray.shape
        valid_faces = []
        
        for (x, y, w, h) in faces:
            # Ignore bottom detections (chest/body)
            if y + h > img_h * 0.85:
                continue
            # Ignore tiny faces
            if w * h < 3000:
                continue
            valid_faces.append((x, y, w, h, w * h))
        
        if not valid_faces:
            return []
        
        # Return only the largest
        valid_faces.sort(key=lambda f: f[4], reverse=True)
        return [valid_faces[0][:4]]

    def _recognize_lbph(self, frame):
        """LBPH detection and recognition with smoothing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self._detect_single_best_face_lbph(gray)
        
        results = []
        for (x, y, w, h) in faces:
            label, confidence, name = -1, 999.0, UNKNOWN_LABEL
            
            if self.model_loaded:
                # Preprocess face
                face_roi = gray[y:y+h, x:x+w]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                face_roi = clahe.apply(face_roi)
                face_roi = cv2.resize(face_roi, (100, 100))
                
                try:
                    pred_label, pred_conf = self.recognizer.predict(face_roi)
                    
                    # Voting smoothing
                    face_id = f"{x//50}_{y//50}"
                    memory = self.recognition_memory[face_id]
                    memory['last_seen'] = time.time()
                    
                    if pred_conf <= RECOGNITION_CONFIDENCE_THRESHOLD:
                        memory['label_votes'].append(pred_label)
                    else:
                        memory['label_votes'].append(-1)
                    
                    # Majority vote
                    if len(memory['label_votes']) >= 3:
                        from collections import Counter
                        votes = Counter(memory['label_votes'])
                        voted_label = votes.most_common(1)[0][0]
                        
                        if voted_label != -1:
                            label = voted_label
                            confidence = pred_conf
                            criminal = self.label_map.get(label)
                            name = criminal["name"] if criminal else f"ID_{label}"
                            
                except cv2.error as e:
                    logger.warning(f"[ENGINE] LBPH prediction error: {e}")
            
            criminal_data = self.label_map.get(label) if label != -1 else None
            
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label,
                "confidence": confidence,
                "name": name,
                "criminal": criminal_data,
                "is_known": label != -1
            })
        
        return results, frame

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN RECOGNITION ROUTER
    # ══════════════════════════════════════════════════════════════════════════

    def recognize_all_faces(self, frame):
        """Routes to the active recognition engine."""
        if self.engine_type == "SFACE":
            return self._recognize_sface(frame)
        else:
            return self._recognize_lbph(frame)

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING (SFACE)
    # ══════════════════════════════════════════════════════════════════════════

    def _train_sface(self, training_data):
        """
        Train SFace by creating embeddings for each person.
        training_data: list of (BGR_image, label) tuples
        """
        print(f"[ENGINE] Training SFace with {len(training_data)} images...")
        
        embeddings_by_label = defaultdict(list)
        
        for img, label in training_data:
            height, width = img.shape[:2]
            self._set_sface_input_size((width, height))
            
            _, faces = self.detector.detect(img)
            
            if faces is None or len(faces) == 0:
                logger.warning(f"[ENGINE] No face detected in training image for label {label}")
                continue
            
            # Use largest face
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces_sorted[0]
            
            # Extract embedding using proper landmark alignment
            try:
                aligned_face = self.recognizer.alignCrop(img, face)
                feature = self.recognizer.feature(aligned_face)
                embeddings_by_label[label].append(feature)
            except Exception as e:
                logger.warning(f"[ENGINE] Failed to extract feature for label {label}: {e}")
        
        if not embeddings_by_label:
            print("[ENGINE] ✗ No valid faces found in training data")
            return False
        
        # Average embeddings per person for maximum stability
        for label, features in embeddings_by_label.items():
            avg_embedding = np.mean(features, axis=0)
            self.embeddings_db[label] = avg_embedding
            print(f"  [+] Label {label}: {len(features)} samples → 1 optimized embedding")
        self._refresh_sface_index()
        
        # Save to disk
        with open(SFACE_DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings_db}, f)
        
        self.model_loaded = True
        print(f"[ENGINE] ✓ SFace training complete: {len(self.embeddings_db)} persons enrolled")
        print(f"[ENGINE] ✓ Database saved to {SFACE_DB_PATH}")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING (LBPH)
    # ══════════════════════════════════════════════════════════════════════════

    def _train_lbph(self, training_data):
        """
        Train LBPH recognizer.
        training_data: list of (grayscale_100x100_image, label) tuples
        """
        if not training_data:
            print("[ENGINE] No training data provided")
            return False
        
        images = [item[0] for item in training_data]
        labels = np.array([item[1] for item in training_data], dtype=np.int32)
        
        print(f"[ENGINE] Training LBPH on {len(images)} samples, {len(set(labels))} persons...")
        
        self.recognizer.train(images, labels)
        self.recognizer.save(LBPH_MODEL_PATH)
        self.model_loaded = True
        
        print(f"[ENGINE] ✓ LBPH training complete")
        print(f"[ENGINE] ✓ Model saved to {LBPH_MODEL_PATH}")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED TRAINING INTERFACE
    # ══════════════════════════════════════════════════════════════════════════

    def train(self, training_data):
        """Routes training to the active engine."""
        if self.engine_type == "SFACE":
            return self._train_sface(training_data)
        else:
            return self._train_lbph(training_data)

    def update_training(self, new_data):
        """Incrementally update model (LBPH only, SFace retrains entirely)."""
        if self.engine_type == "LBPH":
            if not self.model_loaded:
                return self.train(new_data)
            
            images = [item[0] for item in new_data]
            labels = np.array([item[1] for item in new_data], dtype=np.int32)
            self.recognizer.update(images, labels)
            self.recognizer.save(LBPH_MODEL_PATH)
            print(f"[ENGINE] ✓ LBPH model updated with {len(images)} samples")
            return True
        else:
            return self.train(new_data)

    # ══════════════════════════════════════════════════════════════════════════
    # ENROLLMENT (Face Sample Collection)
    # ══════════════════════════════════════════════════════════════════════════

    def _sface_pose_signature(self, face):
        """Estimate coarse yaw/pitch from YuNet 5-point landmarks."""
        if len(face) < 14:
            return None, None

        left_eye = np.array([face[4], face[5]], dtype=np.float32)
        right_eye = np.array([face[6], face[7]], dtype=np.float32)
        nose = np.array([face[8], face[9]], dtype=np.float32)
        mouth_left = np.array([face[10], face[11]], dtype=np.float32)
        mouth_right = np.array([face[12], face[13]], dtype=np.float32)

        eye_dx = right_eye[0] - left_eye[0]
        if abs(float(eye_dx)) < 1e-6:
            return None, None

        # Yaw proxy: nose position between eyes. Front is near 0.5.
        yaw = float((nose[0] - left_eye[0]) / eye_dx)

        eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
        mouth_mid_y = float((mouth_left[1] + mouth_right[1]) * 0.5)
        vertical_span = max(mouth_mid_y - eye_mid_y, 1e-6)

        # Pitch proxy: nose vertical offset between eye-line and mouth-line.
        pitch = float((nose[1] - eye_mid_y) / vertical_span)
        return yaw, pitch

    def _pose_matches_stage(self, angle, yaw, pitch, relaxed=False):
        """Validate coarse face orientation for the requested enrollment stage."""
        if yaw is None or pitch is None:
            return False

        angle = (angle or "").upper()
        if relaxed:
            if angle == "FRONT":
                return 0.38 <= yaw <= 0.62 and 0.34 <= pitch <= 0.76
            if angle == "LEFT":
                return yaw <= 0.49
            if angle == "RIGHT":
                return yaw >= 0.51
            if angle == "UP":
                return pitch <= 0.49
            if angle == "DOWN":
                return pitch >= 0.59

        if angle == "FRONT":
            return 0.42 <= yaw <= 0.58 and 0.40 <= pitch <= 0.70
        if angle == "LEFT":
            return yaw <= 0.46
        if angle == "RIGHT":
            return yaw >= 0.54
        if angle == "UP":
            return pitch <= 0.44
        if angle == "DOWN":
            return pitch >= 0.64
        return True

    def collect_face_samples(self, camera_index=0, target_count=None, label=0, save_dir=None):
        """
        Collect face samples from webcam with guided UI prompts and non-blocking pauses.
        """
        if target_count is None:
            target_count = ENROLL_FRAME_COUNT
        
        strategy = ENROLLMENT_STRATEGY
        total_target = sum(s["count"] for s in strategy)
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # ✨ NEW
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # ✨ NEW
    
        # ✨ NEW: Warmup
        print("[ENGINE] Warming up camera...")
        for _ in range(WARMUP_FRAMES):
            cap.read()
        
        if not cap.isOpened():
            raise RuntimeError(f"[ENGINE] Cannot open camera {camera_index}")
        
        collected = []
        current_angle_idx = 0
        angle_count = 0
        last_capture = 0
        capture_delay = ENROLL_CAPTURE_DELAY
        
        print(f"\n[ENGINE] ═══════════════════════════════════════════")
        print(f"[ENGINE]   ENROLLMENT MODE: {self.engine_type}")
        print(f"[ENGINE] ═══════════════════════════════════════════")
        print(f"[ENGINE] Target: {total_target} images across {len(strategy)} angles")
        print(f"[ENGINE] Press SPACE to capture, Q to cancel\n")
        
        input("[ENGINE] Press ENTER to begin...")
        
        # Initial countdown
        countdown_start = time.time()
        while time.time() - countdown_start < ENROLL_COUNTDOWN:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            remaining = ENROLL_COUNTDOWN - int(time.time() - countdown_start)
            cv2.putText(display, str(remaining), 
                       (FRAME_WIDTH//2 - 60, FRAME_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 6.0, COLOR_ORANGE, 12)
            
            cv2.imshow("Enrollment", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                return []
        
        # Main collection loop variables
        total_collected = 0
        pause_until = 0  # Controls the non-blocking UI pause
        stage_start_time = time.time()
        stage_metrics = [
            {
                "angle": stage.get("angle", f"STAGE_{idx + 1}"),
                "target": int(stage.get("count", 0)),
                "accepted": 0,
                "rejected_attempts": 0,
            }
            for idx, stage in enumerate(strategy)
        ]
        reject_reasons = defaultdict(int)
        
        while current_angle_idx < len(strategy):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            now = time.time()
            
            current_stage = strategy[current_angle_idx]
            current_angle = current_stage["angle"]
            target_for_angle = current_stage["count"]
            instruction = current_stage["instruction"]
            stage_elapsed = now - stage_start_time
            pose_relaxed = stage_elapsed >= ENROLL_POSE_RELAX_AFTER_SECONDS
            
            # ── NON-BLOCKING PAUSE UI ──────────────────────────────────
            if now < pause_until:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
                
                cv2.putText(display, "CHANGING POSE", 
                           (FRAME_WIDTH//2 - 150, FRAME_HEIGHT//2 - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 3)
                cv2.putText(display, instruction, 
                           (FRAME_WIDTH//2 - 200, FRAME_HEIGHT//2 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
                cv2.putText(display, f"{int(pause_until - now) + 1}...", 
                           (FRAME_WIDTH//2 - 30, FRAME_HEIGHT//2 + 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_ORANGE, 4)
                
                cv2.imshow("Enrollment", display)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
                continue 
            # ───────────────────────────────────────────────────────────
            
            # Detect face based on engine
            face_detected = False
            multiple_faces = False
            quality_ok = False
            pose_ok = False
            quality_message = ""
            reject_reason = None
            yaw = None
            pitch = None

            if self.engine_type == "SFACE":
                height, width = frame.shape[:2]
                self._set_sface_input_size((width, height))
                _, faces = self.detector.detect(frame)
                face_detected = faces is not None and len(faces) > 0

                if face_detected:
                    multiple_faces = len(faces) > 1
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    best_face = faces_sorted[0]
                    x, y, w, h = best_face[0:4].astype(int)

                    frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
                    face_area_ratio = (w * h) / frame_area

                    if multiple_faces:
                        quality_message = "ONE FACE ONLY"
                        reject_reason = "multiple_faces"
                    elif face_area_ratio < ENROLL_FACE_MIN_AREA_RATIO:
                        quality_message = "MOVE CLOSER"
                        reject_reason = "face_too_small"
                    else:
                        roi = frame[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)]
                        if roi.size == 0:
                            quality_message = "FACE OUT OF FRAME"
                            reject_reason = "face_out_of_frame"
                        else:
                            if ENABLE_BLUR_DETECTION:
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                blur_score = self.estimate_blur(roi_gray)
                                if blur_score < BLUR_THRESHOLD:
                                    quality_message = "HOLD STILL (BLURRY)"
                                    reject_reason = "blurry"
                                else:
                                    quality_ok = True
                            else:
                                quality_ok = True

                    if quality_ok:
                        yaw, pitch = self._sface_pose_signature(best_face)
                        pose_ok = self._pose_matches_stage(current_angle, yaw, pitch, relaxed=pose_relaxed)
                        if not pose_ok:
                            relax_tag = " (RELAXED)" if pose_relaxed else ""
                            quality_message = f"ADJUST TO {current_angle}{relax_tag}"
                            reject_reason = "pose_mismatch"
                    else:
                        pose_ok = False
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                lbph_faces = self._detect_single_best_face_lbph(gray)
                face_detected = len(lbph_faces) > 0
                if face_detected:
                    x, y, w, h = lbph_faces[0]
                    quality_ok = True
                    pose_ok = True
                else:
                    reject_reason = "no_face"
            
            # Standard UI Overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 120), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
            
            cv2.putText(display, f"ANGLE: {current_angle}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
            cv2.putText(display, instruction, 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            cv2.putText(display, f"{angle_count}/{target_for_angle}", 
                       (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
            cv2.putText(display, f"Total: {total_collected}/{total_target} | {self.engine_type}", 
                       (FRAME_WIDTH - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            if pose_relaxed and self.engine_type == "SFACE":
                cv2.putText(display, "POSE ASSIST: RELAXED", (FRAME_WIDTH - 260, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2)
            
            if face_detected:
                box_color = COLOR_GREEN if (quality_ok and pose_ok) else COLOR_YELLOW
                cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 3)

                if self.engine_type == "SFACE" and yaw is not None and pitch is not None:
                    pose_text = f"Yaw:{yaw:.2f} Pitch:{pitch:.2f}"
                    cv2.putText(display, pose_text,
                               (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
                
                # Auto-capture logic
                if quality_ok and pose_ok and (now - last_capture >= capture_delay):
                    collected.append(frame.copy())
                    angle_count += 1
                    total_collected += 1
                    last_capture = now
                    stage_metrics[current_angle_idx]["accepted"] += 1
                    
                    if save_dir:
                        img_path = os.path.join(save_dir, f"{current_angle}_{angle_count:02d}.jpg")
                        cv2.imwrite(img_path, frame)

                        if ENROLL_SAVE_FACE_CROPS:
                            crop_y1 = max(0, y)
                            crop_y2 = min(frame.shape[0], y + h)
                            crop_x1 = max(0, x)
                            crop_x2 = min(frame.shape[1], x + w)
                            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            if face_crop.size > 0:
                                crop_path = os.path.join(save_dir, f"{current_angle}_{angle_count:02d}_crop.jpg")
                                cv2.imwrite(crop_path, face_crop)
                    
                    print(f"  ✓ {current_angle} [{angle_count}/{target_for_angle}]")
                    
                    if angle_count >= target_for_angle:
                        current_angle_idx += 1
                        angle_count = 0
                        print(f"[ENGINE] ✓ {current_angle} complete!\n")
                        
                        if current_angle_idx < len(strategy):
                            pause_until = time.time() + ENROLL_STAGE_PAUSE_SECONDS
                            stage_start_time = time.time()
                elif quality_message:
                    cv2.putText(display, quality_message,
                               (FRAME_WIDTH//2 - 160, FRAME_HEIGHT - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_YELLOW, 2)

                if now - last_capture >= capture_delay and not (quality_ok and pose_ok):
                    stage_metrics[current_angle_idx]["rejected_attempts"] += 1
                    if reject_reason:
                        reject_reasons[reject_reason] += 1
            else:
                cv2.putText(display, "⚠ NO FACE DETECTED", 
                           (FRAME_WIDTH//2 - 150, FRAME_HEIGHT - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
                if now - last_capture >= capture_delay:
                    stage_metrics[current_angle_idx]["rejected_attempts"] += 1
                    reject_reasons["no_face"] += 1
            
            # Progress bar
            progress = total_collected / total_target
            bar_y = FRAME_HEIGHT - 25
            bar_w = int((FRAME_WIDTH - 40) * progress)
            cv2.rectangle(display, (20, bar_y), (FRAME_WIDTH - 20, bar_y + 15), (60, 60, 60), -1)
            cv2.rectangle(display, (20, bar_y), (20 + bar_w, bar_y + 15), COLOR_GREEN, -1)
            
            cv2.imshow("Enrollment", display)
            
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        print("[ENGINE] Enrollment stage summary:")
        for stage in stage_metrics:
            print(
                f"  - {stage['angle']}: accepted {stage['accepted']}/{stage['target']} | "
                f"rejected attempts {stage['rejected_attempts']}"
            )

        if reject_reasons:
            print("[ENGINE] Rejection reason summary:")
            for reason, count in sorted(reject_reasons.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {reason}: {count}")
        
        print(f"\n[ENGINE] ✓ Enrollment Complete: {len(collected)} images\n")
        return collected

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def update_label_map(self, label_map):
        """Update criminal name/info mapping."""
        self.label_map = label_map
        logger.info(f"[ENGINE] Label map updated: {len(label_map)} entries")

    def save_model(self):
        """Save current model to disk."""
        if self.engine_type == "SFACE":
            if self.embeddings_db:
                with open(SFACE_DB_PATH, 'wb') as f:
                    pickle.dump({'embeddings': self.embeddings_db}, f)
                print(f"[ENGINE] ✓ SFace database saved")
        else:
            if self.model_loaded:
                self.recognizer.save(LBPH_MODEL_PATH)
                print(f"[ENGINE] ✓ LBPH model saved")

    def estimate_blur(self, gray_image):
        """Laplacian variance blur detection."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()