"""
==============================================================
  AI Face Recognition System — Face Recognition Engine
  FIXED: Single stable detection + Better recognition
==============================================================
"""

import cv2
import numpy as np
import os
import time
import logging
from collections import deque, defaultdict
from config import (
    FACE_CASCADE_PATH, FACE_PROFILE_CASCADE, EYE_CASCADE_PATH, LBPH_MODEL_PATH,
    DETECTION_SCALE_FACTOR, DETECTION_MIN_NEIGHBORS, DETECTION_MIN_SIZE,
    ENROLL_SCALE_FACTOR, ENROLL_MIN_NEIGHBORS, ENROLL_MIN_SIZE,
    RECOGNITION_CONFIDENCE_THRESHOLD, UNKNOWN_LABEL,
    RECOGNITION_SMOOTHING_FRAMES, RECOGNITION_CONFIDENCE_ALPHA,
    ENABLE_BLUR_DETECTION, BLUR_THRESHOLD
)

logger = logging.getLogger(__name__)


class FaceEngine:
    """Production-grade face detection and recognition engine."""

    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.profile_cascade = cv2.CascadeClassifier(FACE_PROFILE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        if self.face_cascade.empty():
            raise RuntimeError(f"[ENGINE] Failed to load face cascade")

        # LBPH Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, 
            threshold=float(RECOGNITION_CONFIDENCE_THRESHOLD)
        )
        self.model_loaded = False
        self.label_map = {}
        self.confidence_thresh = RECOGNITION_CONFIDENCE_THRESHOLD

        # Recognition smoothing per face
        self.recognition_memory = defaultdict(lambda: {
            'label_votes': deque(maxlen=5),  # Reduced for faster response
            'last_seen': 0
        })

        self._load_model()
        logger.info("[ENGINE] FaceEngine initialized.")

    # ── Model Persistence ─────────────────────────────────────────────────────

    def _load_model(self):
        """Load pre-trained LBPH model."""
        if os.path.exists(LBPH_MODEL_PATH):
            try:
                self.recognizer.read(LBPH_MODEL_PATH)
                self.model_loaded = True
                logger.info(f"[ENGINE] Model loaded from {LBPH_MODEL_PATH}")
                print(f"[ENGINE] ✓ Trained model loaded.")
            except Exception as e:
                logger.warning(f"[ENGINE] Could not load model: {e}")
                self.model_loaded = False
        else:
            print("[ENGINE] No trained model found.")

    def save_model(self):
        """Save the trained LBPH model."""
        self.recognizer.save(LBPH_MODEL_PATH)
        print(f"[ENGINE] ✓ Model saved → {LBPH_MODEL_PATH}")
        logger.info(f"[ENGINE] Model saved")

    # ── IMPROVED Detection (Single Face Only) ────────────────────────────────

    def detect_single_best_face(self, gray, mode='live'):
        """
        Detect ONLY the best/largest face to avoid multiple overlapping boxes.
        Returns: [(x, y, w, h)] or []
        """
        if mode == 'enroll':
            scale = ENROLL_SCALE_FACTOR
            neighbors = ENROLL_MIN_NEIGHBORS
            min_size = ENROLL_MIN_SIZE
        else:
            scale = DETECTION_SCALE_FACTOR
            neighbors = DETECTION_MIN_NEIGHBORS
            min_size = DETECTION_MIN_SIZE

        # OPTIMIZATION: Downscale image for much faster cascade scanning
        SMALL_SCALE = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=SMALL_SCALE, fy=SMALL_SCALE)
        
        # Adjust min_size
        small_min_size = (int(min_size[0] * SMALL_SCALE), int(min_size[1] * SMALL_SCALE))

        # Detect frontal faces
        frontal = self.face_cascade.detectMultiScale(
            small_gray, scaleFactor=scale, minNeighbors=neighbors, minSize=small_min_size
        )
        
        # Profile faces
        profile = self.profile_cascade.detectMultiScale(
            small_gray, scaleFactor=scale, minNeighbors=max(3, neighbors - 2), minSize=small_min_size
        )
        
        # Profile flipped
        flipped_small_gray = cv2.flip(small_gray, 1)
        profile_flipped = self.profile_cascade.detectMultiScale(
            flipped_small_gray, scaleFactor=scale, minNeighbors=max(3, neighbors - 2), minSize=small_min_size
        )

        all_faces = []
        if len(frontal) > 0:
            for (x, y, w, h) in frontal:
                all_faces.append((int(x/SMALL_SCALE), int(y/SMALL_SCALE), int(w/SMALL_SCALE), int(h/SMALL_SCALE)))
        if len(profile) > 0:
            for (x, y, w, h) in profile:
                all_faces.append((int(x/SMALL_SCALE), int(y/SMALL_SCALE), int(w/SMALL_SCALE), int(h/SMALL_SCALE)))
        if len(profile_flipped) > 0:
            img_w = small_gray.shape[1]
            for (x, y, w, h) in profile_flipped:
                all_faces.append((int((img_w - x - w)/SMALL_SCALE), int(y/SMALL_SCALE), int(w/SMALL_SCALE), int(h/SMALL_SCALE)))

        if not all_faces:
            return []

        # ── Filter by size/area ───────────
        valid_faces = []
        for (x, y, w, h) in all_faces:
            area = w * h
            if area < 3000:
                continue
                
            valid_faces.append((x, y, w, h, area))

        if not valid_faces:
            return []

        # ── Return ONLY the largest face ─────────────────────────────────────
        # Sort by area (largest first)
        valid_faces.sort(key=lambda f: f[4], reverse=True)
        
        # Take only the biggest one
        x, y, w, h, area = valid_faces[0]
        
        return [(x, y, w, h)]

    def detect_faces(self, frame, equalize=True, mode='live'):
        """
        Main detection method - returns single best face.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if equalize:
            gray = cv2.equalizeHist(gray)

        faces = self.detect_single_best_face(gray, mode)
        return faces, gray

    # ── Validation (for enrollment quality check) ────────────────────────────

    def validate_face(self, gray, x, y, w, h):
        """Face quality scoring for enrollment."""
        face_roi = gray[y:y+h, x:x+w]
        score = 0.0
        
        # Eye detection (50 points)
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
        )
        if len(eyes) >= 2:
            score += 50
        elif len(eyes) == 1:
            score += 25
        
        # Aspect ratio (25 points)
        aspect_ratio = w / float(h)
        if 0.75 < aspect_ratio < 1.3:
            score += 25
        elif 0.65 < aspect_ratio < 1.5:
            score += 15
        
        # Size (15 points)
        area = w * h
        if 8000 < area < 150000:
            score += 15
        elif 5000 < area < 200000:
            score += 8
        
        # Sharpness (10 points)
        blur_score = self.estimate_blur(face_roi)
        if blur_score > 150:
            score += 10
        elif blur_score > 80:
            score += 5
        
        is_valid = score >= 50
        return is_valid, score

    def preprocess_face(self, gray, x, y, w, h, target_size=(100, 100)):
        """Extract and preprocess face ROI."""
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        face_roi = gray[y1:y2, x1:x2]
        
        # CLAHE for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)
        
        face_roi = cv2.resize(face_roi, target_size)
        return face_roi

    def estimate_blur(self, gray_image):
        """Estimate blur using Laplacian variance."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()

    # ── IMPROVED Recognition (Voting System) ─────────────────────────────────

    def recognize_face_with_voting(self, gray, x, y, w, h):
        """
        Recognize face with voting to reduce flickering.
        Uses majority vote over last 5 frames.
        """
        if not self.model_loaded:
            return -1, 999.0, UNKNOWN_LABEL

        face_roi = self.preprocess_face(gray, x, y, w, h)

        try:
            label, confidence = self.recognizer.predict(face_roi)
        except cv2.error as e:
            logger.warning(f"[ENGINE] Predict error: {e}")
            return -1, 999.0, UNKNOWN_LABEL

        # Track the main face stably by treating the largest detection as a single target 
        # (This is valid since `detect_single_best_face` limits us to the single largest bounding box)
        face_id = "primary_target"
        
        memory = self.recognition_memory[face_id]
        memory['last_seen'] = time.time()
        
        # Only accept if confidence is good enough
        if confidence <= self.confidence_thresh:
            memory['label_votes'].append(label)
        else:
            memory['label_votes'].append(-1)  # Unknown
        
        # Voting: most common label
        if len(memory['label_votes']) >= 3:
            from collections import Counter
            votes = Counter(memory['label_votes'])
            voted_label = votes.most_common(1)[0][0]
            
            if voted_label == -1:
                return -1, confidence, UNKNOWN_LABEL
            
            # Use voted label
            criminal = self.label_map.get(voted_label)
            name = criminal["name"] if criminal else f"Label_{voted_label}"
            return voted_label, confidence, name
        
        # Not enough votes yet
        if confidence <= self.confidence_thresh:
            criminal = self.label_map.get(label)
            name = criminal["name"] if criminal else f"Label_{label}"
            return label, confidence, name
        
        return -1, confidence, UNKNOWN_LABEL

    def recognize_all_faces(self, frame):
        """Detect and recognize all faces (optimized for single face)."""
        faces, gray = self.detect_faces(frame, mode='live')
        
        results = []
        for (x, y, w, h) in faces:
            label, confidence, name = self.recognize_face_with_voting(gray, x, y, w, h)
            criminal = self.label_map.get(label) if label != -1 else None
            
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label,
                "confidence": confidence,
                "name": name,
                "criminal": criminal,
                "is_known": name != UNKNOWN_LABEL
            })

        return results, gray

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, training_data):
        """Train the LBPH model."""
        if not training_data:
            print("[ENGINE] No training data provided.")
            return False

        images = [item[0] for item in training_data]
        labels = np.array([item[1] for item in training_data], dtype=np.int32)

        print(f"[ENGINE] Training on {len(images)} samples, {len(set(labels))} persons...")
        self.recognizer.train(images, labels)
        self.model_loaded = True
        self.save_model()
        print(f"[ENGINE] ✓ Training complete!")
        return True

    def update_training(self, new_data):
        """Incrementally update the model."""
        if not self.model_loaded:
            return self.train(new_data)

        images = [item[0] for item in new_data]
        labels = np.array([item[1] for item in new_data], dtype=np.int32)
        self.recognizer.update(images, labels)
        self.save_model()
        print(f"[ENGINE] ✓ Model updated with {len(images)} samples.")
        return True

    def update_label_map(self, label_map):
        """Update internal label mapping."""
        self.label_map = label_map
        logger.info(f"[ENGINE] Label map updated: {len(label_map)} entries")

    # ── Professional Enrollment ───────────────────────────────────────────────

    def collect_face_samples(self, camera_index=0, target_count=30, label=0, save_dir=None):
        """Professional multi-angle enrollment."""
        from config import (
            FRAME_WIDTH, FRAME_HEIGHT,
            COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE, 
            COLOR_PURPLE, COLOR_RED, COLOR_BLUE
        )

        CAPTURE_STRATEGY = [
            {"count": target_count, "instruction": "Slowly rotate your head left, right, up, and down"}
        ]
        
        total_target = sum(s["count"] for s in CAPTURE_STRATEGY)

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise RuntimeError(f"[ENGINE] Cannot open camera {camera_index}")

        collected = []
        last_capture = 0
        capture_delay = 0.15  # Much faster continuous capture
        quality_threshold = 40  # Lenient threshold to catch side angles naturally

        print(f"\n[ENGINE] ═══════════════════════════════════════════")
        print(f"[ENGINE]   IPHONE-STYLE CONTINUOUS ENROLLMENT")
        print(f"[ENGINE] ═══════════════════════════════════════════")
        print(f"[ENGINE] Target: {total_target} total images capturing all angles\n")
        
        input("[ENGINE] Press ENTER to begin...")

        # Countdown
        countdown_start = time.time()
        while time.time() - countdown_start < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            remaining = 3 - int(time.time() - countdown_start)
            cv2.putText(display, str(remaining), 
                       (FRAME_WIDTH//2 - 60, FRAME_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 6.0, COLOR_ORANGE, 12)
            
            cv2.imshow("Enrollment", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                return []

        # Main capture
        total_collected = 0
        
        while total_collected < total_target:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            now = time.time()
            
            instruction = CAPTURE_STRATEGY[0]["instruction"]

            # Detect with validation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            raw_faces = self.detect_single_best_face(gray, mode='enroll')
            
            best_face = None
            best_quality = 0
            
            for (x, y, w, h) in raw_faces:
                is_valid, quality = self.validate_face(gray, x, y, w, h)
                if is_valid and quality > best_quality:
                    best_quality = quality
                    best_face = (x, y, w, h)

            # UI
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 120), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
            
            cv2.putText(display, "CONTINUOUS CAPTURE", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PURPLE, 2)
            cv2.putText(display, instruction, 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            cv2.putText(display, f"Total: {total_collected}/{total_target}", 
                       (FRAME_WIDTH - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            if best_face:
                x, y, w, h = best_face
                
                if best_quality >= 75:
                    box_color = COLOR_GREEN
                    quality_text = "EXCELLENT"
                elif best_quality >= 60:
                    box_color = COLOR_BLUE
                    quality_text = "GOOD"
                elif best_quality >= 40:
                    box_color = COLOR_YELLOW
                    quality_text = "OK"
                else:
                    box_color = COLOR_RED
                    quality_text = "LOW"
                
                cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 3)
                cv2.putText(display, f"{quality_text} ({best_quality:.0f}%)", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # Capture
                if now - last_capture >= capture_delay:
                    if best_quality >= quality_threshold:
                        face_img = self.preprocess_face(gray, x, y, w, h)
                        collected.append((face_img, label))
                        total_collected += 1
                        last_capture = now
                        
                        if save_dir:
                            img_path = os.path.join(save_dir, 
                                f"continuous_{total_collected:03d}_q{best_quality:.0f}.jpg")
                            cv2.imwrite(img_path, face_img)
                        
                        print(f"  ✓ Captured [{total_collected}/{total_target}] Q:{best_quality:.0f}%")
            else:
                cv2.putText(display, "⚠ NO FACE DETECTED", 
                           (FRAME_WIDTH//2 - 150, FRAME_HEIGHT - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

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

        print(f"\n[ENGINE] ✓ Enrollment Complete: {len(collected)} continuous images\n")
        return collected