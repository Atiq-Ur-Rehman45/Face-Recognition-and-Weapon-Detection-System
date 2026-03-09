"""
==============================================================
  AI Face Recognition System — Central Configuration
  PRODUCTION GRADE: Optimized for stability
==============================================================
"""

import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
CRIMINAL_DB_DIR  = os.path.join(DATA_DIR, "criminal_db")
CAPTURED_DIR     = os.path.join(DATA_DIR, "captured_faces")
TRAINING_DIR     = os.path.join(DATA_DIR, "training_data")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join(DATA_DIR, "criminal_records.db")

# ── Model Paths ───────────────────────────────────────────────────────────────
LBPH_MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_face_model.xml")

# ── Cascade Classifier ────────────────────────────────────────────────────────
import cv2
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_PROFILE_CASCADE = cv2.data.haarcascades + "haarcascade_profileface.xml"  # NEW: For side profiles
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

# ── Face Detection Parameters — OPTIMIZED FOR STABILITY ───────────────────────
# ENROLLMENT (strict - only high quality faces)
ENROLL_SCALE_FACTOR   = 1.15
ENROLL_MIN_NEIGHBORS  = 7
ENROLL_MIN_SIZE       = (80, 80)

# LIVE MONITORING (lenient - catch all angles, never lose tracking)
DETECTION_SCALE_FACTOR   = 1.05      # Lower = more sensitive (was 1.15)
DETECTION_MIN_NEIGHBORS  = 3         # Lower = more detections (was 7)
DETECTION_MIN_SIZE       = (50, 50)  # Smaller = detect farther away (was 80x80)

# ── Recognition Thresholds — TUNED FOR STABILITY ──────────────────────────────
RECOGNITION_CONFIDENCE_THRESHOLD = 130   # Higher = more lenient (was 70)
UNKNOWN_LABEL = "Unknown"

# Recognition stability
RECOGNITION_SMOOTHING_FRAMES = 5     # Average over last 7 frames
RECOGNITION_CONFIDENCE_ALPHA = 0.3   # Smoothing factor (lower = more stable)

# ── Image Capture Settings ────────────────────────────────────────────────────
ENROLL_FRAME_COUNT    = 100   # iPhone style continuous rotation
ENROLL_CAPTURE_DELAY  = 0.15
ENROLL_COUNTDOWN      = 3

# Camera settings
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480
CAMERA_INDEX          = 0

# ── Enrollment Guide Strategy ─────────────────────────────────────────────────
ENROLLMENT_PROMPTS = [
    "Look STRAIGHT at camera",
    "Turn head slightly to your LEFT",
    "Turn head slightly to your RIGHT", 
    "Tilt head UP slightly",
    "Tilt head DOWN slightly",
    "Look STRAIGHT again",
    "Smile naturally",
    "Neutral expression",
    "Move closer to camera",
    "Move back slightly",
    "Turn LEFT again",
    "Turn RIGHT again",
    "Look STRAIGHT - final shots",
]

# Quality thresholds
ENABLE_BLUR_DETECTION = True
BLUR_THRESHOLD        = 80.0   # Lowered for more acceptance (was 100)

# ── UI Colors (BGR format for OpenCV) ─────────────────────────────────────────
COLOR_GREEN    = (0,   220,  0  )
COLOR_RED      = (0,   0,   220 )
COLOR_YELLOW   = (0,   200, 200 )
COLOR_BLUE     = (220, 100,  0  )
COLOR_WHITE    = (255, 255, 255 )
COLOR_BLACK    = (0,   0,   0   )
COLOR_ORANGE   = (0,   165, 255 )
COLOR_PURPLE   = (255, 0,   255 )

# ── Alert Settings ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30
SNAPSHOT_ON_DETECTION  = True

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ── Ensure all directories exist ──────────────────────────────────────────────
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)