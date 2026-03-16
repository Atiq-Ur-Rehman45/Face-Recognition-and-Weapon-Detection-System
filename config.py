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

# ══════════════════════════════════════════════════════════════════════════════
# 🔥 ENGINE SELECTION — CHANGE THIS TO SWITCH MODES
# ══════════════════════════════════════════════════════════════════════════════
RECOGNITION_ENGINE = "SFACE"     # Options: "SFACE" or "LBPH"
                                 # SFACE = Modern (95%+ accuracy)
                                 # LBPH  = Classic (75-85% accuracy)

# ══════════════════════════════════════════════════════════════════════════════
# SFACE (Deep Learning) Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Model Paths ───────────────────────────────────────────────────────────────
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "sface.onnx")
SFACE_DB_PATH    = os.path.join(MODELS_DIR, "sface_embeddings.pkl")

# ── YuNet Detection Parameters ────────────────────────────────────────────────
YUNET_INPUT_SIZE = (320, 320)       # Input size for YuNet (don't change)
YUNET_SCORE_THRESHOLD = 0.6         # Detection confidence (0.0-1.0)
                                     # 0.5 = more detections, 0.7 = fewer false positives
YUNET_NMS_THRESHOLD = 0.3           # Non-maximum suppression (0.0-1.0)
YUNET_TOP_K = 5000                  # Max detections to consider

# ── SFace Recognition Parameters ──────────────────────────────────────────────
SFACE_OPERATING_PROFILE = "BALANCED"   # BALANCED, SECURITY_FIRST, RECALL_FIRST

if SFACE_OPERATING_PROFILE == "SECURITY_FIRST":
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.60 # Very strict, lowest false positives
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.57
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.08
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.06
    SFACE_CONSENSUS_FRAMES = 4           # Need more agreeing frames before confirmed known
elif SFACE_OPERATING_PROFILE == "RECALL_FIRST":
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.50 # Lenient, higher recall
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.47
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.04
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.03
    SFACE_CONSENSUS_FRAMES = 2
else:
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.53 # Balanced default after field tuning
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.50
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.045
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.03
    SFACE_CONSENSUS_FRAMES = 2

SFACE_MATCH_THRESHOLD = SFACE_MATCH_THRESHOLD_ACQUIRE  # Compatibility alias
SFACE_MARGIN_THRESHOLD = SFACE_MARGIN_THRESHOLD_ACQUIRE
SFACE_CONSENSUS_WINDOW = 6              # Track this many recent frames per face track
SFACE_KNOWN_HOLD_FRAMES = 4             # Keep confirmed identity for N brief unstable frames
SFACE_RECOG_MIN_FACE_AREA_RATIO = 0.035 # Reject tiny faces during recognition
SFACE_RECOG_BLUR_THRESHOLD = 70.0       # Reject blurry face ROI during recognition

# Multi-face stabilization cache
SFACE_CACHE_REUSE_FRAMES = 2        # Reuse last recognition for up to N frames if same face box persists
SFACE_CACHE_TTL_SECONDS = 0.7       # Drop stale cached tracks quickly to avoid ghost identities
SFACE_CACHE_IOU_THRESHOLD = 0.65    # Strict overlap needed to reuse a cached identity
SFACE_CACHE_MAX_TRACKS = 64         # Hard cap to keep cache operations bounded

# ══════════════════════════════════════════════════════════════════════════════
# LBPH (Classic) Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Model Paths ───────────────────────────────────────────────────────────────
LBPH_MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_face_model.xml")

# ── Cascade Classifier Paths ──────────────────────────────────────────────────
import cv2
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_PROFILE_CASCADE = cv2.data.haarcascades + "haarcascade_profileface.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

# ── LBPH Detection Parameters (Live Monitoring) ───────────────────────────────
DETECTION_SCALE_FACTOR   = 1.05     # Lower = more sensitive (1.05-1.3)
DETECTION_MIN_NEIGHBORS  = 3        # Lower = more detections (3-7)
DETECTION_MIN_SIZE       = (50, 50) # Minimum face size in pixels

# Enrollment Detection (Stricter for quality)
ENROLL_SCALE_FACTOR   = 1.15
ENROLL_MIN_NEIGHBORS  = 7
ENROLL_MIN_SIZE       = (80, 80)

# ── LBPH Recognition Parameters ───────────────────────────────────────────────
RECOGNITION_CONFIDENCE_THRESHOLD = 100   # LBPH distance threshold
                                          # Lower = stricter (50-70)
                                          # Higher = more lenient (80-120)
                                          # Recommended: 100

# Recognition Stability
RECOGNITION_SMOOTHING_FRAMES = 5    # Average over last N frames
RECOGNITION_CONFIDENCE_ALPHA = 0.3  # Smoothing factor (0.0-1.0)

# ── Common Settings ───────────────────────────────────────────────────────────
UNKNOWN_LABEL = "Unknown"

# ══════════════════════════════════════════════════════════════════════════════
# Camera & Enrollment Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Camera Configuration ──────────────────────────────────────────────────────
FRAME_WIDTH           = 640         # Optimized for CPU Deep Learning
FRAME_HEIGHT          = 480         # Drastically increases FPS
CAMERA_INDEX          = 0           # 0 = default webcam, 1 = external

# Camera optimization flags
CAMERA_FPS_TARGET     = 30          # Request 30 FPS from camera
CAMERA_AUTOFOCUS      = False       # Disable autofocus for stability
CAMERA_BUFFER_SIZE    = 1           # Reduce lag (1-3)
WARMUP_FRAMES         = 10          # Discard first N frames (camera settling)
ASYNC_CAMERA_CAPTURE  = True        # Read camera frames on a background thread

# ── Enrollment Settings ───────────────────────────────────────────────────────
# CRITICAL: SFace needs fewer images than LBPH!
if RECOGNITION_ENGINE == "SFACE":
    ENROLL_FRAME_COUNT = 20         # SFace: 20 images improves identity separation
else:
    ENROLL_FRAME_COUNT = 30         # LBPH: needs 30 for good accuracy

ENROLL_CAPTURE_DELAY  = 0.4         # Seconds between auto-captures
ENROLL_COUNTDOWN      = 3           # Countdown before enrollment starts
ENROLL_STAGE_PAUSE_SECONDS = 2.5    # Pause between angle stages
ENROLL_SECURITY_PROFILE = "STRICT"  # STRICT or BALANCED

# Enrollment quality/pose assistance
ENROLL_FACE_MIN_AREA_RATIO = 0.04   # Minimum face area ratio in frame for reliable enrollment
ENROLL_POSE_RELAX_AFTER_SECONDS = 8.0  # After this, pose thresholds are relaxed to reduce user friction
ENROLL_SAVE_FACE_CROPS = True       # Save aligned crop-like face ROI alongside full frame for quick QA
ENROLL_STAGE_TIMEOUT_SECONDS = 20.0  # Warn operator if one stage takes too long

# Basic anti-spoof/liveness controls during enrollment
ENROLL_LIVENESS_CHALLENGE_ENABLED = True   # Require pose transition from front baseline
ENROLL_REQUIRED_POSE_DELTA = 0.06          # Minimum yaw/pitch shift to pass movement challenge
ENROLL_MIN_STABLE_FRAMES = 3               # Require N stable valid frames before each capture

# Balanced profile softens strict defaults without changing code paths
if ENROLL_SECURITY_PROFILE == "BALANCED":
    ENROLL_REQUIRED_POSE_DELTA = 0.04
    ENROLL_MIN_STABLE_FRAMES = 2
    ENROLL_FACE_MIN_AREA_RATIO = 0.035

# ── Multi-Angle Enrollment Strategy ───────────────────────────────────────────
if RECOGNITION_ENGINE == "SFACE":
    # SFace: Balanced quality and diversity for stronger embeddings
    ENROLLMENT_STRATEGY = [
        {"angle": "FRONT", "count": 8, "instruction": "Look STRAIGHT at camera"},
        {"angle": "LEFT",  "count": 4, "instruction": "Turn head to the LEFT"},
        {"angle": "RIGHT", "count": 4, "instruction": "Turn head to the RIGHT"},
        {"angle": "UP",    "count": 2, "instruction": "Tilt head UP slightly"},
        {"angle": "DOWN",  "count": 2, "instruction": "Tilt head DOWN slightly"},
    ]
else:
    # LBPH: More images for better training
    ENROLLMENT_STRATEGY = [
        {"angle": "FRONT", "count": 10, "instruction": "Look STRAIGHT at camera - HOLD STILL"},
        {"angle": "LEFT",  "count": 6,  "instruction": "Turn head to the LEFT - HOLD"},
        {"angle": "RIGHT", "count": 6,  "instruction": "Turn head to the RIGHT - HOLD"},
        {"angle": "UP",    "count": 4,  "instruction": "Tilt head UP slightly - HOLD"},
        {"angle": "DOWN",  "count": 4,  "instruction": "Tilt head DOWN slightly - HOLD"},
    ]

# ── Image Quality Settings ────────────────────────────────────────────────────
ENABLE_BLUR_DETECTION = True        # Reject blurry frames during enrollment
BLUR_THRESHOLD        = 80.0        # Laplacian variance threshold

# ══════════════════════════════════════════════════════════════════════════════
# UI & Display Settings
# ══════════════════════════════════════════════════════════════════════════════

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
ALERT_COOLDOWN_SECONDS = 30         # Min seconds between repeated alerts
SNAPSHOT_ON_DETECTION  = True       # Save snapshot when criminal detected
ASYNC_ALERT_PROCESSING = True       # Offload alert I/O away from the live loop
ALERT_WORKER_QUEUE_SIZE = 32        # Bounded queue to avoid runaway alert lag

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ══════════════════════════════════════════════════════════════════════════════
# Auto-Download Models (SFace Only)
# ══════════════════════════════════════════════════════════════════════════════

def download_models_if_needed():
    """Auto-download YuNet and SFace models if not present."""
    if RECOGNITION_ENGINE != "SFACE":
        return True
    
    import urllib.request
    models_to_download = []
    
    if not os.path.exists(YUNET_MODEL_PATH):
        models_to_download.append({
            'name': 'YuNet (Face Detector)',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            'path': YUNET_MODEL_PATH,
            'size': '~2.8 MB'
        })
    
    if not os.path.exists(SFACE_MODEL_PATH):
        models_to_download.append({
            'name': 'SFace (Face Recognizer)',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
            'path': SFACE_MODEL_PATH,
            'size': '~10 MB'
        })
    
    if models_to_download:
        print("\n" + "═" * 60)
        print("  DEEP LEARNING MODELS REQUIRED")
        print("═" * 60)
        print("  Downloading models (one-time setup)...\n")
        
        for model in models_to_download:
            print(f"  → {model['name']} ({model['size']})... ", end='', flush=True)
            try:
                urllib.request.urlretrieve(model['url'], model['path'])
                print("✓ Downloaded")
            except Exception as e:
                print(f"✗ FAILED\n  ERROR: {e}")
                return False
        
        print("\n  ✓ All models downloaded successfully!")
        print("═" * 60 + "\n")
    return True

# Ensure all directories exist
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Auto-download models if using SFace
if RECOGNITION_ENGINE == "SFACE":
    _models_ready = download_models_if_needed()

print("\n" + "═" * 60)
print("  AI FACE RECOGNITION SYSTEM — Configuration Loaded")
print("═" * 60)
print(f"  Engine Mode        : {RECOGNITION_ENGINE}")
print(f"  Camera Resolution  : {FRAME_WIDTH}×{FRAME_HEIGHT}")
print(f"  Enrollment Images  : {ENROLL_FRAME_COUNT} per person")
print("═" * 60 + "\n")