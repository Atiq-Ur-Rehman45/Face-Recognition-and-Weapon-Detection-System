

## 📁 Project Structure

```
face_recognition_system/
│
├── main.py               ← Entry point (run this!)
├── config.py             ← All settings (camera index, thresholds, paths)
├── database.py           ← SQLite criminal database manager
├── face_engine.py        ← Detection (Haar Cascade) + Recognition (LBPH)
├── trainer.py            ← Model training from enrolled images
├── monitor.py            ← Live webcam feed with HUD overlay
├── weapon_engine.py      ← Async weapon detector worker (ONNX)
├── requirements.txt      ← Python dependencies
│
├── data/
│   ├── criminal_db/      ← Face images per enrolled person
│   ├── captured_faces/   ← Snapshots taken during detection
│   └── criminal_records.db  ← SQLite database
│
├── models/
│   ├── lbph_face_model.xml  ← Trained recognition model (auto-generated)
│   └── weapon_yolov8n.onnx  ← Weapon detector model (add manually)
│
└── logs/
    └── system.log        ← Runtime logs
```

---

## ⚙️ Installation

### Step 1 — Install Python dependencies

```bash
pip install opencv-python opencv-contrib-python numpy onnxruntime onnx
```

> ⚠️ You **must** install `opencv-contrib-python` for the LBPH face recognizer.
> If you have `opencv-python` already installed, uninstall it first:
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

### Step 2 — Verify installation

```bash
python -c "import cv2; r = cv2.face.LBPHFaceRecognizer_create(); print('OK')"
```

---

## 🚀 How to Run

```bash
python main.py
```

---

## 📋 Workflow (Follow in order!)

### 1. Enroll a Criminal
- Select **Option 1** from the menu
- Fill in name, CNIC, crime type, status
- Webcam opens — look at the camera
- **60 face images** are captured automatically
- Vary your head angle slightly for better accuracy

### 2. Train the Model
- Select **Option 2**
- Wait a few seconds for training to complete
- Model saved to `models/lbph_face_model.xml`

### 3. Run Live Recognition
- Select **Option 3**
- Webcam opens with real-time detection
- **Red box** = Known criminal detected → Alert triggered!
- **Green box** = Unknown person
- **Weapon box** = Weapon candidate from async ONNX worker (independent alert policy)

### 3.5 Test Video File (Weapon Detection Demo)
- Select **Option 3.5**
- Enter path to a local MP4 or AVI file
- Plays video with face + weapon detection overlay
- **No camera needed** — perfect for testing with downloaded YouTube clips
- Same detection logic as live feed; ideal for safe testing without real weapons

### Controls (during live feed)
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `S` | Save snapshot |
| `P` | Pause / Resume |

---

## 🔧 Configuration (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam index (try `1` if default doesn't work) |
| `ENROLL_FRAME_COUNT` | `60` | Number of face samples to capture per person |
| `RECOGNITION_CONFIDENCE_THRESHOLD` | `70` | Lower = stricter matching |
| `ALERT_COOLDOWN_SECONDS` | `30` | Min gap between repeated alerts |

---

## 🧠 Technical Details

| Component | Technology |
|-----------|-----------|
| Face Detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Face Recognition | LBPH — Local Binary Pattern Histogram |
| Database | SQLite via `sqlite3` (no server needed) |
| Image Processing | CLAHE normalization + resize to 100×100 |
| Alert System | Terminal output + Database log + Snapshot |

### Why LBPH?
- Works **without deep learning** (no GPU needed)
- Handles lighting variation well (with CLAHE)
- Can be **incrementally updated** without full retrain
- Gives an interpretable **confidence score**

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `cv2.face` not found | Install `opencv-contrib-python` (not just `opencv-python`) |
| Camera not opening | Change `CAMERA_INDEX` in `config.py` (try 0, 1, 2) |
| Poor recognition | Enroll more images in better lighting; lower threshold |
| False positives | Increase `RECOGNITION_CONFIDENCE_THRESHOLD` (e.g. 55) |

---

## 📌 Module Status
- ✅ Face Recognition (active)
- ✅ Weapon Detection (YOLOv8 ONNX async worker)
- 🔲 Alert & Notification (email / SMS)
- 🔲 Web Dashboard (Flask + HTML)

## 🎥 Safe Weapon Testing Without Real Weapons
1. Download local MP4 clips from YouTube showing weapon-like motion scenarios.
2. Place clips in a local test folder and run Option 3 against webcam or video runner (next step module).
3. Validate detections under motion blur, low light, and multiple people.
4. Include no-weapon clips to tune thresholds and reduce false positives.
5. Use toy/printed objects for desk testing only; avoid real weapons.


