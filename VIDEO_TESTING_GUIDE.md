# Video File Testing Guide (Weapon Detection)

## Quick Start

### 1. Download Test Video
Download an MP4 video from YouTube showing weapon-related scenarios:
```bash
# Using yt-dlp (install: pip install yt-dlp)
yt-dlp -f worst "https://www.youtube.com/watch?v=..." -o test_video.mp4
```

**Recommended scenarios:**
- Action scenes with weapon props
- Training demonstrations
- Self-defense footage
- Security footage samples
- Motion blur and varying lighting conditions

### 2. Place Video File
Save the video in your workspace folder or note the full path:
```
c:\Users\Admin\Desktop\face_recognition_system\test_video.mp4
```

### 3. Run Video Test
```bash
python main.py
```

Then:
1. Select **Option 3.5** (Test Video File)
2. Enter the video file path (or just filename if in current directory)
3. Press ENTER to start playback

### 4. Controls During Playback
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit and return to menu |
| `S` | Save snapshot of current frame |
| `P` | Pause / Resume |
| `F` | Show FPS statistics |

## Safe Testing (No Real Weapons)

Why video file testing is safe:
✅ **Offline testing** — No connection to real weapons
✅ **Replayable** — Same scenarios every time for consistency
✅ **Controllable** — Test specific lighting, motion, angles
✅ **Repeatable** — Run multiple times without changing setup
✅ **Cost-free** — Use freely available video clips

### Test Scenarios

#### Scenario 1: Motion & Clarity
Use video with:
- Clear visibility of objects
- Natural hand/arm motion
- Changing angles and distances
- Expected result: High detection accuracy

#### Scenario 2: Low Light (Challenge)
Use darker footage:
- Indoor security camera footage
- Dim lighting conditions
- Shadow areas
- Expected result: Lower accuracy; helps tune thresholds

#### Scenario 3: False Positive Test
Use videos with **no weapons but similar objects:**
- Umbrellas
- Pipes
- Tools
- Kitchen utensils
- Toys
- Expected result: Should NOT trigger weapon alerts (goal: zero false positives)

#### Scenario 4: Mixed Content
Multi-person videos:
- Different people holding/not holding objects
- Background objects (phones, bottles, etc)
- Expected result: Detections only when weapons actually present

## Output & Logging

### During Video Playback
- **Terminal output**: Real-time alerts for face + weapon detections
- **HUD overlay**: Live FPS, detection counts, threat levels
- **Snapshots**: Auto-saved when weapons detected (if enabled in config)

### After Video Playback
Check results:
1. **Option 5** (View Detection Logs) → Shows face events
2. **Option 7** (System Status) → Shows weapon event counts
3. **data/captured_faces/** → Contains saved snapshots

## Configuration Tuning

### Adjust Sensitivity
Edit `config.py`:

```python
# Lower = more detections, higher = stricter
WEAPON_CONFIDENCE_THRESHOLD = 0.6  # Try 0.4-0.8

# NMS filtering (overlapping boxes)
WEAPON_NMS_THRESHOLD = 0.45  # Try 0.3-0.6

# Minimum box area (pixels) to count as detection
WEAPON_MIN_BOX_AREA = 900  # Try 500-2000
```

### Reload & Re-test
After config changes, run video again without restarting Python:
1. Complete current video test (press Q to exit)
2. Config values are reloaded on next Option 3.5 run

## Example Workflow

```
Step 1: Download diverse test videos
  - Action movie clip (motion + clarity)
  - Security cam footage (low light)
  - Office scene (no weapons)
  
Step 2: Run main.py → Option 3.5
  - Test each video, note detection accuracy
  
Step 3: Review logs (Option 5)
  - Check false positives in non-weapon video
  - Verify true positives in weapon video
  
Step 4: Adjust thresholds if needed
  - Edit config.py
  
Step 5: Re-test with same videos
  - Compare results after tuning
  
Step 6: Switch to live camera (Option 3) when confident
```

## Troubleshooting

### "Cannot open video file"
- Check file path is correct
- Verify MP4/AVI codec is standard (H.264, MPEG4)
- Try full absolute path: `C:\Users\Admin\Desktop\test.mp4`

### Very slow playback
- Video is being processed frame-by-frame → normal for inference
- Weapon detection runs asynchronously, so FPS may be 15-25 on CPU
- Lower resolution = faster processing (but less detail)

### No weapons detected
- Confidence threshold too high → lower it in config.py
- Object too small in frame → move closer in video or use crop
- Model not loaded → ensure weapon_yolov8n.onnx exists

### Too many false positives
- Confidence threshold too low → raise it in config.py
- Min box area too small → increase it
- Model needs fine-tuning on your domain (future work)

## Next Steps

1. **Live deployment**: Once confident on video tests, switch to Option 3 for live camera feed
2. **Fine-tuning**: Collect detection samples for custom model training
3. **Integration**: Adapt detections to your alert/notification backend
