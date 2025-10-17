# 🎯 YOLO11 Dart Detection - Streamlit Application

Production-ready Streamlit application for real-time dart detection and automatic scoring using YOLO11 and iPhone Continuity Camera.

## 📋 Features

- **🎯 Real-time Detection**: Live dart detection with 99% accuracy (mAP@0.5: 99.0%)
- **📱 Continuity Camera**: Native support for iPhone as camera source
- **🎲 Automatic Scoring**: Calculates dart scores (D20, T19, Bull, Miss, etc.)
- **📊 Live Calibration**: Real-time calibration status with visual feedback
- **⚡ High Performance**: 20-30 FPS on MacBook Pro M2
- **🎨 Intuitive UI**: Clean, user-friendly Streamlit interface

## 🔧 Requirements

### Hardware
- **macOS**: Ventura (13.0) or later
- **iPhone**: iOS 16+ (for Continuity Camera feature)
- **Mac**: M1/M2 recommended (Intel supported)
- **Camera**: MacBook camera or iPhone via Continuity Camera

### Software
- Python 3.8 or later
- pip (Python package manager)

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd /Users/fewzy/Dev/ai/deeper_darts/demo

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

Check that the model file exists:
```bash
ls -l /Users/fewzy/Dev/ai/deeper_darts/models/best6.pt
```

Test camera detection:
```bash
python3 -c "import cv2; [print(f'Camera {i}: Available') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

Expected output:
```
Camera 0: Available  # MacBook camera
Camera 1: Available  # iPhone Continuity Camera (if active)
```

### 3. Run Application

```bash
streamlit run dart_detector.py
```

The application will automatically open in your default browser at `http://localhost:8501`

## 📱 iPhone Continuity Camera Setup

### Enable Continuity Camera

**On macOS:**
1. Open **System Settings** → **General** → **AirDrop & Handoff**
2. Enable **Continuity Camera**

**On iPhone:**
1. Open **Settings** → **General** → **AirPlay & Handoff**
2. Enable **Continuity Camera**

### Activate Camera

1. Place iPhone near MacBook (within Bluetooth/Wi-Fi range)
2. Position iPhone in **landscape mode** (horizontal)
3. iPhone camera will appear as "Camera 1" in the app
4. Select "iPhone (Continuity Camera)" from dropdown

**Tip**: The iPhone screen will show the camera preview automatically when detected.

## 🎮 Usage

### Basic Workflow

1. **Launch Application**
   ```bash
   streamlit run dart_detector.py
   ```

2. **Select Camera**
   - Choose "iPhone (Continuity Camera)" from sidebar
   - Or use "MacBook Camera" if preferred

3. **Configure Detection**
   - Confidence Threshold: 0.50 (default)
   - IoU Threshold: 0.45 (default)
   - Image Size: 800 (recommended)

4. **Start Detection**
   - Click "▶️ Start Detection" button
   - Point camera at dartboard
   - Ensure good lighting

5. **Calibration**
   - Wait for 4 green boxes (calibration points)
   - Calibration points: 5-20, 13-6, 17-3, 8-11
   - Status shows "✅ Calibration: 4/4" when ready

6. **View Scores**
   - Scores appear automatically when darts detected
   - Individual dart scores (D20, T19, B, etc.)
   - Total score displayed in metric card

### Scoring Reference

| Score | Meaning | Points |
|-------|---------|--------|
| `DB` | Double Bull (Inner Bull) | 50 |
| `B` | Bull (Outer Bull) | 25 |
| `D20` | Double 20 | 40 |
| `T19` | Triple 19 | 57 |
| `13` | Single 13 | 13 |
| `Miss` | Outside board | 0 |

## ⚙️ Configuration

### Model Path

Default: `/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt`

To use a different model, update the path in the sidebar or modify `DEFAULT_MODEL_PATH` in `dart_detector.py`.

### Detection Parameters

**Confidence Threshold** (0.0 - 1.0)
- Default: 0.50
- Higher values = fewer false positives
- Lower values = more detections but may include noise

**IoU Threshold** (0.0 - 1.0)
- Default: 0.45
- Controls Non-Maximum Suppression (NMS)
- Higher values = more overlapping boxes kept

**Image Size** (640 or 800)
- Default: 800
- 800px = better accuracy, slower
- 640px = faster, slightly lower accuracy

### Camera Settings

**Resolution**: Automatically set to 1280x720
**FPS**: Depends on hardware (20-30 FPS typical)

## 🐛 Troubleshooting

### No Cameras Detected

**Problem**: "❌ No cameras detected!" message

**Solutions**:
1. Check iPhone is in **landscape mode**
2. Ensure both devices signed into same Apple ID
3. Enable Bluetooth and Wi-Fi on both devices
4. Restart Streamlit app
5. Test camera manually:
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(1); print('✅' if cap.isOpened() else '❌'); cap.release()"
   ```

### Model Not Found

**Problem**: "❌ Model not found" error

**Solutions**:
1. Check model path: `/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt`
2. Verify file exists:
   ```bash
   ls -l /Users/fewzy/Dev/ai/deeper_darts/models/best6.pt
   ```
3. Update model path in sidebar if located elsewhere

### Scores Not Calculating

**Problem**: Detections show but no scores appear

**Solutions**:
1. Verify 4/4 calibration points detected (green boxes)
2. Check calibration status in right panel
3. Ensure dartboard is well-lit and centered
4. All 4 calibration points must be visible simultaneously
5. Adjust confidence threshold if calibration points not detected

### Low FPS (< 10)

**Problem**: Slow inference, choppy video

**Solutions**:
1. Reduce image size: 800 → 640
2. Lower camera resolution in code:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
3. Close other applications
4. Check system resource usage

### Incorrect Scores

**Problem**: Scores don't match actual dart positions

**Solutions**:
1. Improve lighting conditions
2. Reduce camera angle (aim perpendicular to board)
3. Ensure all 4 calibration points correctly detected
4. Clean dartboard surface for better detection
5. Increase confidence threshold to reduce false positives

## 📊 Performance Benchmarks

### Model Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 99.0% |
| Precision | 99.29% |
| Recall | 98.14% |
| Inference Time | 30-50ms |

### Hardware Performance

| Device | Camera | Resolution | FPS | Latency |
|--------|--------|------------|-----|---------|
| MacBook Pro M2 | iPhone (1080p) | 1280x720 | 20-30 | 30-50ms |
| MacBook Pro Intel | iPhone (1080p) | 1280x720 | 15-25 | 40-80ms |
| MacBook Air M1 | iPhone (720p) | 1280x720 | 15-20 | 50-80ms |

## 🔍 Technical Details

### Architecture

```
┌─────────────────────────────────────────────────┐
│           macOS + Continuity Camera              │
│                                                  │
│  ┌───────────┐       Wi-Fi/BT      ┌─────────┐  │
│  │  iPhone   │ ──────────────────→ │ MacBook │  │
│  │(landscape)│                     │         │  │
│  └───────────┘                     └─────────┘  │
│                                          │       │
│                                          ▼       │
│                              ┌──────────────┐   │
│                              │ OpenCV Video │   │
│                              │  Capture     │   │
│                              └──────────────┘   │
│                                          │       │
│                                          ▼       │
│                              ┌──────────────┐   │
│                              │   YOLO11     │   │
│                              │  Inference   │   │
│                              └──────────────┘   │
│                                          │       │
│                                          ▼       │
│                              ┌──────────────┐   │
│                              │   Scoring    │   │
│                              │   Engine     │   │
│                              └──────────────┘   │
│                                          │       │
│                                          ▼       │
│                              ┌──────────────┐   │
│                              │  Streamlit   │   │
│                              │     UI       │   │
│                              └──────────────┘   │
└─────────────────────────────────────────────────┘
```

### Detection Classes

| Class ID | Name | Color | Purpose |
|----------|------|-------|---------|
| 0 | calibration_5_20 | Green | Top calibration point |
| 1 | calibration_13_6 | Green | Bottom calibration point |
| 2 | calibration_17_3 | Green | Left calibration point |
| 3 | calibration_8_11 | Green | Right calibration point |
| 4 | dart_tip | Various | Dart impact point |

### Board Configuration

Standard BDO dartboard dimensions (in meters):

| Parameter | Value | Description |
|-----------|-------|-------------|
| r_double | 0.170 | Center to outside double wire |
| r_treble | 0.1074 | Center to outside treble wire |
| r_outer_bull | 0.0159 | Outer bull radius |
| r_inner_bull | 0.00635 | Inner bull radius |
| w_double_treble | 0.01 | Wire width |

### Scoring Algorithm

1. **Calibration**: Detect 4 calibration points around board
2. **Transform**: Apply perspective transform to normalize coordinates
3. **Distance**: Calculate distance from dart to board center
4. **Angle**: Calculate angle from dart to center (0-360°)
5. **Region**: Determine ring (double, treble, single, bull)
6. **Segment**: Map angle to dartboard segment (20 segments)
7. **Score**: Combine region and segment to calculate score

## 🤝 Contributing

This is a production-ready implementation. For improvements or bug reports, please:

1. Document the issue clearly
2. Include system information (macOS version, Python version)
3. Provide screenshots/videos if applicable
4. Test with both MacBook camera and iPhone camera

## 📚 References

### Documentation

- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Apple Continuity Camera](https://support.apple.com/en-us/102546)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Related Files

- Model: `/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt`
- Config: `/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml`
- Scoring Logic: `/Users/fewzy/Dev/ai/deeper_darts/datasets/annotate.py`

## 📝 License

This application is part of the Deeper Darts project. Please refer to the main project license.

## 🎯 Next Steps

### Potential Enhancements

- [ ] Session history (save scores to CSV)
- [ ] Game modes (301, 501, Cricket)
- [ ] Multi-player support
- [ ] Voice announcements (text-to-speech)
- [ ] Dartboard overlay graphic
- [ ] Slow-motion replay
- [ ] Export to scoreboard apps
- [ ] Statistics and analytics

### Performance Optimization

- [ ] GPU acceleration (CUDA support)
- [ ] Frame skipping with interpolation
- [ ] Adaptive FPS based on system load
- [ ] Model quantization for faster inference

---

**Version**: 1.0.0
**Last Updated**: January 2025
**Status**: Production Ready ✅
