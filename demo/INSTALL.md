# 🚀 Installation Guide - YOLO11 Dart Detection

## ⚠️ IMPORTANT: Fixed Compatibility Issues

**Two errors were fixed**:
1. ✅ **Python 3.13 + numpy 1.24.3 compatibility** - Updated to use numpy 2.0+ for Python 3.13
2. ✅ **Streamlit `use_container_width` error** - Changed to `use_column_width` and added BGR→RGB conversion

---

## 📋 Prerequisites

- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12, or 3.13 (all supported)
- **macOS**: Ventura (13.0)+ for Continuity Camera
- **iPhone**: iOS 16+ (optional, for Continuity Camera)
- **Model File**: `/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt` (115 MB)

---

## 🔧 Installation Steps

### Method 1: Automated (Recommended)

```bash
cd /Users/fewzy/Dev/ai/deeper_darts/demo
./run.sh
```

**The script will**:
- Check Python version
- Create virtual environment
- Install all dependencies
- Verify camera detection
- Launch Streamlit app

### Method 2: Manual Installation

```bash
cd /Users/fewzy/Dev/ai/deeper_darts/demo

# Step 1: Create virtual environment
python3 -m venv venv

# Step 2: Activate virtual environment
source venv/bin/activate

# Step 3: Upgrade pip (important!)
pip install --upgrade pip setuptools wheel

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Verify installation
python3 -c "import streamlit, ultralytics, cv2, numpy; print('✅ All packages installed')"

# Step 6: Test camera detection
python3 -c "import cv2; print('Cameras:', [i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Step 7: Launch application
streamlit run dart_detector.py
```

---

## 🐛 Troubleshooting

### Error 1: numpy compatibility with Python 3.13

**Symptom**:
```
ERROR: Cannot import 'setuptools.build_meta'
```

**Solution**: ✅ **FIXED** in requirements.txt
- Now uses numpy 2.0+ for Python 3.13
- Uses numpy 1.24+ for Python 3.8-3.12

### Error 2: `use_container_width` not found

**Symptom**:
```
TypeError: image() got an unexpected keyword argument 'use_container_width'
```

**Solution**: ✅ **FIXED** in dart_detector.py (line 491-493)
- Changed to `use_column_width=True`
- Added BGR→RGB conversion for proper color display

### Still Having Issues?

1. **Delete and recreate venv**:
   ```bash
   cd /Users/fewzy/Dev/ai/deeper_darts/demo
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

2. **Check Python version**:
   ```bash
   python3 --version
   # Should be 3.8-3.13
   ```

3. **Verify model file**:
   ```bash
   ls -lh /Users/fewzy/Dev/ai/deeper_darts/models/best6.pt
   # Should show ~115 MB file
   ```

4. **Test camera access**:
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened()); cap.release()"
   ```

---

## 📦 Dependencies (Updated)

```
streamlit>=1.40.0           # Fixed: use_column_width support
ultralytics>=8.3.0          # YOLO11
opencv-python>=4.9.0        # Camera access
numpy>=1.24.3,<2.0; python_version < "3.13"  # Python 3.8-3.12
numpy>=2.0.0; python_version >= "3.13"        # Python 3.13+
pillow>=10.0.0              # Image processing
```

---

## ✅ Verification

After installation, verify everything works:

```bash
cd /Users/fewzy/Dev/ai/deeper_darts/demo
source venv/bin/activate

# Test 1: Check imports
python3 -c "
import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
print('✅ All imports successful')
"

# Test 2: Test camera detection
python3 -c "
import cv2
cameras = [i for i in range(5) if cv2.VideoCapture(i).isOpened()]
print(f'✅ Cameras detected: {cameras}')
"

# Test 3: Test model loading
python3 -c "
from ultralytics import YOLO
model = YOLO('/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt')
print(f'✅ Model loaded: {model.names}')
"

# Test 4: Test scoring logic
python3 test_scoring.py
```

**Expected output**:
```
✅ All imports successful
✅ Cameras detected: [0, 1]
✅ Model loaded: {0: 'calibration_5_20', ...}
✅ All scoring tests completed!
```

---

## 🚀 Launch Application

```bash
cd /Users/fewzy/Dev/ai/deeper_darts/demo
source venv/bin/activate
streamlit run dart_detector.py
```

**Browser will open at**: `http://localhost:8501`

---

## 📱 iPhone Continuity Camera Setup

1. **Enable on Mac**: System Settings → General → AirDrop & Handoff → Continuity Camera ✅
2. **Enable on iPhone**: Settings → General → AirPlay & Handoff → Continuity Camera ✅
3. **Activate**: Place iPhone in landscape mode near MacBook
4. **Select**: Choose "iPhone (Continuity Camera)" from app dropdown

---

## 🎯 Quick Test

After launching the app:

1. Click "Start Detection"
2. Point camera at dartboard
3. Verify:
   - ✅ Video feed displays
   - ✅ FPS counter shows 15-30 FPS
   - ✅ Calibration points detected (green boxes)
   - ✅ Dart tips detected (red boxes)
   - ✅ Scores calculate when 4 calibration points visible

---

## 💡 Performance Tips

- **Use iPhone camera** for best quality (1920x1080 vs 1280x720)
- **Good lighting** improves detection accuracy
- **Keep dartboard centered** in frame
- **Adjust confidence** if detections are too sensitive/insensitive
- **Use image size 800** for best accuracy (640 for speed)

---

## 📞 Support

If you encounter any other issues:

1. Check this guide first
2. Review `demo/README.md` for detailed documentation
3. Test with `test_scoring.py` to isolate scoring issues
4. Verify model file is not corrupted (should be 115 MB)

---

**Installation Time**: 2-5 minutes
**Status**: ✅ All known issues fixed
**Last Updated**: January 2025
