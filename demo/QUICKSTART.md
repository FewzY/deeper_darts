# 🚀 Quick Start Guide

Get up and running in 2 minutes!

## One-Line Start

```bash
cd /Users/fewzy/Dev/ai/deeper_darts/demo && ./run.sh
```

That's it! The script automatically:
- Creates virtual environment
- Installs dependencies
- Checks cameras
- Launches Streamlit app

## Manual Setup (if preferred)

```bash
# 1. Navigate to demo folder
cd /Users/fewzy/Dev/ai/deeper_darts/demo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run dart_detector.py
```

## iPhone Continuity Camera Setup

1. **Place iPhone in landscape mode** (horizontal) near MacBook
2. Wait 2-3 seconds for Continuity Camera to activate
3. In Streamlit app, select **"iPhone (Continuity Camera)"** from dropdown
4. Click **"▶️ Start Detection"**

## First Use Checklist

- [ ] iPhone in landscape mode
- [ ] Both devices signed into same Apple ID
- [ ] Bluetooth and Wi-Fi enabled on both devices
- [ ] Camera permissions granted
- [ ] Good lighting on dartboard
- [ ] All 4 calibration points visible

## Expected Results

✅ **Camera**: iPhone appears at index 1 (1920x1080)
✅ **FPS**: 20-30 frames per second
✅ **Calibration**: 4/4 points detected (green boxes)
✅ **Darts**: Detected with colored boxes
✅ **Scores**: Automatic calculation (D20, T19, B, etc.)

## Troubleshooting

**iPhone not detected?**
- Ensure iPhone is in **landscape mode**
- Check Continuity Camera is enabled in System Settings
- Restart both devices if needed

**Scores not calculating?**
- Ensure 4/4 calibration points visible (green boxes)
- Improve lighting conditions
- Center dartboard in frame

## Next Steps

See full documentation in [README.md](README.md) for:
- Detailed configuration options
- Advanced troubleshooting
- Performance optimization
- Technical architecture

---

**Need help?** Check [README.md](README.md) or refer to the plan in `/research/13_continuity_camera_streamlit_plan.md`
