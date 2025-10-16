# 🚀 Browser Testing Quick Start

**Goal**: Test your YOLO11 dart detection model on iPhone Safari in **1-3 days**

---

## 📊 Your Model Performance

✅ **mAP@0.5**: 0.9900 (99.0% accuracy!)
✅ **Precision**: 0.9929
✅ **Recall**: 0.9814
✅ **Model**: `/Users/fewzy/Dev/ai/deeper_darts/models/best.pt`

**Your model is excellent!** Ready for browser deployment.

---

## 🎯 Fastest Path: Streamlit MVP (Recommended)

### Why Streamlit?

- ✅ **Zero frontend code** - pure Python
- ✅ **Built-in camera support** - works on iPhone
- ✅ **No model conversion** - use best.pt directly
- ✅ **1-3 days** - fastest validation path
- ✅ **Free** - ngrok free tier sufficient

### 5-Minute Setup

```bash
cd /Users/fewzy/Dev/ai/deeper_darts

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install streamlit ultralytics opencv-python numpy pillow

# 3. Create app directory
mkdir -p streamlit_app

# 4. Copy Streamlit code (see below)
# ... create streamlit_app/app.py

# 5. Run app
streamlit run streamlit_app/app.py --server.port 8501
```

### Minimal Streamlit App

**File**: `streamlit_app/app.py`

```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="🎯 Dart Detection", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return YOLO('/Users/fewzy/Dev/ai/deeper_darts/models/best.pt')

model = load_model()

st.title("🎯 YOLO11 Dart Detection")
st.markdown("**mAP@0.5**: 99.0% | **Precision**: 99.29% | **Recall**: 98.14%")

# Controls
conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)

# Camera input
img = st.camera_input("📷 Take a photo")

if img:
    image = Image.open(img)

    # Inference
    with st.spinner("Running YOLO11..."):
        results = model.predict(source=np.array(image), conf=conf, verbose=False)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(results[0].plot(), caption="Detections", use_container_width=True)

    # Stats
    detections = results[0].boxes
    calib = sum(1 for b in detections if int(b.cls[0]) < 4)
    darts = sum(1 for b in detections if int(b.cls[0]) == 4)

    st.success(f"✅ Calibration: {calib}/4 | 🎯 Darts: {darts}")
```

**That's it!** Run and test locally first.

### iPhone Testing with ngrok

```bash
# Install ngrok (one-time)
brew install ngrok

# Start ngrok tunnel (in separate terminal)
ngrok http 8501

# Output:
# Forwarding    https://abc123.ngrok.io -> http://localhost:8501
```

**Open the HTTPS URL on your iPhone Safari** → Test camera detection!

---

## 🎯 Expected Results

| Test Scenario | Expected Result |
|--------------|----------------|
| **Well-lit dartboard** | 4/4 calibration points + dart tips |
| **Multiple darts** | All dart tips detected (>90% confidence) |
| **Angled view** | Still detects (model trained with augmentation) |
| **Low light** | May miss some detections (adjust confidence) |

### Success Criteria

✅ 4/4 calibration points detected consistently
✅ Dart tips detected with >90% confidence
✅ Works in various lighting conditions
✅ Detection completes in <5 seconds

**If successful**: Your model is ready for production deployment!

---

## 📋 Full Implementation Path

After validating with Streamlit, you have **three options** for production:

### Option 1: Keep Streamlit (Simplest)

**Pros**: No code changes, works immediately
**Cons**: Limited real-time performance (1-3 FPS)
**Best For**: Testing, demos, low-frequency use
**Cost**: Free (ngrok) or $7/month (Streamlit Cloud)

### Option 2: FastAPI + React (Recommended)

**Pros**: Real-time (15-30 FPS), professional UI
**Cons**: Requires frontend development (2-5 weeks)
**Best For**: Production app, league play
**Cost**: $5-20/month (Railway + Vercel)

### Option 3: Full Client-Side (ONNX.js)

**Pros**: Offline-capable, no server costs
**Cons**: Slower (10-15 FPS), model conversion required
**Best For**: PWA, offline use
**Cost**: Free (GitHub Pages or Vercel)

---

## 📚 Detailed Documentation

Comprehensive implementation plans available:

**Main Plan**: `research/12_browser_inference_implementation_plan.md`
- 3 phased approaches with complete code
- Performance benchmarks for your model
- Technology stack comparisons
- Testing checklists
- Timeline: 1-3 days (MVP) to 4-8 weeks (production)

**Supporting Research**:
- `docs/architecture/system-architecture.md` - System design
- `docs/research/model_conversion_guide.md` - ONNX export guide
- `docs/research/github_yolo_projects.md` - 7 reference projects
- `docs/research/dart_scoring_ui_research.md` - UI/UX patterns

---

## 🚀 Next Steps

### Today (30 minutes)

1. ✅ Create Streamlit app with minimal code above
2. ✅ Run locally: `streamlit run streamlit_app/app.py`
3. ✅ Test with webcam first
4. ✅ Start ngrok tunnel
5. ✅ Test on iPhone Safari

### This Week

1. Validate detection accuracy on iPhone
2. Test multiple lighting conditions
3. Test various dartboard angles
4. Gather user feedback
5. Decide on production approach

### Next Month (Optional)

1. **Week 1**: Continue Streamlit validation
2. **Week 2-3**: Build FastAPI backend (if going real-time)
3. **Week 4**: Build React frontend
4. **Week 5+**: Add homography and scoring

---

## 🆘 Troubleshooting

### Camera permission denied on iPhone

**Solution**: iOS requires HTTPS for camera access. Always use ngrok HTTPS URL, not localhost.

### Streamlit connection issues

**Solution**: Ensure ports aren't blocked. Try different port: `streamlit run app.py --server.port 8502`

### Slow inference on iPhone

**Expected**: Streamlit sends full image to server. For real-time, use FastAPI + WebSocket (Phase 2).

### ngrok URL expires

**Solution**: Free tier gives 2-hour sessions. Re-run `ngrok http 8501` for new URL. Or upgrade to ngrok paid ($8/month) for persistent URLs.

---

## ✅ Success Criteria Checklist

After testing with Streamlit MVP, verify:

- [ ] App loads on iPhone Safari (HTTPS required)
- [ ] Camera permission granted
- [ ] Photos captured successfully
- [ ] YOLO inference completes in <5 seconds
- [ ] 4/4 calibration points detected (well-lit)
- [ ] Dart tips detected with >90% confidence
- [ ] Works at multiple angles
- [ ] Confidence slider adjusts detections
- [ ] User feedback: "Model works on iPhone!"

**All checked?** → Your model is production-ready! Proceed to Phase 2 for real-time inference.

---

## 💡 Pro Tips

1. **Start with photos, not video** - Streamlit camera_input is perfect for validation
2. **Test lighting early** - Most detection issues are lighting-related
3. **Keep ngrok terminal open** - Closing it kills the tunnel
4. **Use confidence=0.5** - Start here, adjust based on results
5. **Document issues** - Track edge cases for Phase 2 improvements

---

## 📧 Questions?

Refer to comprehensive guides:
- **Implementation Plan**: `research/12_browser_inference_implementation_plan.md`
- **Architecture**: `docs/architecture/system-architecture.md`
- **Model Export**: `docs/research/model_conversion_guide.md`

---

**Ready?** Run these commands:

```bash
cd /Users/fewzy/Dev/ai/deeper_darts
python3 -m venv venv
source venv/bin/activate
pip install streamlit ultralytics opencv-python numpy pillow
streamlit run streamlit_app/app.py --server.port 8501
```

**Then in another terminal**:

```bash
ngrok http 8501
```

**Open ngrok HTTPS URL on iPhone → Start testing!** 🎯

---

**Status**: Ready to implement ✅
**Duration**: 30 minutes setup + 1-3 days testing
**Cost**: $0 (free tier)
**Output**: Validated model on iPhone camera
