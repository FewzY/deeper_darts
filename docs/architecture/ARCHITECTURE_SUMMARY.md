# Architecture Summary - Web-Based YOLO11 Dart Detection

**Date**: 2025-10-17
**Status**: ✅ COMPLETE
**Decision**: Phased Implementation Strategy

---

## 📋 Quick Reference

### Recommended Approach: Phased Implementation

| Phase | Technology | Timeline | Cost | Performance | Purpose |
|-------|-----------|----------|------|-------------|---------|
| **Phase 1: MVP** | Streamlit + ngrok | Week 1 | $0 | 1-3 FPS | Rapid validation |
| **Phase 2: Production** | FastAPI + React | Week 2-5 | $5-20/mo | 15-30 FPS | Real-time deployment |
| **Phase 3: Enhancement** | ONNX.js PWA | Future | $0 | 20-30 FPS | Offline capability |

---

## 🎯 Architecture Decision

**Selected**: **Hybrid Architecture (FastAPI + React)** for production
**Path**: Streamlit MVP first → Hybrid production → Optional client-side PWA

### Key Rationale
1. **Fast Validation**: Streamlit MVP in 2-3 days proves concept
2. **Real-Time Performance**: Hybrid achieves 15-30 FPS target
3. **Production Ready**: Battle-tested stack (FastAPI + React)
4. **Cost Effective**: $5-20/month hosting on Railway + Vercel
5. **Progressive Enhancement**: Can add offline mode in Phase 3

---

## 🏗️ System Architecture Diagrams

### Phase 1: Streamlit MVP
```
┌────────────────┐
│ iPhone Safari  │  User captures frames manually
└───────┬────────┘
        │ HTTPS (ngrok tunnel)
        ▼
┌────────────────┐
│   Streamlit    │  Python backend with UI
│   + YOLO11     │  Uses best.pt directly
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Score Calc    │  Dartboard geometry
│  + Homography  │  Perspective correction
└────────────────┘

Performance: 1-3 FPS, 500-1000ms latency
Cost: $0 (local + ngrok free tier)
```

### Phase 2: Hybrid Production
```
┌─────────────────────────────────────────────┐
│            iPhone Safari                     │
│  ┌───────────────────────────────────────┐  │
│  │  React Frontend (TypeScript + Vite)  │  │
│  │  - MediaStream video capture         │  │
│  │  - Canvas bounding box rendering     │  │
│  │  - WebSocket client                  │  │
│  └──────────────┬────────────────────────┘  │
└─────────────────┼────────────────────────────┘
                  │ WebSocket (WSS)
                  │ Bi-directional streaming
                  ▼
┌─────────────────────────────────────────────┐
│         FastAPI Backend (Python)             │
│  ┌───────────────────────────────────────┐  │
│  │  WebSocket Connection Manager        │  │
│  └──────────────┬────────────────────────┘  │
│                 ▼                            │
│  ┌───────────────────────────────────────┐  │
│  │  YOLO11 Inference (best.pt)          │  │
│  │  50-100ms inference time             │  │
│  └──────────────┬────────────────────────┘  │
│                 ▼                            │
│  ┌───────────────────────────────────────┐  │
│  │  Score Calculator + Homography       │  │
│  │  Dartboard geometry logic            │  │
│  └──────────────┬────────────────────────┘  │
└─────────────────┼────────────────────────────┘
                  │ JSON response
                  ▼
        (Render on client canvas)

Performance: 15-30 FPS, 100-250ms latency
Cost: $5-20/month (Railway + Vercel)
Deployment: Docker containerized backend
```

### Phase 3: Client-Side PWA (Future)
```
┌─────────────────────────────────────────────┐
│         iPhone Safari (Offline)              │
│  ┌───────────────────────────────────────┐  │
│  │     React PWA                         │  │
│  │     ┌───────────────────────────┐     │  │
│  │     │  MediaStream Video        │     │  │
│  │     └──────────┬────────────────┘     │  │
│  │                ▼                       │  │
│  │     ┌───────────────────────────┐     │  │
│  │     │  ONNX Runtime Web         │     │  │
│  │     │  (WebGL acceleration)     │     │  │
│  │     │  model.onnx (< 10 MB)     │     │  │
│  │     └──────────┬────────────────┘     │  │
│  │                ▼                       │  │
│  │     ┌───────────────────────────┐     │  │
│  │     │  Score Calculator         │     │  │
│  │     │  (Client-side JS)         │     │  │
│  │     └──────────┬────────────────┘     │  │
│  │                ▼                       │  │
│  │     ┌───────────────────────────┐     │  │
│  │     │  Canvas Renderer          │     │  │
│  │     └───────────────────────────┘     │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘

Performance: 20-30 FPS, 40-70ms latency
Cost: $0 (GitHub Pages / Cloudflare)
Offline: ✅ Full PWA with Service Workers
```

---

## 🔧 Technology Stack

### Phase 1: Streamlit MVP
```yaml
Backend:
  - Python 3.10+
  - Streamlit 1.30+
  - Ultralytics 8.1+ (YOLO11)
  - OpenCV 4.8+
  - NumPy 1.24+

Deployment:
  - Local development machine
  - ngrok 3.x for HTTPS tunnel

Model:
  - best.pt (PyTorch format, no conversion)
```

### Phase 2: Hybrid Production
```yaml
Frontend:
  - React 18
  - TypeScript 5.x
  - Vite 5 (build tool)
  - Socket.IO Client 4.x
  - TailwindCSS 3.x

Backend:
  - Python 3.10+
  - FastAPI 0.109+
  - Ultralytics 8.1+ (YOLO11)
  - python-socketio 5.x
  - OpenCV 4.8+
  - NumPy 1.24+

Deployment:
  - Frontend: Vercel (free hobby tier)
  - Backend: Railway ($5-20/month)
  - Docker + Docker Compose

Model:
  - best.pt (PyTorch, server-side)
```

### Phase 3: Client-Side PWA
```yaml
Frontend:
  - React 18
  - TypeScript 5.x
  - ONNX Runtime Web 1.17+
  - Web Workers (for inference)
  - Service Workers (for offline)

Model Conversion:
  - best.pt → best.onnx (Ultralytics export)
  - onnx-simplifier (graph optimization)
  - INT8 quantization (4x size reduction)
  - Target: < 10 MB model size

Deployment:
  - GitHub Pages (free)
  - Cloudflare Pages (free)
  - Static hosting with CDN
```

---

## 📊 Performance Targets

| Metric | Phase 1 (MVP) | Phase 2 (Production) | Phase 3 (PWA) |
|--------|---------------|----------------------|---------------|
| **FPS** | 1-3 | 15-30 | 20-30 |
| **Latency (end-to-end)** | 500-1000ms | 100-250ms | 40-70ms |
| **Inference Time** | 50-100ms | 50-100ms | 30-50ms |
| **Model Load Time** | N/A (server) | N/A (server) | 5-8s |
| **Initial Page Load** | 2s | 3-5s | 5-8s (incl model) |
| **Model Size** | N/A | N/A | < 10 MB |
| **Memory Usage (client)** | < 100 MB | < 150 MB | 200-300 MB |

---

## 💰 Cost Analysis

| Phase | Hosting | Monthly Cost | Setup Time |
|-------|---------|--------------|------------|
| **Phase 1** | Local + ngrok | $0 | 5 minutes |
| **Phase 2 (Low Traffic)** | Railway + Vercel | $5-10 | 1 hour |
| **Phase 2 (Production)** | Railway Pro + Vercel | $20-30 | 1 hour |
| **Phase 3** | GitHub Pages | $0 | 30 minutes |

---

## 🚀 Development Timeline

### Week 1: Streamlit MVP
- **Day 1-2**: Basic detection + UI (Streamlit + YOLO11)
- **Day 3-4**: Score calculation + homography
- **Day 5**: iPhone Safari testing + bug fixes

**Deliverable**: Working prototype for user validation

### Week 2-3: Model Optimization
- **Week 2**: ONNX conversion experiments
- **Week 3**: Performance benchmarking

**Deliverable**: Optimized model (optional for Phase 2)

### Week 4-5: Hybrid Production
- **Week 4**: Backend development (FastAPI + WebSocket)
- **Week 5**: Frontend development (React) + deployment

**Deliverable**: Production app at public URL

### Week 6+: Refinement
- User testing and feedback
- Bug fixes and performance tuning
- Evaluate Phase 3 (PWA) necessity

---

## 🔑 Key Technical Components

### 1. Model Inference Pipeline
```python
# Phase 1 & 2: PyTorch (best.pt)
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(image, conf=0.5, iou=0.45)

# Parse detections
boxes = results[0].boxes.xyxy.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
```

```javascript
// Phase 3: ONNX Runtime Web
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('best_opt.onnx', {
  executionProviders: ['webgl', 'wasm']
});

const tensor = preprocessImage(imageData);
const results = await session.run({ images: tensor });
```

### 2. Camera Stream Handling
```javascript
// MediaStream API (Phase 2 & 3)
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    facingMode: 'environment', // Rear camera on iPhone
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 }
  }
});

videoElement.srcObject = stream;
```

### 3. Score Calculation
```python
# Dartboard geometry constants
BULLSEYE_RADIUS = 6.35  # mm
OUTER_BULL_RADIUS = 15.9
TRIPLE_INNER = 99
TRIPLE_OUTER = 107
DOUBLE_INNER = 162
DOUBLE_OUTER = 170

SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
            3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

def calculate_score(dart_point, board_center):
    distance = np.linalg.norm(dart_point - board_center)
    angle = np.degrees(np.arctan2(
        dart_point[1] - board_center[1],
        dart_point[0] - board_center[0]
    ))

    # Determine segment and multiplier
    # ... (full logic in system-architecture.md)

    return score
```

### 4. Canvas Rendering
```javascript
// Bounding box overlay (Phase 2 & 3)
function renderDetections(ctx, detections) {
  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;

    // Draw box
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label
    ctx.fillStyle = '#00ff00';
    ctx.font = '16px Arial';
    ctx.fillText(
      `${det.class} ${(det.confidence * 100).toFixed(1)}%`,
      x1, y1 - 5
    );
  });
}
```

### 5. WebSocket Communication (Phase 2)
```python
# Backend (FastAPI)
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async for message in websocket.iter_json():
        # Decode base64 image
        image_bytes = base64.b64decode(message['data'])

        # Inference
        results = model.predict(image_bytes)

        # Send response
        await websocket.send_json({
            'detections': results.tolist(),
            'timestamp': time.time()
        })
```

```typescript
// Frontend (React)
const ws = new WebSocket('wss://your-backend.railway.app/ws');

ws.onmessage = (event) => {
  const results = JSON.parse(event.data);
  renderDetections(results.detections);
};

// Send frame every 66ms (15 FPS)
setInterval(() => {
  const frame = captureFrame(videoElement);
  ws.send(JSON.stringify({ data: frame }));
}, 66);
```

---

## 🎯 Success Criteria

### Phase 1 (MVP) - Go/No-Go Decision
- ✅ Detection accuracy > 85% on test images
- ✅ Camera works on iPhone Safari with HTTPS
- ✅ Score calculation produces correct results
- ✅ UI is usable on mobile (responsive)
- ✅ No major Safari compatibility blockers

### Phase 2 (Production) - Acceptance Criteria
- ✅ Real-time video streaming (15-30 FPS)
- ✅ End-to-end latency < 250ms
- ✅ Stable WebSocket connections (auto-reconnect)
- ✅ Deployed with HTTPS on public domain
- ✅ Error handling and loading states

### Phase 3 (PWA) - Optional Enhancement
- ✅ Offline functionality with Service Workers
- ✅ Model cached in IndexedDB
- ✅ 20-30 FPS inference on-device
- ✅ < 10 MB total bundle size

---

## 📁 Documentation Structure

```
/docs/architecture/
├── ARCHITECTURE_SUMMARY.md          # This file (executive summary)
├── system-architecture.md           # Complete technical specification
├── ADR-001-architecture-decision.md # Architecture Decision Record
├── technology-stack-comparison.md   # Detailed technology evaluation
└── /diagrams/                       # Visual diagrams (future)
```

---

## 🔄 Next Steps

### Immediate Actions (Week 1)
1. **Setup Streamlit MVP**
   ```bash
   pip install streamlit ultralytics opencv-python
   streamlit run app.py
   ngrok http 8501
   ```

2. **Implement Core Features**
   - Camera input with `st.camera_input()`
   - YOLO11 inference pipeline
   - Score calculation logic
   - Bounding box visualization

3. **iPhone Testing**
   - Test camera access on Safari
   - Validate detection accuracy
   - Measure latency and FPS
   - Collect user feedback

### Phase 2 Go/No-Go (End of Week 1)
**Decision Point**: Proceed to production implementation?

**Criteria**:
- Technical validation complete ✓
- User feedback positive ✓
- No major blockers identified ✓
- Budget approved ($5-20/month) ✓

### Future Enhancements (Post Phase 2)
- Multi-player support (WebSocket rooms)
- Game history and statistics
- Camera calibration wizard
- Automatic board detection
- PWA offline mode (Phase 3)

---

## 📚 Key References

- **Main Documentation**: `/docs/architecture/system-architecture.md`
- **Architecture Decision**: `/docs/architecture/ADR-001-architecture-decision.md`
- **Technology Comparison**: `/docs/architecture/technology-stack-comparison.md`

**External Resources**:
- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com)
- [FastAPI WebSocket Guide](https://fastapi.tiangolo.com/advanced/websockets/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [MediaStream API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_API)

---

## ✅ Architecture Design Status

**Status**: ✅ **COMPLETE**
**Date Completed**: 2025-10-17
**Stored in Memory**: `web_inference_architecture` (namespace: `deeper_darts`)

**Files Created**:
1. `/docs/architecture/system-architecture.md` (8000+ words)
2. `/docs/architecture/ADR-001-architecture-decision.md` (Architecture Decision Record)
3. `/docs/architecture/technology-stack-comparison.md` (Comprehensive tech comparison)
4. `/docs/architecture/ARCHITECTURE_SUMMARY.md` (This file)

**Ready for Implementation**: ✅ Phase 1 (Streamlit MVP) can begin immediately
