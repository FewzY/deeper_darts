# Web-Based YOLO11 Dart Detection System Architecture

## Executive Summary

This document outlines the system architecture for a browser-based YOLO11 dart detection system optimized for iPhone camera input. After evaluating three architectural approaches, we recommend a **Hybrid Architecture (Option C)** with progressive enhancement for production, but **Streamlit + ngrok (Option A)** for rapid prototyping.

---

## 1. Architecture Options Analysis

### Option A: Streamlit + ngrok (RECOMMENDED FOR MVP)

#### Architecture Overview
```
┌─────────────────┐
│  iPhone Safari  │
│   (Camera UI)   │
└────────┬────────┘
         │ HTTPS (ngrok)
         ▼
┌─────────────────┐
│  ngrok Tunnel   │
│  (HTTPS Proxy)  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Streamlit     │
│   Web Server    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ultralytics     │
│ YOLO11 Engine   │
│   (best.pt)     │
└─────────────────┘
```

#### Component Breakdown

**1. Frontend (Streamlit)**
- `st.camera_input()` for image capture
- Canvas overlay for bounding boxes
- Real-time score display
- Session state management

**2. Backend (Python)**
- Ultralytics YOLO11 inference
- Image preprocessing pipeline
- Homography transformation
- Score calculation logic

**3. Tunneling (ngrok)**
- HTTPS endpoint generation
- Secure camera access on iOS
- Port forwarding (8501)

#### Pros
✅ **Fastest time to market**: 2-3 hours to working prototype
✅ **No model conversion**: Use best.pt directly
✅ **Native Python ecosystem**: Full Ultralytics API
✅ **Easy debugging**: Familiar Python stack
✅ **Minimal dependencies**: pip install streamlit ultralytics

#### Cons
❌ **Poor real-time performance**: Upload-process-display cycle
❌ **High latency**: 500-1000ms round-trip
❌ **No continuous video**: Frame-by-frame capture only
❌ **ngrok limitations**: Free tier tunnels expire
❌ **Not production-ready**: Temporary solution

#### Performance Targets
- **Inference latency**: 50-100ms (server-side)
- **Round-trip latency**: 500-1000ms (network overhead)
- **FPS**: 1-3 (manual capture)
- **Model size**: N/A (server-side)

#### Technology Stack
```yaml
Backend:
  - Python 3.10+
  - Streamlit 1.30+
  - Ultralytics 8.1+
  - OpenCV 4.8+
  - NumPy 1.24+

Deployment:
  - ngrok 3.x
  - uvicorn (optional)

Development:
  - Local MacBook (M1/M2)
  - iPhone Safari for testing
```

#### Deployment Workflow
```bash
# 1. Install dependencies
pip install streamlit ultralytics ngrok

# 2. Start Streamlit
streamlit run app.py --server.port 8501

# 3. Create ngrok tunnel (separate terminal)
ngrok http 8501

# 4. Access from iPhone
# Open ngrok HTTPS URL in Safari
```

#### Use Cases
- ✅ **MVP testing**: Validate detection accuracy
- ✅ **Model iteration**: Quick experiments
- ✅ **Demo**: Show stakeholders
- ❌ **Production**: Not suitable

---

### Option B: Full Client-Side (ONNX.js/TensorFlow.js)

#### Architecture Overview
```
┌─────────────────────────────────────┐
│         iPhone Safari               │
│  ┌─────────────────────────────┐   │
│  │   React/Vanilla JS App      │   │
│  │                              │   │
│  │  ┌────────────────────────┐ │   │
│  │  │  MediaStream API       │ │   │
│  │  │  (Camera Access)       │ │   │
│  │  └───────────┬────────────┘ │   │
│  │              │                │   │
│  │              ▼                │   │
│  │  ┌────────────────────────┐ │   │
│  │  │  ONNX Runtime / TF.js  │ │   │
│  │  │  (model.onnx / tfjs)   │ │   │
│  │  └───────────┬────────────┘ │   │
│  │              │                │   │
│  │              ▼                │   │
│  │  ┌────────────────────────┐ │   │
│  │  │  Canvas Renderer       │ │   │
│  │  │  (Bounding Boxes)      │ │   │
│  │  └────────────────────────┘ │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Static Hosting │
│  (Vercel/Pages) │
└─────────────────┘
```

#### Component Breakdown

**1. Frontend Application**
- React (Vite) or Vanilla JS
- MediaStream API for video
- Canvas for rendering
- Web Workers for inference
- IndexedDB for model caching

**2. Inference Engine**
- ONNX Runtime Web (preferred)
- TensorFlow.js (alternative)
- WebGL acceleration
- WASM fallback

**3. Model Conversion Pipeline**
```
best.pt (PyTorch)
    ↓
export to ONNX
    ↓
optimize (onnx-simplifier)
    ↓
quantize (INT8)
    ↓
model.onnx (< 10 MB)
```

#### Pros
✅ **True real-time**: 20-30 FPS on modern iPhones
✅ **Zero server costs**: Static hosting
✅ **Offline-capable**: PWA with Service Workers
✅ **Low latency**: < 50ms inference
✅ **Scalable**: No backend bottleneck

#### Cons
❌ **Complex model conversion**: ONNX export challenges
❌ **Browser limitations**: Memory, WebGL constraints
❌ **iOS Safari quirks**: Limited WebGL features
❌ **Larger bundle size**: 15-30 MB initial load
❌ **Debugging complexity**: Browser-specific issues

#### Performance Targets
- **Inference latency**: 30-50ms (on-device)
- **FPS**: 20-30 (iPhone 13+)
- **Model size**: 8-15 MB (quantized ONNX)
- **Initial load**: 3-5 seconds

#### Technology Stack
```yaml
Frontend:
  - React 18 + Vite 5
  - TypeScript 5.x
  - ONNX Runtime Web 1.17+
  - Canvas API + WebGL

Model Conversion:
  - Ultralytics export API
  - onnx-simplifier
  - onnxruntime-tools

Deployment:
  - Vercel / Netlify
  - GitHub Pages
  - Cloudflare Pages
```

#### Model Conversion Workflow
```python
# conversion_pipeline.py
from ultralytics import YOLO
import onnx
from onnxsim import simplify

# 1. Load trained model
model = YOLO('best.pt')

# 2. Export to ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)

# 3. Optimize ONNX
onnx_model = onnx.load('best.onnx')
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, 'best_optimized.onnx')

# 4. Quantize (optional)
# Reduces size by 4x with minimal accuracy loss
# Use onnxruntime-tools or TensorRT
```

#### Use Cases
- ✅ **Production app**: Full-featured deployment
- ✅ **High-performance**: Real-time requirements
- ✅ **Offline mode**: No internet required
- ❌ **Rapid prototyping**: Too complex for MVP

---

### Option C: Hybrid (FastAPI + React) - RECOMMENDED FOR PRODUCTION

#### Architecture Overview
```
┌─────────────────────────────────────────────┐
│              iPhone Safari                   │
│  ┌──────────────────────────────────────┐   │
│  │        React Frontend                │   │
│  │  ┌────────────────────────────────┐  │   │
│  │  │     MediaStream Video          │  │   │
│  │  │     (Camera Access)            │  │   │
│  │  └──────────────┬─────────────────┘  │   │
│  │                 │                     │   │
│  │                 ▼                     │   │
│  │  ┌────────────────────────────────┐  │   │
│  │  │   Canvas Rendering Engine      │  │   │
│  │  │   (Bounding Boxes + Score)     │  │   │
│  │  └────────────────────────────────┘  │   │
│  └────────────┬─────────────────────────┘   │
└───────────────┼─────────────────────────────┘
                │ WebSocket / REST
                ▼
┌───────────────────────────────────────────┐
│          FastAPI Backend                   │
│  ┌─────────────────────────────────────┐  │
│  │    WebSocket Connection Manager     │  │
│  └──────────────┬──────────────────────┘  │
│                 │                          │
│                 ▼                          │
│  ┌─────────────────────────────────────┐  │
│  │   YOLO11 Inference Engine           │  │
│  │   (Ultralytics + TensorRT)          │  │
│  └──────────────┬──────────────────────┘  │
│                 │                          │
│                 ▼                          │
│  ┌─────────────────────────────────────┐  │
│  │   Score Calculation Service         │  │
│  │   (Homography + Geometry)           │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│          Deployment Options                │
│  - Railway (Docker)                        │
│  - AWS EC2 + CloudFront                   │
│  - Digital Ocean + nginx                  │
└───────────────────────────────────────────┘
```

#### Component Breakdown

**1. Frontend (React + TypeScript)**
```typescript
// Core Components
- CameraStream: MediaStream management
- DetectionCanvas: Bounding box overlay
- ScoreDisplay: Real-time score updates
- WebSocketClient: Backend communication
```

**2. Backend (FastAPI + Python)**
```python
# Core Modules
- websocket_manager.py: Connection handling
- inference_service.py: YOLO11 engine
- score_calculator.py: Geometry logic
- homography_transformer.py: Perspective correction
```

**3. Communication Layer**
- **WebSocket**: Real-time frame streaming
- **REST API**: Configuration, model metadata
- **SSE (Optional)**: Server-sent events for updates

#### Pros
✅ **Real-time performance**: 15-30 FPS
✅ **Full Python backend**: Native Ultralytics
✅ **Production-ready**: Scalable architecture
✅ **Flexible deployment**: Multiple hosting options
✅ **Progressive enhancement**: Start with REST, add WebSocket

#### Cons
❌ **Infrastructure costs**: Backend hosting required
❌ **More complex**: Frontend + Backend development
❌ **Network dependency**: Requires internet
❌ **Longer development**: 1-2 weeks to production

#### Performance Targets
- **Inference latency**: 50-100ms (backend)
- **Network latency**: 50-150ms (WebSocket)
- **Total latency**: 100-250ms (end-to-end)
- **FPS**: 15-30 (real-time video)
- **Model size**: N/A (server-side)

#### Technology Stack
```yaml
Frontend:
  - React 18 + Vite 5
  - TypeScript 5.x
  - Socket.IO Client
  - Canvas API
  - TailwindCSS

Backend:
  - Python 3.10+
  - FastAPI 0.109+
  - Ultralytics 8.1+
  - python-socketio
  - OpenCV 4.8+
  - NumPy 1.24+

Deployment:
  - Docker + Docker Compose
  - Railway / Render
  - AWS EC2 + S3
  - Vercel (frontend) + Railway (backend)
```

#### Data Flow Sequence
```
1. iPhone Safari opens React app
2. MediaStream API captures video frame (30 FPS)
3. Frontend throttles to 15 FPS, sends via WebSocket
4. Backend receives frame as base64 JPEG
5. YOLO11 inference (50ms)
6. Score calculation (5ms)
7. Backend sends results via WebSocket
8. Frontend renders bounding boxes on Canvas
9. Repeat from step 2
```

#### Use Cases
- ✅ **Production deployment**: Full-featured app
- ✅ **Real-time scoring**: Tournament mode
- ✅ **Scalability**: Multiple concurrent users
- ✅ **Analytics**: Server-side logging

---

## 2. Technical Requirements Specification

### 2.1 Model Conversion Workflow

#### Option A (Streamlit): No Conversion
```bash
# Use best.pt directly
model = YOLO('best.pt')
results = model.predict(image)
```

#### Option B (Client-Side): ONNX Pipeline
```bash
# 1. Export to ONNX
python -c "from ultralytics import YOLO; \
           YOLO('best.pt').export(format='onnx', imgsz=640)"

# 2. Simplify ONNX
python -m onnxsim best.onnx best_sim.onnx

# 3. Quantize (optional)
python scripts/quantize_onnx.py --input best_sim.onnx \
                                --output best_int8.onnx

# 4. Validate conversion
python scripts/validate_onnx.py --pytorch best.pt \
                                --onnx best_sim.onnx
```

#### Option C (Hybrid): Optional TensorRT
```bash
# For GPU acceleration on server
yolo export model=best.pt format=engine device=0
```

### 2.2 Camera Stream Handling

#### MediaStream API Implementation
```javascript
// camera_manager.js
class CameraManager {
  constructor() {
    this.stream = null;
    this.videoElement = null;
  }

  async initialize() {
    const constraints = {
      video: {
        facingMode: 'environment', // Rear camera
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 }
      }
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement.srcObject = this.stream;
      return true;
    } catch (error) {
      console.error('Camera access denied:', error);
      return false;
    }
  }

  captureFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = this.videoElement.videoWidth;
    canvas.height = this.videoElement.videoHeight;

    ctx.drawImage(this.videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
  }
}
```

#### iOS Safari Considerations
```javascript
// iOS requires user interaction to start camera
// Wrap in button click handler
document.getElementById('startBtn').addEventListener('click', async () => {
  const camera = new CameraManager();
  await camera.initialize();
});

// iOS Safari doesn't autoplay without user gesture
videoElement.setAttribute('playsinline', true);
videoElement.setAttribute('autoplay', true);
videoElement.setAttribute('muted', true);
```

### 2.3 Canvas Rendering for Bounding Boxes

```javascript
// detection_renderer.js
class DetectionRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.colorMap = {
      'dart': '#00ff00',
      'bullseye': '#ff0000',
      'double': '#ffff00',
      'triple': '#00ffff'
    };
  }

  render(detections, imageWidth, imageHeight) {
    // Clear previous frame
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Scale factors
    const scaleX = this.canvas.width / imageWidth;
    const scaleY = this.canvas.height / imageHeight;

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const scaledBox = {
        x: x1 * scaleX,
        y: y1 * scaleY,
        width: (x2 - x1) * scaleX,
        height: (y2 - y1) * scaleY
      };

      // Draw bounding box
      this.ctx.strokeStyle = this.colorMap[det.class] || '#ffffff';
      this.ctx.lineWidth = 3;
      this.ctx.strokeRect(
        scaledBox.x,
        scaledBox.y,
        scaledBox.width,
        scaledBox.height
      );

      // Draw label
      const label = `${det.class} ${(det.confidence * 100).toFixed(1)}%`;
      this.ctx.fillStyle = this.colorMap[det.class] || '#ffffff';
      this.ctx.font = '16px Arial';
      this.ctx.fillText(label, scaledBox.x, scaledBox.y - 5);
    });
  }
}
```

### 2.4 Score Calculation Logic

```python
# score_calculator.py
import numpy as np
from typing import List, Tuple

class DartScoreCalculator:
    """Calculate dart scores from board detections"""

    # Dartboard geometry (pixels from board center)
    BOARD_RADIUS = 225.5  # mm converted to pixels
    BULLSEYE_RADIUS = 6.35
    OUTER_BULL_RADIUS = 15.9
    TRIPLE_INNER = 99
    TRIPLE_OUTER = 107
    DOUBLE_INNER = 162
    DOUBLE_OUTER = 170

    SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    def __init__(self, board_center: Tuple[float, float]):
        self.board_center = np.array(board_center)

    def calculate_score(self, dart_point: Tuple[float, float]) -> int:
        """Calculate score for a single dart"""
        dart = np.array(dart_point)

        # Calculate distance from center
        distance = np.linalg.norm(dart - self.board_center)

        # Calculate angle (0-360 degrees, starting at 20 segment)
        angle = np.degrees(np.arctan2(
            dart[1] - self.board_center[1],
            dart[0] - self.board_center[0]
        ))
        angle = (angle + 90) % 360  # Adjust for 20 at top

        # Determine segment
        segment_angle = 360 / 20
        segment_idx = int((angle + segment_angle/2) % 360 / segment_angle)
        segment_value = self.SEGMENTS[segment_idx]

        # Determine multiplier
        if distance <= self.BULLSEYE_RADIUS:
            return 50  # Double bull
        elif distance <= self.OUTER_BULL_RADIUS:
            return 25  # Single bull
        elif self.TRIPLE_INNER <= distance <= self.TRIPLE_OUTER:
            return segment_value * 3
        elif self.DOUBLE_INNER <= distance <= self.DOUBLE_OUTER:
            return segment_value * 2
        elif distance <= self.DOUBLE_INNER:
            return segment_value
        else:
            return 0  # Missed board

    def calculate_total_score(self, dart_points: List[Tuple[float, float]]) -> int:
        """Calculate total score for multiple darts"""
        return sum(self.calculate_score(pt) for pt in dart_points)
```

### 2.5 Homography Transformation

```python
# homography_transformer.py
import cv2
import numpy as np

class HomographyTransformer:
    """Transform perspective to top-down dartboard view"""

    def __init__(self):
        # Ideal dartboard corners (top-down view)
        self.ideal_points = np.array([
            [0, 0],           # Top-left
            [451, 0],         # Top-right (451mm = 17.75")
            [451, 451],       # Bottom-right
            [0, 451]          # Bottom-left
        ], dtype=np.float32)

    def detect_board_corners(self, image: np.ndarray) -> np.ndarray:
        """Detect dartboard corners using Hough circles or ArUco markers"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Option 1: Detect outer circle
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=150,
            maxRadius=300
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            center_x, center_y, radius = circles[0]

            # Estimate corners from circle
            corners = np.array([
                [center_x - radius, center_y - radius],
                [center_x + radius, center_y - radius],
                [center_x + radius, center_y + radius],
                [center_x - radius, center_y + radius]
            ], dtype=np.float32)

            return corners

        return None

    def compute_homography(self, source_corners: np.ndarray) -> np.ndarray:
        """Compute homography matrix"""
        H, _ = cv2.findHomography(source_corners, self.ideal_points)
        return H

    def transform_point(self, point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
        """Transform a point using homography"""
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        return tuple(transformed[0][0])

    def transform_image(self, image: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Transform entire image to top-down view"""
        warped = cv2.warpPerspective(image, H, (451, 451))
        return warped
```

---

## 3. Performance Optimization Strategies

### 3.1 Model Optimization

#### Quantization
```python
# Reduce model size by 4x with minimal accuracy loss
# INT8 quantization: 32-bit float → 8-bit integer
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', int8=True, data='dart_dataset.yaml')
```

#### Pruning (Advanced)
```python
# Remove redundant weights
import torch
from torch.nn.utils import prune

# Prune 30% of convolutional layers
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### 3.2 Frame Processing Optimization

#### Throttling Strategy
```javascript
// Limit inference to 15 FPS even if camera is 30 FPS
class FrameThrottler {
  constructor(targetFPS = 15) {
    this.interval = 1000 / targetFPS;
    this.lastFrameTime = 0;
  }

  shouldProcess(currentTime) {
    if (currentTime - this.lastFrameTime >= this.interval) {
      this.lastFrameTime = currentTime;
      return true;
    }
    return false;
  }
}
```

#### Image Preprocessing
```python
# Backend optimization
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Resize to model input size (640x640)
    resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Normalize (0-255 → 0-1)
    normalized = resized / 255.0

    # Convert to CHW format (for ONNX/PyTorch)
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)

    return batched
```

### 3.3 Network Optimization

#### WebSocket Binary Protocol
```python
# Send binary data instead of JSON
import msgpack

def serialize_detections(detections):
    # Use MessagePack for 5x smaller payload
    data = {
        'boxes': detections['boxes'].tolist(),
        'scores': detections['scores'].tolist(),
        'classes': detections['classes'].tolist()
    }
    return msgpack.packb(data)
```

#### Compression
```javascript
// Frontend decompression
import pako from 'pako';

websocket.onmessage = (event) => {
  const compressed = new Uint8Array(event.data);
  const decompressed = pako.inflate(compressed);
  const detections = msgpack.decode(decompressed);
};
```

---

## 4. Development Workflow

### 4.1 Phase 1: MVP with Streamlit (Week 1)

#### Day 1-2: Setup and Basic Detection
```bash
# Project structure
deeper_darts/
├── app.py                 # Streamlit app
├── models/
│   └── best.pt           # YOLO11 model
├── utils/
│   ├── detector.py       # YOLO wrapper
│   └── visualizer.py     # Bounding box rendering
└── requirements.txt

# Implementation checklist
- [ ] Create Streamlit UI with st.camera_input()
- [ ] Load YOLO11 model
- [ ] Implement inference pipeline
- [ ] Add bounding box overlay
- [ ] Setup ngrok tunnel
```

#### Day 3-4: Score Calculation
```bash
# Add components
utils/
├── score_calculator.py   # Scoring logic
├── homography.py         # Perspective transform
└── geometry.py           # Dartboard geometry

# Implementation checklist
- [ ] Implement dartboard geometry constants
- [ ] Add score calculation logic
- [ ] Integrate homography transformation
- [ ] Display real-time scores
```

#### Day 5: Testing and Refinement
```bash
# Testing workflow
1. Test on desktop with webcam
2. Start ngrok tunnel
3. Test on iPhone Safari
4. Fix iOS-specific issues
5. Optimize for mobile performance

# Validation checklist
- [ ] Camera works on iPhone Safari
- [ ] Detection accuracy > 85%
- [ ] Score calculation correct
- [ ] UI responsive on mobile
```

### 4.2 Phase 2: Client-Side ONNX (Week 2-3)

#### Week 2: Model Conversion
```bash
# Setup conversion pipeline
scripts/
├── export_onnx.py        # PyTorch → ONNX
├── optimize_onnx.py      # Simplify + quantize
└── validate_onnx.py      # Accuracy validation

# Conversion workflow
python scripts/export_onnx.py --model best.pt --imgsz 640
python scripts/optimize_onnx.py --input best.onnx
python scripts/validate_onnx.py --pytorch best.pt --onnx best_opt.onnx

# Target: < 10 MB model, < 2% accuracy loss
```

#### Week 3: Frontend Implementation
```bash
# React app structure
frontend/
├── src/
│   ├── components/
│   │   ├── CameraStream.tsx
│   │   ├── DetectionCanvas.tsx
│   │   └── ScoreDisplay.tsx
│   ├── services/
│   │   ├── ONNXInference.ts
│   │   └── ScoreCalculator.ts
│   └── utils/
│       └── imageProcessing.ts
└── public/
    └── models/
        └── best_opt.onnx

# Implementation steps
1. Setup Vite + React + TypeScript
2. Integrate ONNX Runtime Web
3. Implement MediaStream capture
4. Add inference worker
5. Render detections on Canvas
```

### 4.3 Phase 3: Hybrid Production (Week 4-5)

#### Week 4: Backend Development
```bash
# FastAPI structure
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── websocket.py         # WebSocket manager
│   ├── inference.py         # YOLO service
│   ├── scoring.py           # Score calculator
│   └── models.py            # Pydantic schemas
├── tests/
└── Dockerfile

# Implementation checklist
- [ ] Setup FastAPI with WebSocket
- [ ] Implement connection manager
- [ ] Add YOLO inference service
- [ ] Integrate score calculator
- [ ] Write unit tests
```

#### Week 5: Frontend Integration
```bash
# Full-stack integration
1. Deploy backend to Railway
2. Update frontend WebSocket client
3. Add error handling and reconnection
4. Implement loading states
5. Add analytics tracking

# Deployment checklist
- [ ] Backend deployed with HTTPS
- [ ] Frontend on Vercel
- [ ] WebSocket connections stable
- [ ] Error handling robust
- [ ] Mobile testing complete
```

### 4.4 Testing on iPhone Safari

#### Setup Steps
```bash
# 1. Enable Safari debugging (Mac + iPhone)
# iPhone: Settings → Safari → Advanced → Web Inspector (ON)
# Mac: Safari → Preferences → Advanced → Show Develop menu (ON)

# 2. Connect iPhone via USB
# 3. Open app on iPhone Safari
# 4. Mac Safari → Develop → [Your iPhone] → [Page]

# 5. Debug console in Mac Safari shows iOS logs
```

#### Common iOS Issues and Fixes

**Issue 1: Camera Permission Denied**
```javascript
// Solution: Must use HTTPS (not HTTP)
// Use ngrok for local testing
// Add error handling
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  alert('Camera API not supported. Use HTTPS.');
}
```

**Issue 2: Video Won't Autoplay**
```html
<!-- Solution: Add required attributes -->
<video
  autoplay
  playsinline
  muted
  ref={videoRef}
/>
```

**Issue 3: WebGL Not Available**
```javascript
// Solution: Check and fallback to WASM
const gl = canvas.getContext('webgl2');
if (!gl) {
  console.warn('WebGL not available, using WASM backend');
  ort.env.wasm.numThreads = 2;
}
```

**Issue 4: Memory Leaks on Long Sessions**
```javascript
// Solution: Dispose tensors and sessions
useEffect(() => {
  return () => {
    if (inferenceSession) {
      inferenceSession.release();
    }
  };
}, [inferenceSession]);
```

---

## 5. Deployment and Hosting

### 5.1 Option A: Streamlit Deployment

#### Local with ngrok (Development)
```bash
# Terminal 1: Start Streamlit
streamlit run app.py --server.port 8501

# Terminal 2: Start ngrok
ngrok http 8501 --region us

# Output: https://xxxx-xxx-xxx-xxx.ngrok-free.app
# Access from iPhone Safari
```

#### Streamlit Community Cloud (Free)
```yaml
# .streamlit/config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

# Deployment steps
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect repository
4. Deploy app
5. Get public URL: https://yourapp.streamlit.app
```

**Pros**: Free, automatic SSL
**Cons**: Cold starts, limited resources

### 5.2 Option B: Client-Side Deployment

#### Vercel (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy frontend
cd frontend
vercel --prod

# Output: https://your-app.vercel.app
```

#### GitHub Pages
```bash
# Build frontend
npm run build

# Deploy to gh-pages
npm install -g gh-pages
gh-pages -d dist

# Access: https://username.github.io/repo-name
```

#### Cloudflare Pages
```bash
# Push to GitHub
git push origin main

# Connect in Cloudflare Dashboard
1. Go to Pages
2. Connect repository
3. Set build command: npm run build
4. Set output directory: dist
5. Deploy

# Output: https://your-app.pages.dev
```

**Pros**: Free, global CDN, instant deploys
**Cons**: Client-side only

### 5.3 Option C: Hybrid Deployment

#### Backend: Railway (Recommended)
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# railway.toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

Deployment:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up

# Output: https://your-app.railway.app
```

**Cost**: $5-20/month
**Pros**: Automatic SSL, easy scaling, persistent storage

#### Frontend: Vercel + Backend: Railway
```typescript
// frontend/.env.production
VITE_BACKEND_URL=https://your-app.railway.app
VITE_WS_URL=wss://your-app.railway.app
```

```bash
# Deploy backend
cd backend
railway up

# Deploy frontend
cd frontend
vercel --prod
```

**Total Cost**: $5-20/month
**Pros**: Best performance, scalable

#### Alternative: AWS EC2 + S3
```bash
# Backend on EC2 (t3.small)
# Frontend on S3 + CloudFront

# Cost: $10-30/month
# Pros: Full control, GPU options
# Cons: More complex setup
```

---

## 6. Architecture Decision Record (ADR)

### ADR-001: Choose Hybrid Architecture for Production

**Status**: ACCEPTED

**Context**:
We need to build a web-based YOLO11 dart detection system accessible from iPhone Safari. Three architectures were evaluated:
1. Streamlit + ngrok (MVP)
2. Full client-side (ONNX.js)
3. Hybrid (FastAPI + React)

**Decision**:
- **Phase 1 (MVP)**: Use Streamlit + ngrok for rapid prototyping (2-3 days)
- **Phase 2 (Production)**: Transition to Hybrid architecture (FastAPI + React)

**Rationale**:

**Why Streamlit for MVP**:
1. Fastest time-to-market (2-3 hours to working prototype)
2. No model conversion required (use best.pt directly)
3. Native Python ecosystem (Ultralytics API)
4. Easy debugging and iteration
5. Low risk for initial testing

**Why Hybrid for Production**:
1. Real-time performance (15-30 FPS vs 1-3 FPS)
2. Better user experience (continuous video vs frame capture)
3. Scalability (multiple concurrent users)
4. Professional deployment options
5. Balance between performance and development cost

**Why NOT Full Client-Side**:
1. Complex model conversion (ONNX export challenges)
2. iOS Safari WebGL limitations
3. Difficult debugging (browser-specific issues)
4. Higher risk for first production release
5. Can be added later as Progressive Web App

**Consequences**:

**Positive**:
- Fast MVP validation (1 week)
- Smooth transition to production (4 weeks total)
- Backend flexibility for analytics and logging
- Can add client-side inference later for offline mode

**Negative**:
- Requires backend hosting ($5-20/month)
- Network dependency (no offline mode initially)
- Two deployment environments to manage

**Risks and Mitigations**:
1. **Risk**: Backend costs exceed budget
   **Mitigation**: Start with Railway ($5/mo), scale to AWS if needed

2. **Risk**: WebSocket latency too high
   **Mitigation**: Use regional deployment, add CDN, implement adaptive quality

3. **Risk**: iPhone Safari compatibility issues
   **Mitigation**: Extensive mobile testing in Phase 1 with Streamlit

**Performance Targets**:
- MVP (Streamlit): 1-3 FPS, 500-1000ms latency ✓ Acceptable for testing
- Production (Hybrid): 15-30 FPS, 100-250ms latency ✓ Real-time experience

**Timeline**:
- Week 1: Streamlit MVP + iPhone testing
- Week 2-3: Model optimization + conversion experiments
- Week 4-5: Hybrid production implementation
- Week 6: Deployment + user testing

**Review Date**: End of Week 1 (after MVP validation)

---

## 7. Component Interaction Diagrams

### 7.1 Streamlit MVP Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                    iPhone Safari                         │
│                                                           │
│  1. User clicks "Capture Frame"                          │
│  2. st.camera_input() triggers                           │
│  3. Image uploaded to Streamlit server                   │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTPS (ngrok)
                         │ POST /upload
                         ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit Backend (Python)                   │
│                                                           │
│  4. Receive image bytes                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  image = Image.open(BytesIO(uploaded_file))       │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│  5. YOLO11 Inference                                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  model = YOLO('best.pt')                          │  │
│  │  results = model.predict(image, conf=0.5)         │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│  6. Parse Detections                                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  boxes = results[0].boxes.xyxy                    │  │
│  │  scores = results[0].boxes.conf                   │  │
│  │  classes = results[0].boxes.cls                   │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│  7. Calculate Scores                                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  dart_points = extract_dart_tips(boxes)           │  │
│  │  scores = [calc_score(pt) for pt in dart_points] │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│  8. Render Visualization                                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │  annotated = draw_boxes(image, boxes, scores)    │  │
│  │  st.image(annotated)                              │  │
│  │  st.metric("Total Score", sum(scores))           │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP Response
                         │ HTML + Image
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    iPhone Safari                         │
│                                                           │
│  9. Display annotated image                              │
│  10. Show score breakdown                                │
│  11. "Capture Again" button enabled                      │
└──────────────────────────────────────────────────────────┘

Latency Breakdown:
- Upload: 200-400ms (image size ~500 KB)
- Inference: 50-100ms (YOLO11)
- Score calc: 5-10ms
- Response: 200-400ms (annotated image)
Total: 455-910ms per frame
```

### 7.2 Hybrid Architecture Real-Time Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         iPhone Safari                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  React Frontend                           │  │
│  │                                                            │  │
│  │  1. MediaStream captures frame (30 FPS)                  │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │  const stream = await navigator.mediaDevices     │    │  │
│  │  │    .getUserMedia({ video: true })                │    │  │
│  │  └────────────────────┬─────────────────────────────┘    │  │
│  │                       │                                    │  │
│  │                       ▼                                    │  │
│  │  2. Throttle to 15 FPS                                   │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │  if (now - lastTime > 66ms) {                    │    │  │
│  │  │    processFrame()                                 │    │  │
│  │  │  }                                                │    │  │
│  │  └────────────────────┬─────────────────────────────┘    │  │
│  │                       │                                    │  │
│  │                       ▼                                    │  │
│  │  3. Capture & Encode                                     │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │  canvas.drawImage(video, 0, 0)                   │    │  │
│  │  │  const jpeg = canvas.toDataURL('image/jpeg', 0.7)│    │  │
│  │  └────────────────────┬─────────────────────────────┘    │  │
│  │                       │                                    │  │
│  │                       ▼                                    │  │
│  │  4. Send via WebSocket                                   │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │  ws.send(JSON.stringify({                        │    │  │
│  │  │    type: 'frame',                                 │    │  │
│  │  │    data: jpeg,                                    │    │  │
│  │  │    timestamp: Date.now()                          │    │  │
│  │  │  }))                                              │    │  │
│  │  └────────────────────┬─────────────────────────────┘    │  │
│  └───────────────────────┼────────────────────────────────┘  │
└────────────────────────┬─┼────────────────────────────────────┘
                         │ │ WebSocket (WSS)
                         │ │ 50-100ms latency
                         │ │
┌────────────────────────▼─▼────────────────────────────────────┐
│                  FastAPI Backend                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │          WebSocket Connection Manager                    │ │
│  │                                                           │ │
│  │  5. Receive frame                                        │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  async def receive_frame(websocket, data):      │    │ │
│  │  │    image_bytes = base64.decode(data['data'])    │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  └───────────────────────┼─────────────────────────────────┘ │
│                          │                                     │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            YOLO11 Inference Service                      │ │
│  │                                                           │ │
│  │  6. Preprocess                                           │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  image = cv2.imdecode(image_bytes)              │    │ │
│  │  │  resized = cv2.resize(image, (640, 640))        │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  │                       │                                   │ │
│  │                       ▼                                   │ │
│  │  7. Inference (50ms)                                     │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  results = model.predict(resized, conf=0.5)     │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  └───────────────────────┼─────────────────────────────────┘ │
│                          │                                     │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Score Calculation Service                        │ │
│  │                                                           │ │
│  │  8. Extract dart points & calculate scores (5ms)         │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  dart_tips = extract_tips(boxes)                │    │ │
│  │  │  scores = [score_calc.calc(pt) for pt in tips] │    │ │
│  │  │  total_score = sum(scores)                      │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  └───────────────────────┼─────────────────────────────────┘ │
│                          │                                     │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │               Response Formatter                         │ │
│  │                                                           │ │
│  │  9. Send results                                         │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  await websocket.send_json({                    │    │ │
│  │  │    'detections': boxes.tolist(),                │    │ │
│  │  │    'scores': scores,                            │    │ │
│  │  │    'total': total_score,                        │    │ │
│  │  │    'timestamp': time.time()                     │    │ │
│  │  │  })                                             │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  └───────────────────────┼─────────────────────────────────┘ │
└────────────────────────┬─┼────────────────────────────────────┘
                         │ │ WebSocket Response
                         │ │ 50-100ms latency
                         │ │
┌────────────────────────▼─▼────────────────────────────────────┐
│                     iPhone Safari                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 React Frontend                           │ │
│  │                                                           │ │
│  │  10. Receive results                                     │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  ws.onmessage = (event) => {                    │    │ │
│  │  │    const results = JSON.parse(event.data)       │    │ │
│  │  │    renderDetections(results)                    │    │ │
│  │  │  }                                              │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  │                       │                                   │ │
│  │                       ▼                                   │ │
│  │  11. Render on Canvas (< 10ms)                          │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  ctx.clearRect(0, 0, canvas.width, canvas.height)   │ │
│  │  │  results.detections.forEach(box => {            │    │ │
│  │  │    ctx.strokeRect(box.x, box.y, box.w, box.h)  │    │ │
│  │  │  })                                             │    │ │
│  │  └────────────────────┬────────────────────────────┘    │ │
│  │                       │                                   │ │
│  │                       ▼                                   │ │
│  │  12. Update score display                                │ │
│  │  ┌─────────────────────────────────────────────────┐    │ │
│  │  │  setTotalScore(results.total)                   │    │ │
│  │  │  setScoreBreakdown(results.scores)              │    │ │
│  │  └─────────────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘

Total Latency: 115-215ms per frame
- Capture & Encode: 10-15ms
- WebSocket Send: 50-100ms
- Inference: 50-100ms
- WebSocket Receive: 50-100ms
- Render: 5-10ms
- Achieves: 15-30 FPS real-time
```

### 7.3 Client-Side ONNX Flow (Future Enhancement)

```
┌─────────────────────────────────────────────────────────┐
│                   iPhone Safari                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              React App (PWA)                      │  │
│  │                                                    │  │
│  │  1. First Load: Download ONNX model (once)       │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  fetch('/models/best_opt.onnx')            │  │  │
│  │  │  cache in IndexedDB for offline use        │  │  │
│  │  └────────────────┬───────────────────────────┘  │  │
│  │                   │                               │  │
│  │                   ▼                               │  │
│  │  2. Initialize ONNX Runtime (Web Worker)         │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  const session = await ort.InferenceSession  │  │
│  │  │    .create('best_opt.onnx', {             │  │  │
│  │  │      executionProviders: ['webgl', 'wasm']│  │  │
│  │  │    })                                      │  │  │
│  │  └────────────────┬───────────────────────────┘  │  │
│  │                   │                               │  │
│  │                   ▼                               │  │
│  │  3. Start camera stream                          │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  const stream = await navigator.mediaDevices  │  │
│  │  │    .getUserMedia({ video: true })         │  │  │
│  │  └────────────────┬───────────────────────────┘  │  │
│  │                   │                               │  │
│  │                   ▼                               │  │
│  │  4. Continuous inference loop (30 FPS)           │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  requestAnimationFrame(() => {             │  │  │
│  │  │    // a. Capture frame                     │  │  │
│  │  │    canvas.drawImage(video, 0, 0)          │  │  │
│  │  │                                            │  │  │
│  │  │    // b. Preprocess (10ms)                │  │  │
│  │  │    const tensor = preprocessImage(canvas) │  │  │
│  │  │                                            │  │  │
│  │  │    // c. Inference in worker (30-50ms)    │  │  │
│  │  │    worker.postMessage({ tensor })         │  │  │
│  │  │  })                                        │  │  │
│  │  └────────────────┬───────────────────────────┘  │  │
│  │                   │                               │  │
│  │                   ▼                               │  │
│  │  5. Web Worker processes inference               │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  onmessage = async (e) => {                │  │  │
│  │  │    const results = await session.run({    │  │  │
│  │  │      images: e.data.tensor               │  │  │
│  │  │    })                                     │  │  │
│  │  │                                           │  │  │
│  │  │    // Post-process detections            │  │  │
│  │  │    const boxes = parseOutputs(results)   │  │  │
│  │  │    postMessage({ boxes })                │  │  │
│  │  │  }                                        │  │  │
│  │  └────────────────┬───────────────────────────┘  │  │
│  │                   │                               │  │
│  │                   ▼                               │  │
│  │  6. Main thread receives results                 │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  worker.onmessage = (e) => {               │  │  │
│  │  │    const { boxes } = e.data                │  │  │
│  │  │                                            │  │  │
│  │  │    // Calculate scores locally            │  │  │
│  │  │    const scores = boxes.map(calcScore)    │  │  │
│  │  │                                            │  │  │
│  │  │    // Render on canvas                    │  │  │
│  │  │    renderDetections(boxes, scores)        │  │  │
│  │  │  }                                         │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Total Latency: 40-70ms per frame
- Capture: 5ms
- Preprocess: 10ms
- Inference: 30-50ms (WebGL) or 50-80ms (WASM)
- Score calc: 5ms
- Render: 5ms
- Achieves: 20-30 FPS (no network!)
```

---

## 8. Memory System Storage

