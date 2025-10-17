# FPS Optimization Research: Streamlit + YOLO11 + OpenCV Real-Time Inference

**Research Date**: 2025-10-17
**Target System**: MacBook Pro M2, iPhone 15 Pro (1920x1080) via Continuity Camera
**Current Issue**: Low FPS, stuttering video stream
**Current Implementation**: Full frame processing every loop iteration (800x800 YOLO input)

---

## Executive Summary

Current bottlenecks identified from code analysis:
1. **Full resolution processing** (1920x1080 → 800x800) every frame
2. **Synchronous inference** blocking the main loop
3. **Streamlit re-rendering** entire UI on every frame update
4. **No frame skipping** or detection caching
5. **Buffer accumulation** in VideoCapture (default buffer=4)
6. **Heavy score calculation** on every frame with detections

**Expected Current FPS**: 8-12 FPS
**Target FPS**: 25-30 FPS
**Achievable with optimizations**: 30-45 FPS

---

## 1. STREAMLIT PERFORMANCE OPTIMIZATION

### 1.1 Replace st.image() with st.empty() Reuse (HIGH IMPACT)
**Expected Improvement**: +5-8 FPS
**Difficulty**: Easy

**Problem**: `st.image()` creates new DOM elements on each call, causing re-renders.

**Current Code** (Lines 490-493):
```python
with frame_placeholder:
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_frame_rgb, use_column_width=True)
```

**Optimized Code**:
```python
# In run_detection(), before loop:
frame_placeholder = st.empty()

# In loop:
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
frame_placeholder.image(annotated_frame_rgb, use_column_width=True)
```

**Note**: Already partially implemented, but ensure ALL UI elements use `.empty()` pattern.

---

### 1.2 Batch UI Updates with st.container() (MEDIUM IMPACT)
**Expected Improvement**: +2-3 FPS
**Difficulty**: Easy

**Problem**: Multiple `st.` calls trigger separate re-renders.

**Optimized Code**:
```python
# Update scores only every Nth frame or when changed
if frame_count % 3 == 0 or scores_changed:
    with scores_container:
        scores_container.empty()
        for i, score in enumerate(dart_scores, 1):
            st.markdown(f"🎯 **Dart {i}:** `{score}`")
```

---

### 1.3 Aggressive Caching with @st.cache_resource (MEDIUM IMPACT)
**Expected Improvement**: +3-5 FPS (first load improvement)
**Difficulty**: Easy

**Current Code** (Lines 391-394):
```python
with st.spinner("Loading YOLO11 model..."):
    model = YOLO(config['model_path'])
```

**Optimized Code**:
```python
@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLO model with caching to avoid reloads."""
    return YOLO(model_path)

# In run_detection():
model = load_yolo_model(config['model_path'])
```

**Additional Caching**:
```python
@st.cache_data
def get_board_geometry():
    """Cache dartboard geometry calculations."""
    return BOARD_DICT, BOARD_CONFIG

@st.cache_resource
def get_opencv_capture(camera_idx: int):
    """Cache VideoCapture object across reruns."""
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    return cap
```

---

### 1.4 Disable Streamlit's Auto-Rerun (HIGH IMPACT)
**Expected Improvement**: +4-6 FPS
**Difficulty**: Easy

**Add to config**:
```python
# .streamlit/config.toml
[server]
runOnSave = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[client]
toolbarMode = "minimal"
showErrorDetails = false
```

---

## 2. OPENCV VIDEO CAPTURE OPTIMIZATION

### 2.1 Reduce Buffer Size (HIGH IMPACT)
**Expected Improvement**: +8-12 FPS (reduces latency)
**Difficulty**: Easy

**Problem**: Default buffer (4 frames) accumulates old frames, causing lag.

**Current Code** (Lines 408-409):
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

**Optimized Code**:
```python
cap = cv2.VideoCapture(config['selected_camera'])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL: Minimize buffer lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from camera
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG codec
```

**Alternative - Aggressive Buffer Flush**:
```python
# In main loop, flush old frames:
for _ in range(2):  # Skip 2 frames to get latest
    cap.grab()
ret, frame = cap.retrieve()
```

---

### 2.2 Downscale Frame Before Processing (HIGH IMPACT)
**Expected Improvement**: +10-15 FPS
**Difficulty**: Easy

**Problem**: Processing full 1920x1080 frame is unnecessary.

**Optimized Code**:
```python
# Read frame
ret, frame = cap.read()
if not ret:
    break

# Downscale for processing (maintain aspect ratio)
target_width = 960  # Half resolution
scale = target_width / frame.shape[1]
frame_resized = cv2.resize(frame, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR)

# Run inference on smaller frame
results = model.predict(
    source=frame_resized,
    conf=config['conf_threshold'],
    iou=config['iou_threshold'],
    imgsz=config['img_size'],
    verbose=False
)

# Scale detection coordinates back to original if needed
# boxes *= (1/scale)
```

---

### 2.3 Threading for Video Capture (MEDIUM IMPACT)
**Expected Improvement**: +5-8 FPS
**Difficulty**: Medium

**Problem**: `cap.read()` blocks the main loop.

**Optimized Code**:
```python
import threading
import queue

class VideoStreamThread:
    def __init__(self, camera_idx):
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if ret:
                    if self.q.full():
                        self.q.get()  # Drop old frame
                    self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# Usage:
video_stream = VideoStreamThread(config['selected_camera']).start()
time.sleep(1.0)  # Warm up

while st.session_state.running:
    frame = video_stream.read()
    if frame is None:
        continue
    # Process frame...
```

---

## 3. YOLO INFERENCE OPTIMIZATION

### 3.1 Frame Skipping with Detection Caching (HIGH IMPACT)
**Expected Improvement**: +15-20 FPS
**Difficulty**: Easy

**Problem**: Running inference every frame is overkill.

**Optimized Code**:
```python
# Global state
inference_every_n_frames = 3  # Run inference every 3rd frame
frame_count = 0
cached_detections = None
cached_scores = []

while st.session_state.running:
    ret, frame = cap.read()
    frame_count += 1

    # Only run inference every Nth frame
    if frame_count % inference_every_n_frames == 0:
        results = model.predict(
            source=frame,
            conf=config['conf_threshold'],
            iou=config['iou_threshold'],
            imgsz=config['img_size'],
            verbose=False
        )
        cached_detections = results[0].boxes

        # Recalculate scores only when new detections
        if len(cached_detections) > 0:
            # ... score calculation logic ...
            cached_scores = dart_scores

    # Reuse cached detections for display
    if cached_detections is not None:
        # Annotate with cached detections
        annotated_frame = draw_detections(frame, cached_detections)
```

---

### 3.2 Use FP16 Quantization for M2 GPU (MEDIUM IMPACT)
**Expected Improvement**: +5-8 FPS
**Difficulty**: Easy

**Problem**: Default FP32 precision is slower.

**Optimized Code**:
```python
# Export model to FP16
model.export(format='torchscript', half=True)

# Or load with half precision
model = YOLO(config['model_path'])
model.to('mps')  # Use Metal Performance Shaders on M2
model.half()  # Convert to FP16
```

---

### 3.3 Reduce Input Resolution Dynamically (HIGH IMPACT)
**Expected Improvement**: +10-15 FPS
**Difficulty**: Easy

**Problem**: 800x800 input is slower than necessary.

**Current Code** (Line 318-323):
```python
img_size = st.sidebar.selectbox(
    "Image Size",
    [640, 800],
    index=1,
    help="Input size for model (larger = more accurate but slower)"
)
```

**Optimized Code**:
```python
# Add dynamic resolution scaling
img_size = st.sidebar.selectbox(
    "Image Size",
    [320, 480, 640, 800],  # Add smaller sizes
    index=2,  # Default to 640
    help="Lower = faster, Higher = more accurate"
)

# Or adaptive based on FPS
if avg_fps < 20:
    img_size = 480  # Reduce automatically
elif avg_fps > 28:
    img_size = 640  # Increase for accuracy
```

---

### 3.4 Batch Processing (for multiple frames) - NOT RECOMMENDED
**Expected Improvement**: N/A (doesn't apply to real-time)
**Difficulty**: Hard

**Note**: Batch processing is for offline processing only. For real-time streams, frame skipping is better.

---

### 3.5 Model Compilation with TorchScript (MEDIUM IMPACT)
**Expected Improvement**: +3-5 FPS
**Difficulty**: Medium

**Optimized Code**:
```python
# One-time model compilation
import torch
model = YOLO(config['model_path'])
model.to('mps')  # Use Metal backend

# Compile model for faster inference
compiled_model = torch.compile(model.model, mode='reduce-overhead')
```

---

## 4. PROCESSING PIPELINE OPTIMIZATION

### 4.1 Decouple Capture → Inference → Display (HIGH IMPACT)
**Expected Improvement**: +10-15 FPS
**Difficulty**: Hard

**Problem**: Synchronous pipeline blocks each step.

**Optimized Code**:
```python
import multiprocessing as mp
from queue import Queue

def capture_worker(cap, frame_queue):
    """Dedicated thread for frame capture."""
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Drop old frame
            frame_queue.put(frame)

def inference_worker(frame_queue, result_queue, model, config):
    """Dedicated process for YOLO inference."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model.predict(
                source=frame,
                conf=config['conf_threshold'],
                imgsz=config['img_size'],
                verbose=False
            )
            result_queue.put((frame, results))

# Main loop only handles display
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)

capture_thread = threading.Thread(target=capture_worker, args=(cap, frame_queue))
inference_thread = threading.Thread(target=inference_worker, args=(frame_queue, result_queue, model, config))

capture_thread.start()
inference_thread.start()

while st.session_state.running:
    if not result_queue.empty():
        frame, results = result_queue.get()
        # Display only
        frame_placeholder.image(frame, use_column_width=True)
```

---

### 4.2 Async Score Calculation (MEDIUM IMPACT)
**Expected Improvement**: +3-5 FPS
**Difficulty**: Medium

**Problem**: Score calculation blocks rendering.

**Optimized Code**:
```python
import concurrent.futures

# Create thread pool
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Submit score calculation asynchronously
if num_calibration >= 4 and num_darts > 0:
    future_scores = executor.submit(get_dart_scores, xy_array, False)
    # Continue with display...

    # Get scores when ready (non-blocking)
    try:
        dart_scores = future_scores.result(timeout=0.01)
    except concurrent.futures.TimeoutError:
        dart_scores = []  # Use previous scores
```

---

### 4.3 Optimize Score Calculation Algorithm (MEDIUM IMPACT)
**Expected Improvement**: +2-4 FPS
**Difficulty**: Medium

**Problem**: Unnecessary matrix operations every frame.

**Current Code** (Lines 155-198):
```python
def get_dart_scores(xy, numeric=False):
    # ... perspective transform ...
    xy_transformed, _ = transform(xy.copy(), angle=0)
    # ... score calculation ...
```

**Optimized Code**:
```python
# Cache transformation matrix
@st.cache_data
def get_transform_matrix(calibration_points):
    """Cache perspective transform matrix."""
    _, M = transform(calibration_points.copy(), angle=0)
    return M

# Reuse cached matrix
if 'transform_matrix' not in st.session_state:
    st.session_state.transform_matrix = None

# Only recalculate if calibration changes
if calibration_changed:
    st.session_state.transform_matrix = get_transform_matrix(xy[:4])

# Apply cached transform
if st.session_state.transform_matrix is not None:
    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1)
    xyz_dst = np.matmul(st.session_state.transform_matrix, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]
```

---

## 5. PROFILING AND BOTTLENECK IDENTIFICATION

### 5.1 Add Timing Instrumentation (HIGH IMPACT for debugging)
**Expected Improvement**: N/A (diagnostic tool)
**Difficulty**: Easy

**Optimized Code**:
```python
import time

# Timing dictionary
timings = {
    'capture': [],
    'inference': [],
    'scoring': [],
    'display': [],
    'total': []
}

while st.session_state.running:
    t_start = time.perf_counter()

    # Capture
    t0 = time.perf_counter()
    ret, frame = cap.read()
    timings['capture'].append(time.perf_counter() - t0)

    # Inference
    t1 = time.perf_counter()
    results = model.predict(source=frame, ...)
    timings['inference'].append(time.perf_counter() - t1)

    # Scoring
    t2 = time.perf_counter()
    dart_scores = get_dart_scores(xy_array, numeric=False)
    timings['scoring'].append(time.perf_counter() - t2)

    # Display
    t3 = time.perf_counter()
    frame_placeholder.image(annotated_frame_rgb, ...)
    timings['display'].append(time.perf_counter() - t3)

    timings['total'].append(time.perf_counter() - t_start)

    # Report every 30 frames
    if len(timings['total']) >= 30:
        with st.sidebar.expander("⏱️ Performance Profile"):
            st.write(f"Capture: {np.mean(timings['capture'])*1000:.1f}ms")
            st.write(f"Inference: {np.mean(timings['inference'])*1000:.1f}ms")
            st.write(f"Scoring: {np.mean(timings['scoring'])*1000:.1f}ms")
            st.write(f"Display: {np.mean(timings['display'])*1000:.1f}ms")
            st.write(f"Total: {np.mean(timings['total'])*1000:.1f}ms")
            st.write(f"Max FPS: {1/np.mean(timings['total']):.1f}")
        # Reset
        timings = {k: [] for k in timings.keys()}
```

---

### 5.2 Memory Profiling (LOW IMPACT)
**Expected Improvement**: N/A (diagnostic)
**Difficulty**: Easy

**Optimized Code**:
```python
import tracemalloc

tracemalloc.start()

# In loop, periodically check:
if frame_count % 100 == 0:
    current, peak = tracemalloc.get_traced_memory()
    st.sidebar.text(f"Memory: {current / 1024**2:.1f} MB (peak: {peak / 1024**2:.1f} MB)")
```

---

## 6. ADDITIONAL OPTIMIZATION TECHNIQUES

### 6.1 Use OpenCV's CUDA/GPU Acceleration (if available)
**Expected Improvement**: +20-30 FPS (requires CUDA GPU)
**Difficulty**: Hard
**Status**: Not applicable for M2 Mac (no CUDA)

**Alternative for M2**: Use Metal backend via PyTorch MPS
```python
import torch
device = torch.device('mps')  # Metal Performance Shaders
model.to(device)
```

---

### 6.2 Reduce Color Conversion Overhead (LOW IMPACT)
**Expected Improvement**: +1-2 FPS
**Difficulty**: Easy

**Current Code** (Line 492):
```python
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
```

**Optimized Code**:
```python
# Reuse buffer to avoid allocation
if 'rgb_buffer' not in st.session_state:
    st.session_state.rgb_buffer = np.empty_like(annotated_frame)

cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB, dst=st.session_state.rgb_buffer)
frame_placeholder.image(st.session_state.rgb_buffer, use_column_width=True)
```

---

### 6.3 Disable Verbose Logging (LOW IMPACT)
**Expected Improvement**: +1-2 FPS
**Difficulty**: Easy

**Current Code** (Line 445):
```python
results = model.predict(
    source=frame,
    verbose=False  # Already optimized
)
```

**Additional**:
```python
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
```

---

### 6.4 Use Lightweight Annotation (MEDIUM IMPACT)
**Expected Improvement**: +3-5 FPS
**Difficulty**: Easy

**Problem**: `results[0].plot()` is slow (antialiasing, thick lines).

**Current Code** (Lines 483-487):
```python
annotated_frame = results[0].plot(
    conf=config['show_conf'],
    labels=config['show_labels'],
    line_width=2
)
```

**Optimized Code**:
```python
def fast_draw_detections(frame, boxes, config):
    """Lightweight detection drawing without antialiasing."""
    for box in boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        # Simple rectangle (no antialiasing)
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                     (0, 255, 0), 1, cv2.LINE_8)

        if config['show_labels']:
            label = f"{CLASS_NAMES[cls]}"
            if config['show_conf']:
                conf = box.conf[0]
                label += f" {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return frame

# Use instead of results[0].plot()
annotated_frame = fast_draw_detections(frame.copy(), results[0].boxes, config)
```

---

## 7. IMPLEMENTATION PRIORITY MATRIX

| Technique | Impact | Difficulty | Priority | Expected FPS Gain |
|-----------|--------|------------|----------|-------------------|
| **1. Reduce buffer size (CAP_PROP_BUFFERSIZE=1)** | HIGH | Easy | 🔥 1 | +8-12 FPS |
| **2. Frame skipping (inference every 3 frames)** | HIGH | Easy | 🔥 2 | +15-20 FPS |
| **3. Downscale input resolution (960px width)** | HIGH | Easy | 🔥 3 | +10-15 FPS |
| **4. Use FP16 + MPS backend** | MEDIUM | Easy | 🔥 4 | +5-8 FPS |
| **5. Reduce YOLO input size (640 → 480)** | HIGH | Easy | 🔥 5 | +10-15 FPS |
| **6. Threading for video capture** | MEDIUM | Medium | ⚡ 6 | +5-8 FPS |
| **7. Cache st.empty() UI elements** | HIGH | Easy | ⚡ 7 | +5-8 FPS |
| **8. Lightweight annotation (no plot())** | MEDIUM | Easy | ⚡ 8 | +3-5 FPS |
| **9. Cache transformation matrix** | MEDIUM | Medium | ⚡ 9 | +2-4 FPS |
| **10. Decouple pipeline (threads/processes)** | HIGH | Hard | 🔄 10 | +10-15 FPS |
| **11. Batch UI updates** | MEDIUM | Easy | 🔄 11 | +2-3 FPS |
| **12. Disable Streamlit auto-rerun** | HIGH | Easy | 🔄 12 | +4-6 FPS |
| **13. Async score calculation** | MEDIUM | Medium | 🔄 13 | +3-5 FPS |
| **14. Model caching (@st.cache_resource)** | MEDIUM | Easy | 📌 14 | +3-5 FPS |
| **15. Reduce color conversion overhead** | LOW | Easy | 📌 15 | +1-2 FPS |

**Legend**:
- 🔥 = Implement FIRST (high impact, low effort)
- ⚡ = Implement SECOND (good ROI)
- 🔄 = Implement THIRD (requires more work)
- 📌 = Nice-to-have

---

## 8. RECOMMENDED IMPLEMENTATION PLAN

### Phase 1: Quick Wins (1-2 hours) - Expected +30-40 FPS
```python
# 1. Set buffer size to 1
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 2. Frame skipping (inference every 3 frames)
if frame_count % 3 == 0:
    results = model.predict(...)

# 3. Downscale frames before inference
frame_resized = cv2.resize(frame, (960, 540))

# 4. Use MPS backend
model.to('mps')
model.half()

# 5. Reduce YOLO input size
imgsz=480  # Instead of 800
```

### Phase 2: Medium Effort (3-5 hours) - Expected +15-20 FPS
```python
# 6. Threading for video capture
video_stream = VideoStreamThread(camera_idx).start()

# 7. Lightweight annotation
annotated_frame = fast_draw_detections(frame, boxes, config)

# 8. Cache transformation matrix
st.session_state.transform_matrix = get_transform_matrix(calibration)

# 9. Add profiling
timings = profile_pipeline()
```

### Phase 3: Advanced (1-2 days) - Expected +10-15 FPS
```python
# 10. Decouple pipeline with multiprocessing
capture_thread, inference_thread, display_thread = setup_pipeline()

# 11. Async score calculation
executor = concurrent.futures.ThreadPoolExecutor()
future_scores = executor.submit(get_dart_scores, ...)
```

---

## 9. EXPECTED PERFORMANCE IMPROVEMENTS

| Configuration | Current FPS | Expected FPS | Improvement |
|---------------|-------------|--------------|-------------|
| **Baseline (current)** | 8-12 FPS | - | - |
| **Phase 1 only** | 8-12 FPS | 25-35 FPS | +17-27 FPS |
| **Phase 1 + 2** | 8-12 FPS | 30-42 FPS | +22-34 FPS |
| **All phases** | 8-12 FPS | 40-50 FPS | +32-42 FPS |

---

## 10. COMPATIBILITY WITH STREAMLIT FRAMEWORK

All techniques are **fully compatible** with Streamlit:

✅ **Compatible**:
- Frame skipping, caching, threading
- `st.empty()` reuse pattern
- `@st.cache_resource`, `@st.cache_data`
- Session state for persistence
- OpenCV optimizations

⚠️ **Partial Compatibility**:
- Multiprocessing (requires careful handling of Streamlit context)
- Shared memory (use `multiprocessing.Queue` instead)

❌ **Not Compatible**:
- Direct DOM manipulation (not needed)
- Custom WebSocket servers (Streamlit handles this)

---

## 11. MONITORING AND VALIDATION

### Add FPS/Performance Dashboard:
```python
# In sidebar
with st.sidebar.expander("📊 Performance Metrics", expanded=True):
    col1, col2 = st.columns(2)

    col1.metric("Current FPS", f"{avg_fps:.1f}")
    col2.metric("Target FPS", "30")

    st.progress(min(avg_fps / 30, 1.0))

    st.write("**Pipeline Breakdown:**")
    st.write(f"- Capture: {timing_capture:.1f}ms")
    st.write(f"- Inference: {timing_inference:.1f}ms")
    st.write(f"- Scoring: {timing_scoring:.1f}ms")
    st.write(f"- Display: {timing_display:.1f}ms")
```

---

## 12. HARDWARE-SPECIFIC OPTIMIZATIONS (M2 MacBook Pro)

### Metal Performance Shaders (MPS):
```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device('mps')
    model.to(device)
    st.success("✅ Using Metal GPU acceleration")
else:
    st.warning("⚠️ MPS not available, using CPU")
```

### M2-Specific Settings:
```python
# Optimize for M2 Neural Engine
model.export(format='coreml', nms=True)  # Use CoreML for M2 Neural Engine

# Or use ONNX with optimization
model.export(format='onnx', simplify=True, dynamic=True)
```

---

## 13. FINAL RECOMMENDATIONS

### For Immediate Implementation (Today):
1. ✅ Set `CAP_PROP_BUFFERSIZE=1`
2. ✅ Implement frame skipping (every 3 frames)
3. ✅ Downscale to 960x540 before inference
4. ✅ Reduce YOLO input size to 480px
5. ✅ Use MPS backend with FP16

**Expected Result**: 25-35 FPS (3x improvement)

### For This Week:
6. ✅ Add video capture threading
7. ✅ Implement lightweight annotation
8. ✅ Cache transformation matrix
9. ✅ Add performance profiling

**Expected Result**: 35-45 FPS (4x improvement)

### For Production (Optional):
10. ✅ Full pipeline decoupling with multiprocessing
11. ✅ Export model to CoreML for M2 Neural Engine
12. ✅ Implement adaptive resolution scaling

**Expected Result**: 45-60 FPS (5-6x improvement)

---

## 14. CODE SNIPPET: Complete Optimized Main Loop

```python
def run_detection_optimized(config):
    """Optimized detection loop with all Phase 1 improvements."""

    # Load model with caching
    @st.cache_resource
    def load_model(path):
        model = YOLO(path)
        model.to('mps')  # M2 GPU
        model.half()     # FP16
        return model

    model = load_model(config['model_path'])

    # Open camera with optimizations
    cap = cv2.VideoCapture(config['selected_camera'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # UI placeholders (reuse)
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    scores_placeholder = st.empty()

    # State
    frame_count = 0
    cached_detections = None
    cached_scores = []
    fps_counter = []

    # Profiling
    timings = {'capture': [], 'inference': [], 'scoring': [], 'display': []}

    while st.session_state.running:
        t_start = time.perf_counter()

        # 1. CAPTURE (with timing)
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        timings['capture'].append(time.perf_counter() - t0)

        # 2. DOWNSCALE
        frame_small = cv2.resize(frame, (960, 540),
                                interpolation=cv2.INTER_LINEAR)

        # 3. INFERENCE (every 3rd frame)
        t1 = time.perf_counter()
        if frame_count % 3 == 0:
            results = model.predict(
                source=frame_small,
                conf=config['conf_threshold'],
                iou=config['iou_threshold'],
                imgsz=480,  # Reduced from 800
                verbose=False
            )
            cached_detections = results[0].boxes
        timings['inference'].append(time.perf_counter() - t1)

        # 4. SCORING (cached)
        t2 = time.perf_counter()
        if frame_count % 3 == 0 and cached_detections is not None:
            # ... score calculation ...
            cached_scores = dart_scores
        timings['scoring'].append(time.perf_counter() - t2)

        # 5. DISPLAY (lightweight)
        t3 = time.perf_counter()
        if cached_detections is not None:
            annotated = fast_draw_detections(frame, cached_detections, config)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, use_column_width=True)
        timings['display'].append(time.perf_counter() - t3)

        # 6. FPS UPDATE (every 10 frames)
        if frame_count % 10 == 0:
            fps = 1 / (time.perf_counter() - t_start)
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)
            fps_placeholder.metric("FPS", f"{avg_fps:.1f}")

        frame_count += 1

    cap.release()
```

---

## REFERENCES

1. OpenCV VideoCapture Buffer Management: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
2. Streamlit Performance Best Practices: https://docs.streamlit.io/develop/concepts/architecture/caching
3. YOLOv11 Optimization Guide: https://docs.ultralytics.com/guides/speed/
4. PyTorch MPS Backend: https://pytorch.org/docs/stable/notes/mps.html
5. Threading vs Multiprocessing: https://docs.python.org/3/library/concurrent.futures.html

---

**Research Completed**: 2025-10-17
**Next Steps**: Implement Phase 1 optimizations (estimated 30-40 FPS improvement)
