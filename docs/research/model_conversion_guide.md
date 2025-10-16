# YOLO11 Model Conversion to Web-Compatible Formats
## Comprehensive Research Guide

**Project**: Deeper Darts - YOLO11 Dart Detection
**Model**: `/Users/fewzy/Dev/ai/deeper_darts/models/best.pt` (PyTorch YOLO11)
**Target**: Web deployment (Browser-based inference)
**Date**: 2025-01-17

---

## Executive Summary

This guide provides complete conversion pathways for deploying YOLO11 dart detection models in web browsers. Three primary pathways exist:

1. **PyTorch → ONNX → ONNX Runtime Web** (Recommended for performance)
2. **PyTorch → ONNX → TensorFlow.js** (Better framework compatibility)
3. **PyTorch → CoreML → Web** (Apple ecosystem, limited web support)

Current Ultralytics version: `8.3.214`

---

## 1. CONVERSION PATHWAYS

### 1.1 PyTorch → ONNX → ONNX Runtime Web (PRIMARY RECOMMENDATION)

#### A. Export to ONNX

**Basic Export Command:**
```bash
# CLI method (recommended for scripts)
yolo export model=models/best.pt format=onnx imgsz=800

# Python method
python3 << EOF
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(
    format="onnx",
    imgsz=800,           # Match training size
    simplify=True,       # Enable OnnxSlim optimization
    dynamic=False,       # Fixed input size for web
    half=False,          # FP32 for compatibility
    opset=17            # Latest stable opset
)
EOF
```

**Output**: `models/best.onnx` (~6-12 MB for nano model, ~25-50 MB for small/medium)

#### B. Advanced Optimization Options

**FP16 Quantization (50% size reduction):**
```bash
yolo export model=models/best.pt format=onnx imgsz=800 half=True simplify=True
```

**With Dynamic Shapes (flexible input sizes):**
```bash
yolo export model=models/best.pt format=onnx imgsz=800 dynamic=True simplify=True
```

**Key Export Parameters:**

| Parameter | Default | Description | Web Impact |
|-----------|---------|-------------|------------|
| `format` | - | Export format | Use "onnx" |
| `imgsz` | 640 | Input image size | Match training (800 for this project) |
| `half` | False | FP16 quantization | 50% size reduction, may reduce accuracy |
| `dynamic` | False | Dynamic input sizes | Reduces browser compatibility |
| `simplify` | True | OnnxSlim optimization | 10-15% speed improvement |
| `opset` | Latest | ONNX opset version | Use 17 for stability |
| `nms` | False | Include NMS in model | Keep False, implement in JS |
| `batch` | 1 | Batch size | Use 1 for real-time web inference |

#### C. ONNX Optimization with OnnxSlim

The `simplify=True` parameter uses **OnnxSlim** (modern, 2025) instead of the older `onnx-simplifier`:

**OnnxSlim Benefits:**
- 10-15% faster inference
- Actively maintained (merged into Hugging Face optimum)
- Used by ultralytics, yolov10, yolov12
- Better layer optimization than onnx-simplifier

**Manual OnnxSlim (if needed):**
```bash
pip install onnxslim
onnxslim models/best.onnx models/best_slim.onnx
```

#### D. ONNX Runtime Web Integration

**Install Dependencies:**
```bash
npm install onnxruntime-web opencv.js
```

**JavaScript Implementation:**
```javascript
// main.js - ONNX Runtime Web with WebGPU/WASM
import * as ort from 'onnxruntime-web';
import cv from 'opencv.js';

// Configure execution providers
ort.env.wasm.numThreads = 4;
ort.env.wasm.simd = true;

// Load model
const session = await ort.InferenceSession.create('models/best.onnx', {
  executionProviders: ['webgpu', 'wasm'],  // WebGPU first, fallback to WASM
  graphOptimizationLevel: 'all'
});

// Preprocess image (OpenCV.js)
function preprocessImage(imageSrc, inputSize = 800) {
  const img = cv.imread(imageSrc);
  const resized = new cv.Mat();

  // Resize to model input size
  cv.resize(img, resized, new cv.Size(inputSize, inputSize));

  // Convert to RGB
  cv.cvtColor(resized, resized, cv.COLOR_RGBA2RGB);

  // Normalize to [0, 1]
  resized.convertTo(resized, cv.CV_32F, 1.0 / 255.0);

  // Convert to CHW format (channels, height, width)
  const data = new Float32Array(3 * inputSize * inputSize);
  const pixels = resized.data32F;

  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < inputSize; h++) {
      for (let w = 0; w < inputSize; w++) {
        data[c * inputSize * inputSize + h * inputSize + w] =
          pixels[(h * inputSize + w) * 3 + c];
      }
    }
  }

  img.delete();
  resized.delete();

  return data;
}

// Run inference
async function detectDarts(imageElement) {
  const inputData = preprocessImage(imageElement, 800);

  // Create tensor
  const tensor = new ort.Tensor('float32', inputData, [1, 3, 800, 800]);

  // Run model
  const results = await session.run({ images: tensor });

  // results.output0.data contains raw YOLO output
  // Shape: [1, 84, 8400] for YOLO11 (4 bbox + 80 classes, or 4 + n_classes)
  return results.output0.data;
}

// Post-processing: NMS (Non-Maximum Suppression)
function nonMaxSuppression(predictions, confThreshold = 0.25, iouThreshold = 0.45) {
  // YOLO output format: [batch, 4 + num_classes, num_predictions]
  // For dart detection: [1, 8, 8400] (4 bbox coords + 4 classes: dart, cal_pt_1-4)

  const boxes = [];
  const scores = [];
  const classIds = [];

  // Parse predictions
  for (let i = 0; i < predictions.length; i += (4 + numClasses)) {
    const [x, y, w, h] = predictions.slice(i, i + 4);
    const classScores = predictions.slice(i + 4, i + 4 + numClasses);

    const maxScore = Math.max(...classScores);
    const classId = classScores.indexOf(maxScore);

    if (maxScore > confThreshold) {
      boxes.push([x - w/2, y - h/2, w, h]);  // Convert to xywh
      scores.push(maxScore);
      classIds.push(classId);
    }
  }

  // Apply NMS (simplified - use a library for production)
  return applyNMS(boxes, scores, classIds, iouThreshold);
}

// Usage
const detections = await detectDarts(imageElement);
const filtered = nonMaxSuppression(detections);
```

**Performance Expectations:**
- **WebGPU**: 15-30 FPS on desktop (Chrome 113+, Safari 26+)
- **WASM (CPU)**: 5-15 FPS on desktop, 2-8 FPS on mobile
- **Model Load Time**: 100-500ms depending on model size

---

### 1.2 PyTorch → ONNX → TensorFlow.js

#### A. Export to TensorFlow.js Directly

**Direct Export (Ultralytics):**
```bash
# Export with TF.js format
yolo export model=models/best.pt format=tfjs imgsz=800

# With quantization
yolo export model=models/best.pt format=tfjs imgsz=800 int8=True
```

**Output**: `models/best_web_model/` directory with:
- `model.json` - Model architecture
- `group1-shard*.bin` - Weight files

#### B. Multi-Step Conversion (ONNX → TF → TF.js)

**If direct export fails:**
```bash
# Step 1: Export to ONNX
yolo export model=models/best.pt format=onnx imgsz=800 simplify=True

# Step 2: ONNX to TensorFlow (requires onnx2tf)
pip install onnx2tf tensorflow
onnx2tf -i models/best.onnx -o models/tf_model

# Step 3: TensorFlow to TensorFlow.js
pip install tensorflowjs
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --quantization_bytes=4 \
  models/tf_model \
  models/best_tfjs
```

#### C. TensorFlow.js Implementation

**Install Dependencies:**
```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgl
```

**JavaScript Implementation:**
```javascript
import * as tf from '@tensorflow/tfjs';

// Load model
const model = await tf.loadGraphModel('models/best_web_model/model.json');

// Preprocess
async function preprocessImageTF(imageElement) {
  let tensor = tf.browser.fromPixels(imageElement);

  // Resize to 800x800
  tensor = tf.image.resizeBilinear(tensor, [800, 800]);

  // Normalize to [0, 1]
  tensor = tensor.div(255.0);

  // Add batch dimension and rearrange to [1, 800, 800, 3]
  tensor = tensor.expandDims(0);

  return tensor;
}

// Run inference
async function detectDartsTF(imageElement) {
  const inputTensor = await preprocessImageTF(imageElement);

  // Run model
  const predictions = await model.predict(inputTensor);

  // Get data
  const data = await predictions.data();

  // Clean up
  inputTensor.dispose();
  predictions.dispose();

  return data;
}

// NMS using TensorFlow.js built-in
async function postprocessTF(predictions, scoreThreshold = 0.25, iouThreshold = 0.45) {
  // Parse predictions (format depends on YOLO export)
  const boxes = tf.tensor2d(boxesArray);  // [N, 4]
  const scores = tf.tensor1d(scoresArray);  // [N]

  // Apply NMS
  const selectedIndices = await tf.image.nonMaxSuppressionAsync(
    boxes,
    scores,
    maxOutputSize = 100,
    iouThreshold,
    scoreThreshold
  );

  const selected = await selectedIndices.data();

  // Clean up
  boxes.dispose();
  scores.dispose();
  selectedIndices.dispose();

  return selected;
}
```

**Known Issues:**
1. **Layer Compatibility**: Some YOLO layers may not convert properly
2. **Dynamic Shapes**: Can get corrupted in multi-step conversion (use fixed 800x800)
3. **NMS Implementation**: May need custom JavaScript implementation
4. **Performance**: Generally slower than ONNX Runtime Web

---

### 1.3 PyTorch → CoreML (Alternative for Apple Devices)

#### A. Export to CoreML

**Export Command:**
```bash
yolo export model=models/best.pt format=coreml imgsz=800 nms=True
```

**With Quantization:**
```bash
yolo export model=models/best.pt format=coreml imgsz=800 int8=True nms=True
```

**Output**: `models/best.mlpackage/`

#### B. Web Integration Options

**CoreML has LIMITED web support:**
1. **Native iOS Apps**: Use Swift/Objective-C with CoreML
2. **React Native**: Bridge to native CoreML
3. **Capacitor/Cordova**: Native plugin for web apps
4. **Safari WebKit**: No direct support yet

**Not recommended for pure web deployment** - use ONNX or TF.js instead.

---

## 2. EXACT COMMAND EXAMPLES

### For This Project (Deeper Darts)

**Current Setup:**
- Training image size: 800x800
- Model: YOLO11 (Ultralytics v8.3.214)
- Classes: 5 (dart, cal_pt_1, cal_pt_2, cal_pt_3, cal_pt_4)
- Best model: `/Users/fewzy/Dev/ai/deeper_darts/models/best.pt`

**Recommended Export Commands:**

```bash
cd /Users/fewzy/Dev/ai/deeper_darts

# Option 1: ONNX (Best for performance)
yolo export model=models/best.pt format=onnx imgsz=800 simplify=True dynamic=False half=False opset=17

# Option 2: ONNX with FP16 (Smaller size)
yolo export model=models/best.pt format=onnx imgsz=800 simplify=True half=True

# Option 3: TensorFlow.js (Direct)
yolo export model=models/best.pt format=tfjs imgsz=800

# Option 4: Multiple formats at once
yolo export model=models/best.pt format=onnx,tfjs imgsz=800 simplify=True
```

**Validation After Export:**
```bash
# Check ONNX model
pip install onnx
python3 << EOF
import onnx
model = onnx.load("models/best.onnx")
print(f"Model inputs: {[i.name for i in model.graph.input]}")
print(f"Model outputs: {[o.name for o in model.graph.output]}")
print(f"Opset version: {model.opset_import[0].version}")
EOF

# Test inference
python3 << EOF
from ultralytics import YOLO
import cv2

# Load ONNX model
model = YOLO("models/best.onnx", task="detect")

# Test prediction
results = model.predict("datasets/test/images/d1_03_31_2020/sample.JPG")
print(f"Detections: {len(results[0].boxes)}")
EOF
```

---

## 3. WEB RUNTIME PERFORMANCE

### 3.1 Browser Compatibility (2025)

| Feature | Chrome | Safari | Firefox | Edge | Mobile |
|---------|--------|--------|---------|------|--------|
| **ONNX Runtime Web (WASM)** | ✅ 100+ | ✅ 14+ | ✅ 80+ | ✅ 100+ | ✅ All |
| **ONNX Runtime Web (WebGL)** | ⚠️ Maintenance | ⚠️ Limited | ⚠️ Limited | ⚠️ Maintenance | ⚠️ Limited |
| **ONNX Runtime Web (WebGPU)** | ✅ 113+ | ✅ 26+ (iOS 26) | ✅ 141+ (Win) | ✅ 113+ | ✅ Android, iOS 26+ |
| **TensorFlow.js (WebGL)** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **TensorFlow.js (WebGPU)** | ✅ 113+ | ✅ 26+ | ✅ 141+ | ✅ 113+ | ✅ Limited |
| **WebNN API** | 🚧 Experimental | 🚧 Not yet | 🚧 Not yet | 🚧 Experimental | ❌ Not yet |

### 3.2 Acceleration Options

**ONNX Runtime Web Backends:**
1. **WebGPU** (Recommended for 2025)
   - Best performance: 3-5x faster than WebGL
   - FP16 support reduces memory usage
   - Available: Chrome 113+, Safari 26+, Android Chrome
   - Compute shader support for complex operations

2. **WASM (WebAssembly)**
   - CPU-based, universal compatibility
   - SIMD support: 2-3x faster than regular WASM
   - Reliable fallback option
   - Works on all modern browsers

3. **WebGL** (Legacy)
   - Maintenance mode only
   - Limited operator support
   - Not recommended for new projects

**TensorFlow.js Backends:**
1. **WebGL** - Mature, widely supported
2. **WASM** - CPU fallback
3. **WebGPU** - New, faster (Chrome 113+)

### 3.3 Expected Performance Metrics

**ONNX Runtime Web (YOLO11n on best.pt):**

| Device | Backend | FPS | Latency | Load Time |
|--------|---------|-----|---------|-----------|
| Desktop (RTX 3060) | WebGPU | 25-35 | 30-40ms | 200ms |
| Desktop (RTX 3060) | WASM | 8-12 | 80-125ms | 150ms |
| MacBook Pro M2 | WebGPU | 20-30 | 35-50ms | 250ms |
| MacBook Pro M2 | WASM | 10-15 | 65-100ms | 180ms |
| iPhone 15 Pro | WebGPU (Safari 26+) | 15-25 | 40-65ms | 300ms |
| Android (Pixel 8) | WebGPU | 12-20 | 50-85ms | 350ms |
| iPhone 15 Pro | WASM | 5-8 | 125-200ms | 250ms |

**TensorFlow.js (Generally 20-30% slower than ONNX):**

| Device | Backend | FPS | Latency |
|--------|---------|-----|---------|
| Desktop GPU | WebGL | 18-28 | 35-55ms |
| Desktop CPU | WASM | 5-10 | 100-200ms |
| Mobile | WebGL | 8-15 | 65-125ms |

### 3.4 Model Size Impact

| Format | Precision | Size | Accuracy Impact |
|--------|-----------|------|-----------------|
| PyTorch (.pt) | FP32 | ~6 MB (nano) | Baseline |
| ONNX (FP32) | FP32 | ~6-7 MB | Negligible (<0.1%) |
| ONNX (FP16) | FP16 | ~3-4 MB | Small (<1%) |
| ONNX (INT8) | INT8 | ~1.5-2 MB | Moderate (1-3%) |
| TF.js | FP32 | ~7-9 MB | Small (<0.5%) |
| TF.js (INT8) | INT8 | ~2-3 MB | Moderate (2-4%) |

**File Size Estimates for best.pt:**
- If trained with yolov8n (nano): ~6 MB → ONNX ~6 MB, FP16 ~3 MB
- If trained with yolov8s (small): ~22 MB → ONNX ~22 MB, FP16 ~11 MB
- If trained with yolov8m (medium): ~50 MB → ONNX ~50 MB, FP16 ~25 MB

---

## 4. KNOWN ISSUES & SOLUTIONS

### 4.1 Layer Compatibility Issues

**Problem**: Some YOLO layers don't convert to ONNX/TF.js properly

**Solutions:**
1. Use `simplify=True` to optimize graph
2. Ensure opset version >= 12 for ONNX
3. Use fixed input shapes (`dynamic=False`)
4. Export with `nms=False` and implement NMS in JavaScript

### 4.2 Dynamic Shapes in Browser

**Problem**: Dynamic input sizes cause issues in web runtimes

**Solution:**
```bash
# Export with FIXED shapes
yolo export model=models/best.pt format=onnx imgsz=800 dynamic=False

# If multiple sizes needed, export separate models
yolo export model=models/best.pt format=onnx imgsz=640 dynamic=False
yolo export model=models/best.pt format=onnx imgsz=800 dynamic=False
```

### 4.3 NMS (Non-Maximum Suppression) Implementation

**Problem**: YOLO models export raw predictions without NMS for web

**Solution**: Implement NMS in JavaScript

```javascript
// Fast NMS implementation
function nms(boxes, scores, iouThreshold = 0.45) {
  const indices = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .map(item => item.idx);

  const keep = [];

  while (indices.length > 0) {
    const current = indices.shift();
    keep.push(current);

    indices = indices.filter(idx => {
      const iou = calculateIoU(boxes[current], boxes[idx]);
      return iou < iouThreshold;
    });
  }

  return keep;
}

function calculateIoU(box1, box2) {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const area1 = w1 * h1;
  const area2 = w2 * h2;

  const xi1 = Math.max(x1, x2);
  const yi1 = Math.max(y1, y2);
  const xi2 = Math.min(x1 + w1, x2 + w2);
  const yi2 = Math.min(y1 + h1, y2 + h2);

  const intersectionArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
  const unionArea = area1 + area2 - intersectionArea;

  return intersectionArea / unionArea;
}
```

### 4.4 ONNX to TensorFlow.js Conversion Issues

**Problem**: ONNX → TF conversion fails with unsupported ops

**Solutions:**

1. **Use onnx2tf with custom config:**
```python
# conversion_config.json
{
  "operations": {
    "ArgMax": {"transpose": "before"},
    "ReduceMax": {"transpose": "before"}
  }
}
```

```bash
onnx2tf -i models/best.onnx -o models/tf_model -c conversion_config.json
```

2. **Simplify ONNX before conversion:**
```bash
pip install onnxslim
onnxslim models/best.onnx models/best_slim.onnx
onnx2tf -i models/best_slim.onnx -o models/tf_model
```

3. **Use direct TF.js export instead:**
```bash
yolo export model=models/best.pt format=tfjs imgsz=800
```

### 4.5 WebGL Backend Issues

**Problem**: WebGL doesn't support all ONNX operators

**Solution**: Use WebGPU or WASM instead
```javascript
const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: ['webgpu', 'wasm'],  // Prefer WebGPU, fallback to WASM
});
```

### 4.6 Memory Management in Browser

**Problem**: Large models cause memory issues on mobile

**Solutions:**
1. Use FP16 or INT8 quantization
2. Implement model chunking/tiling for large images
3. Dispose tensors after use:
```javascript
tensor.dispose();
session.dispose();
```

---

## 5. COMPLETE WORKFLOW EXAMPLE

### End-to-End Dart Detection in Browser

**Project Structure:**
```
deeper_darts_web/
├── models/
│   ├── best.onnx (exported model)
│   └── best.json (metadata)
├── src/
│   ├── detector.js (inference logic)
│   ├── preprocessing.js (image processing)
│   ├── postprocessing.js (NMS, drawing)
│   └── utils.js (helpers)
├── public/
│   ├── index.html
│   └── styles.css
├── package.json
└── webpack.config.js
```

**Step 1: Export Model**
```bash
cd /Users/fewzy/Dev/ai/deeper_darts
yolo export model=models/best.pt format=onnx imgsz=800 simplify=True opset=17
```

**Step 2: Setup Web Project**
```bash
mkdir deeper_darts_web && cd deeper_darts_web
npm init -y
npm install onnxruntime-web opencv.js webpack webpack-cli webpack-dev-server
```

**Step 3: Implement Detector (detector.js)**
```javascript
import * as ort from 'onnxruntime-web';

class DartDetector {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.inputSize = 800;
    this.classes = ['dart', 'cal_pt_1', 'cal_pt_2', 'cal_pt_3', 'cal_pt_4'];
  }

  async initialize() {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all'
    });

    console.log('Model loaded successfully');
  }

  async detect(imageElement, options = {}) {
    const {
      confidenceThreshold = 0.25,
      iouThreshold = 0.45,
      maxDetections = 100
    } = options;

    // Preprocess
    const inputTensor = this.preprocessImage(imageElement);

    // Inference
    const results = await this.session.run({ images: inputTensor });

    // Postprocess
    const detections = this.postprocess(
      results.output0.data,
      confidenceThreshold,
      iouThreshold,
      maxDetections
    );

    return detections;
  }

  preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext('2d');

    // Draw and resize
    ctx.drawImage(imageElement, 0, 0, this.inputSize, this.inputSize);

    // Get pixel data
    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const pixels = imageData.data;

    // Convert to CHW format and normalize
    const data = new Float32Array(3 * this.inputSize * this.inputSize);

    for (let i = 0; i < pixels.length; i += 4) {
      const pixelIndex = i / 4;
      const h = Math.floor(pixelIndex / this.inputSize);
      const w = pixelIndex % this.inputSize;

      data[0 * this.inputSize * this.inputSize + h * this.inputSize + w] = pixels[i] / 255.0;     // R
      data[1 * this.inputSize * this.inputSize + h * this.inputSize + w] = pixels[i + 1] / 255.0; // G
      data[2 * this.inputSize * this.inputSize + h * this.inputSize + w] = pixels[i + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', data, [1, 3, this.inputSize, this.inputSize]);
  }

  postprocess(rawOutput, confThreshold, iouThreshold, maxDetections) {
    // YOLO output format: [1, 4 + num_classes, num_predictions]
    const numClasses = this.classes.length;
    const numPredictions = rawOutput.length / (4 + numClasses);

    const detections = [];

    for (let i = 0; i < numPredictions; i++) {
      const offset = i * (4 + numClasses);

      // Get bbox coordinates
      const x = rawOutput[offset];
      const y = rawOutput[offset + 1];
      const w = rawOutput[offset + 2];
      const h = rawOutput[offset + 3];

      // Get class scores
      const scores = rawOutput.slice(offset + 4, offset + 4 + numClasses);
      const maxScore = Math.max(...scores);
      const classId = scores.indexOf(maxScore);

      if (maxScore > confThreshold) {
        detections.push({
          bbox: [x - w/2, y - h/2, w, h],  // Convert to xyxy
          score: maxScore,
          classId: classId,
          className: this.classes[classId]
        });
      }
    }

    // Apply NMS
    return this.nms(detections, iouThreshold).slice(0, maxDetections);
  }

  nms(detections, iouThreshold) {
    // Sort by score descending
    detections.sort((a, b) => b.score - a.score);

    const keep = [];

    while (detections.length > 0) {
      const current = detections.shift();
      keep.push(current);

      detections = detections.filter(det => {
        const iou = this.calculateIoU(current.bbox, det.bbox);
        return iou < iouThreshold;
      });
    }

    return keep;
  }

  calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    const area1 = w1 * h1;
    const area2 = w2 * h2;

    const xi1 = Math.max(x1, x2);
    const yi1 = Math.max(y1, y2);
    const xi2 = Math.min(x1 + w1, x2 + w2);
    const yi2 = Math.min(y1 + h1, y2 + h2);

    const intersectionArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
    const unionArea = area1 + area2 - intersectionArea;

    return intersectionArea / unionArea;
  }
}

export default DartDetector;
```

**Step 4: HTML Interface (index.html)**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Dart Detection - Web</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    #canvas { border: 2px solid #333; max-width: 100%; }
    .controls { margin: 20px 0; }
    button { padding: 10px 20px; margin: 5px; }
    #stats { margin-top: 10px; padding: 10px; background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>Dart Detection System</h1>

  <div class="controls">
    <input type="file" id="imageInput" accept="image/*">
    <button id="detectBtn">Detect Darts</button>
    <button id="webcamBtn">Use Webcam</button>
  </div>

  <canvas id="canvas" width="800" height="800"></canvas>

  <div id="stats">
    <p>Status: <span id="status">Ready</span></p>
    <p>Inference Time: <span id="inferenceTime">-</span></p>
    <p>Detections: <span id="detectionCount">-</span></p>
  </div>

  <script type="module">
    import DartDetector from './src/detector.js';

    let detector;
    let image;

    async function init() {
      document.getElementById('status').textContent = 'Loading model...';
      detector = new DartDetector('./models/best.onnx');
      await detector.initialize();
      document.getElementById('status').textContent = 'Ready';
    }

    document.getElementById('imageInput').addEventListener('change', (e) => {
      const file = e.target.files[0];
      const reader = new FileReader();

      reader.onload = (event) => {
        image = new Image();
        image.onload = () => {
          const canvas = document.getElementById('canvas');
          const ctx = canvas.getContext('2d');
          ctx.drawImage(image, 0, 0, 800, 800);
        };
        image.src = event.target.result;
      };

      reader.readAsDataURL(file);
    });

    document.getElementById('detectBtn').addEventListener('click', async () => {
      if (!image) {
        alert('Please select an image first');
        return;
      }

      document.getElementById('status').textContent = 'Detecting...';
      const startTime = performance.now();

      const detections = await detector.detect(image);

      const endTime = performance.now();
      const inferenceTime = (endTime - startTime).toFixed(2);

      // Draw results
      drawDetections(detections);

      // Update stats
      document.getElementById('status').textContent = 'Complete';
      document.getElementById('inferenceTime').textContent = `${inferenceTime} ms`;
      document.getElementById('detectionCount').textContent = detections.length;
    });

    function drawDetections(detections) {
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');

      detections.forEach(det => {
        const [x, y, w, h] = det.bbox;

        // Draw bbox
        ctx.strokeStyle = det.className === 'dart' ? '#ff0000' : '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label
        ctx.fillStyle = '#000000';
        ctx.font = '16px Arial';
        ctx.fillText(
          `${det.className} ${(det.score * 100).toFixed(1)}%`,
          x, y - 5
        );
      });
    }

    init();
  </script>
</body>
</html>
```

---

## 6. OPTIMIZATION RECOMMENDATIONS

### For Best Performance:

1. **Use ONNX with WebGPU** (fastest)
   - 3-5x faster than WebGL
   - FP16 support
   - Requires Chrome 113+, Safari 26+

2. **Optimize Model Size**
   - Use FP16 quantization: `half=True`
   - Expected: ~3 MB for nano model
   - Acceptable accuracy loss: <1%

3. **Fixed Input Shapes**
   - Export with `dynamic=False`
   - Better browser compatibility
   - Faster inference

4. **Enable OnnxSlim**
   - Always use `simplify=True`
   - 10-15% speed improvement
   - No accuracy loss

5. **Implement Efficient NMS**
   - Keep NMS in JavaScript
   - Use optimized IoU calculation
   - Consider Web Workers for parallelization

### For Best Compatibility:

1. **Use WASM backend**
   - Universal browser support
   - Reliable performance
   - Good mobile support

2. **Provide Multiple Backends**
   ```javascript
   const providers = ['webgpu', 'webgl', 'wasm'];  // Try in order
   ```

3. **Progressive Enhancement**
   - Detect capabilities
   - Fall back gracefully
   - Show loading states

---

## 7. NEXT STEPS

### Immediate Actions:

1. **Export Model to ONNX**
   ```bash
   cd /Users/fewzy/Dev/ai/deeper_darts
   yolo export model=models/best.pt format=onnx imgsz=800 simplify=True opset=17
   ```

2. **Test ONNX Model**
   ```bash
   python3 << EOF
   from ultralytics import YOLO
   model = YOLO("models/best.onnx")
   results = model.predict("datasets/test/images/d1_03_31_2020/sample.JPG")
   print(f"Detections: {len(results[0].boxes)}")
   EOF
   ```

3. **Setup Web Project**
   ```bash
   mkdir deeper_darts_web
   cd deeper_darts_web
   npm init -y
   npm install onnxruntime-web webpack webpack-cli webpack-dev-server
   ```

4. **Implement Basic Detector**
   - Copy detector.js code above
   - Create HTML interface
   - Test with sample images

### Future Enhancements:

1. **Real-time Webcam Detection**
2. **Progressive Web App (PWA)**
3. **Offline Capability**
4. **Score Calculation Integration**
5. **Mobile Optimization**
6. **A/B Testing (ONNX vs TF.js)**

---

## 8. TROUBLESHOOTING CHECKLIST

### Model Export Issues:
- [ ] Ultralytics version >= 8.0.0
- [ ] PyTorch compatibility
- [ ] Sufficient disk space
- [ ] Correct model path
- [ ] Valid export format

### Browser Loading Issues:
- [ ] CORS headers configured
- [ ] Model file accessible
- [ ] Correct MIME types
- [ ] No console errors
- [ ] WebGPU/WASM supported

### Performance Issues:
- [ ] Backend selection correct
- [ ] Model quantized appropriately
- [ ] Input size matches training
- [ ] Tensor disposal implemented
- [ ] NMS optimized

### Accuracy Issues:
- [ ] Preprocessing matches training
- [ ] Normalization correct
- [ ] Input format (CHW/HWC)
- [ ] Confidence threshold tuned
- [ ] IoU threshold appropriate

---

## 9. USEFUL RESOURCES

### Official Documentation:
- Ultralytics Export: https://docs.ultralytics.com/modes/export/
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- TensorFlow.js: https://www.tensorflow.org/js

### Tutorials:
- PyImageSearch YOLO Browser Tutorial (2025): https://pyimagesearch.com/2025/07/28/run-yolo-model-in-the-browser-with-onnx-webassembly-and-next-js/
- ONNX to TF.js Conversion: https://medium.com/geekculture/continue-the-journey-of-adding-non-max-suppression-nms-to-yolov8-onnx-model-fix-issue-s-226a0df2f339

### GitHub Examples:
- YOLO ONNX Web Detection: https://github.com/nomi30701/yolo-object-detection-onnxruntime-web
- Ultralytics Examples: https://github.com/ultralytics/ultralytics/tree/main/examples

---

## 10. SUMMARY TABLE

| Conversion Path | Pros | Cons | Recommended Use |
|----------------|------|------|-----------------|
| **PyTorch → ONNX → ONNX Runtime Web** | ✅ Best performance<br>✅ WebGPU support<br>✅ Active development<br>✅ FP16/INT8 | ⚠️ WebGPU limited on old devices<br>⚠️ Manual NMS | **Primary choice for production** |
| **PyTorch → TF.js** | ✅ Built-in NMS<br>✅ Mature ecosystem<br>✅ Good compatibility | ❌ Slower (20-30%)<br>❌ Larger model size<br>⚠️ Conversion issues | Fallback if ONNX fails |
| **PyTorch → CoreML** | ✅ Best on iOS<br>✅ Hardware optimized | ❌ No web support<br>❌ Apple only | Native iOS apps only |

---

**Research Status**: ✅ Complete
**Storage Key**: `model_conversion_guide`
**Last Updated**: 2025-01-17
**Ultralytics Version**: 8.3.214
