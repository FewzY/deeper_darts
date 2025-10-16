# Mobile Deployment Strategy: iPhone Optimization

## Executive Summary

Comprehensive guide for deploying YOLO11 dart detection model on iPhone with **CoreML**, **INT8 quantization**, and **Neural Engine optimization**, targeting **30-60 FPS** real-time performance.

## Deployment Architecture

### End-to-End Pipeline

```
Camera Frame (1920x1080)
    ↓
Preprocessing (resize, normalize)
    ↓
YOLO11 CoreML Model (416×416, INT8)
    ↓
Neural Engine / GPU / CPU
    ↓
Detections (darts + calibration points)
    ↓
Homography Transformation
    ↓
Score Calculation
    ↓
UI Update (score, visualization)
```

---

## Part 1: Model Export and Optimization

### 1.1 CoreML Export Options

**Basic Export**:
```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Standard CoreML export
model.export(
    format='coreml',
    imgsz=416,          # Mobile-optimized input size
    nms=True,           # Include NMS in model
)
```

**Optimized Export with Quantization**:
```python
# INT8 quantization (recommended)
model.export(
    format='coreml',
    int8=True,          # INT8 quantization
    imgsz=416,
    nms=True,
    keras=False,
    optimize=True,
    half=False,         # Don't use with int8
    dynamic=False,      # Fixed input size
    simplify=True,      # Simplify ONNX operations
)

# Output: best_int8.mlpackage
```

**Advanced Export with coremltools**:
```python
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# 1. Export to ONNX first
model.export(format='onnx', imgsz=416)

# 2. Convert ONNX to CoreML
mlmodel = ct.convert(
    'best.onnx',
    inputs=[ct.ImageType(name="image", shape=(1, 3, 416, 416))],
    compute_units=ct.ComputeUnit.ALL,
)

# 3. Quantize (INT8)
mlmodel_int8 = quantization_utils.quantize_weights(mlmodel, nbits=8)

# 4. Save
mlmodel_int8.save('yolo11_dart_int8.mlpackage')
```

---

### 1.2 Quantization Strategies

**Quantization Options**:

| Type | Size Reduction | Speed Gain | Accuracy Loss | Use Case |
|------|---------------|------------|---------------|----------|
| **FP16** | 50% | 1.5-2x | ~1% | Balanced |
| **INT8** | 75% | 2-3x | ~2-3% | Recommended |
| **W8A8** | 75%+ | 3-4x | ~2-3% | iPhone 15 Pro+ |

**FP16 Export**:
```python
model.export(
    format='coreml',
    half=True,          # FP16 precision
    imgsz=416,
    nms=True,
)
```

**INT8 with Calibration**:
```python
import coremltools as ct

# Load FP32 model
model_fp32 = ct.models.MLModel('best.mlpackage')

# Prepare calibration data (representative dataset)
import numpy as np

# Load sample images
calibration_images = []
for img_path in sample_images[:100]:  # 100 images for calibration
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    calibration_images.append(img)

# Quantize with calibration
model_int8 = ct.compression.quantize_weights(
    model_fp32,
    mode='linear',
    dtype=np.int8,
)

model_int8.save('yolo11_dart_calibrated_int8.mlpackage')
```

**Blockwise Quantization** (2024 feature):
```python
# Advanced quantization (better accuracy)
model_int8 = ct.compression.quantize_weights(
    model_fp32,
    mode='blockwise',  # Blockwise quantization
    block_size=32,     # Block size
    dtype=np.int8,
)
```

---

### 1.3 Model Compression

**Pruning**:
```python
# Structured pruning (remove channels)
import coremltools as ct
from coremltools.optimize.coreml import prune_weights

config = ct.optimize.coreml.OpPrunerConfig(
    mode='threshold',
    threshold=0.01,  # Remove weights < 0.01
)

model_pruned = prune_weights(model_fp32, config=config)
model_pruned.save('yolo11_dart_pruned.mlpackage')
```

**Combined Optimization**:
```python
# Pruning + Quantization
model_optimized = prune_weights(model_fp32, config=prune_config)
model_optimized = quantize_weights(model_optimized, nbits=8)
model_optimized.save('yolo11_dart_optimized.mlpackage')

# Expected size: 80MB → 12-15MB
```

---

### 1.4 Compute Unit Configuration

**Compute Unit Options**:
```python
import coremltools as ct

# Option 1: All (automatic selection)
model = ct.models.MLModel('best.mlpackage')
model.compute_units = ct.ComputeUnit.ALL  # CPU + GPU + Neural Engine

# Option 2: Neural Engine only (best for INT8)
model.compute_units = ct.ComputeUnit.NEURAL_ENGINE

# Option 3: CPU + GPU (fallback)
model.compute_units = ct.ComputeUnit.CPU_AND_GPU

model.save('yolo11_configured.mlpackage')
```

**Device-Specific Optimization**:
```python
# A17 Pro / M4 (iPhone 15 Pro) - W8A8 optimized
model_a17 = model.export(
    format='coreml',
    int8=True,
    imgsz=416,
)

# A15 / A16 (iPhone 13-14) - INT8 standard
model_a15 = model.export(
    format='coreml',
    int8=True,
    imgsz=320,  # Smaller for older devices
)
```

---

## Part 2: iOS Integration

### 2.1 Xcode Project Setup

**Project Structure**:
```
DartDetectorApp/
├── DartDetectorApp/
│   ├── Models/
│   │   └── yolo11_dart_int8.mlpackage
│   ├── Detection/
│   │   ├── DartDetector.swift
│   │   ├── DartScorer.swift
│   │   └── HomographyCalculator.swift
│   ├── Camera/
│   │   ├── CameraManager.swift
│   │   └── CameraViewController.swift
│   ├── UI/
│   │   ├── GameViewController.swift
│   │   └── ResultsView.swift
│   └── Utilities/
│       ├── ImagePreprocessor.swift
│       └── BoundingBoxDrawer.swift
└── DartDetectorAppTests/
```

**Add CoreML Model**:
1. Drag `yolo11_dart_int8.mlpackage` into Xcode
2. Target: DartDetectorApp
3. Verify in Build Phases → Copy Bundle Resources

---

### 2.2 Model Loading and Configuration

**DartDetector.swift**:
```swift
import CoreML
import Vision

class DartDetector {
    private var model: VNCoreMLModel?
    private let configuration: MLModelConfiguration
    private var isReady = false

    init() {
        // Configure for Neural Engine
        configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.allowLowPrecisionAccumulationOnGPU = true

        loadModel()
    }

    private func loadModel() {
        guard let modelURL = Bundle.main.url(
            forResource: "yolo11_dart_int8",
            withExtension: "mlpackage"
        ) else {
            print("Error: Model file not found")
            return
        }

        do {
            let mlModel = try MLModel(
                contentsOf: modelURL,
                configuration: configuration
            )
            model = try VNCoreMLModel(for: mlModel)
            isReady = true
            print("Model loaded successfully")
            print("Available compute units: \(mlModel.configuration.computeUnits)")
        } catch {
            print("Error loading model: \(error.localizedDescription)")
        }
    }

    func detect(
        in image: CIImage,
        completion: @escaping (Result<[Detection], Error>) -> Void
    ) {
        guard isReady, let model = model else {
            completion(.failure(DetectionError.modelNotReady))
            return
        }

        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                completion(.failure(error))
                return
            }

            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion(.failure(DetectionError.invalidResults))
                return
            }

            // Parse detections
            let detections = results.compactMap { observation -> Detection? in
                guard let label = observation.labels.first else { return nil }

                return Detection(
                    classLabel: label.identifier,
                    confidence: observation.confidence,
                    boundingBox: observation.boundingBox,
                    classId: self.getClassId(from: label.identifier)
                )
            }

            completion(.success(detections))
        }

        // Configure request
        request.imageCropAndScaleOption = .scaleFill
        request.usesCPUOnly = false  // Use GPU/Neural Engine

        // Perform detection
        let handler = VNImageRequestHandler(
            ciImage: image,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            completion(.failure(error))
        }
    }

    private func getClassId(from label: String) -> Int {
        // Map label to class ID
        switch label {
        case "calibration_5_20": return 0
        case "calibration_13_6": return 1
        case "calibration_17_3": return 2
        case "calibration_8_11": return 3
        case "dart_tip": return 4
        default: return -1
        }
    }
}

// Detection structure
struct Detection {
    let classLabel: String
    let confidence: Float
    let boundingBox: CGRect
    let classId: Int

    var isCalibration: Bool {
        classId >= 0 && classId <= 3
    }

    var isDart: Bool {
        classId == 4
    }
}

enum DetectionError: Error {
    case modelNotReady
    case invalidResults
}
```

---

### 2.3 Image Preprocessing

**ImagePreprocessor.swift**:
```swift
import CoreImage
import Accelerate

class ImagePreprocessor {
    static func preprocessForModel(_ image: CIImage, targetSize: CGSize = CGSize(width: 416, height: 416)) -> CIImage {
        // Resize
        let scaleX = targetSize.width / image.extent.width
        let scaleY = targetSize.height / image.extent.height
        let scale = min(scaleX, scaleY)

        let resized = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        // Center crop
        let xOffset = (resized.extent.width - targetSize.width) / 2
        let yOffset = (resized.extent.height - targetSize.height) / 2

        let cropRect = CGRect(
            x: xOffset,
            y: yOffset,
            width: targetSize.width,
            height: targetSize.height
        )

        return resized.cropped(to: cropRect)
    }

    static func normalizeImage(_ image: CIImage) -> CIImage {
        // Normalize to [0, 1] (usually handled by CoreML)
        return image
    }
}
```

---

### 2.4 Camera Integration

**CameraManager.swift**:
```swift
import AVFoundation
import CoreImage

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didCapture image: CIImage)
}

class CameraManager: NSObject {
    weak var delegate: CameraManagerDelegate?

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.darts.camera.session")
    private let outputQueue = DispatchQueue(label: "com.darts.camera.output")

    var previewLayer: AVCaptureVideoPreviewLayer?

    // Frame control
    private var isProcessingFrame = false
    private var frameSkipCount = 0
    private let frameSkipInterval = 2  // Process every 2nd frame

    func setup() {
        sessionQueue.async { [weak self] in
            self?.setupCaptureSession()
        }
    }

    private func setupCaptureSession() {
        captureSession.beginConfiguration()

        // Set preset
        captureSession.sessionPreset = .hd1280x720  // 720p

        // Add camera input
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            print("Camera not available")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
        } catch {
            print("Error adding camera input: \(error)")
            return
        }

        // Configure output
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        captureSession.commitConfiguration()
    }

    func startRunning() {
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stopRunning() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    func createPreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        previewLayer = layer
        return layer
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Frame skipping
        frameSkipCount += 1
        guard frameSkipCount >= frameSkipInterval else { return }
        frameSkipCount = 0

        // Avoid processing if previous frame is still being processed
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        defer { isProcessingFrame = false }

        // Convert to CIImage
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Notify delegate
        delegate?.cameraManager(self, didCapture: ciImage)
    }
}
```

---

### 2.5 Real-Time Detection Pipeline

**GameViewController.swift**:
```swift
import UIKit
import AVFoundation

class GameViewController: UIViewController {
    private let cameraManager = CameraManager()
    private let dartDetector = DartDetector()
    private let dartScorer = DartScorer()

    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var detectionOverlay = CAShapeLayer()

    // Performance tracking
    private var frameTimes: [Double] = []
    private var lastFrameTime: Date?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupOverlay()
    }

    private func setupCamera() {
        cameraManager.delegate = self
        cameraManager.setup()

        // Add preview layer
        previewLayer = cameraManager.createPreviewLayer()
        previewLayer?.frame = view.bounds
        if let previewLayer = previewLayer {
            view.layer.insertSublayer(previewLayer, at: 0)
        }

        cameraManager.startRunning()
    }

    private func setupOverlay() {
        detectionOverlay.frame = view.bounds
        view.layer.addSublayer(detectionOverlay)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        detectionOverlay.frame = view.bounds
    }
}

extension GameViewController: CameraManagerDelegate {
    func cameraManager(_ manager: CameraManager, didCapture image: CIImage) {
        let frameStart = Date()

        // Preprocess
        let preprocessed = ImagePreprocessor.preprocessForModel(image)

        // Detect
        dartDetector.detect(in: preprocessed) { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let detections):
                // Calculate score
                let score = self.dartScorer.calculateScore(from: detections)

                // Update UI (on main thread)
                DispatchQueue.main.async {
                    self.updateUI(score: score, detections: detections)
                    self.trackPerformance(frameStart: frameStart)
                }

            case .failure(let error):
                print("Detection error: \(error)")
            }
        }
    }

    private func updateUI(score: Int, detections: [Detection]) {
        // Update score label
        // scoreLabel.text = "\(score)"

        // Draw bounding boxes
        drawDetections(detections)
    }

    private func drawDetections(_ detections: [Detection]) {
        let path = UIBezierPath()

        for detection in detections {
            let rect = convertToViewCoordinates(detection.boundingBox)

            let color: UIColor = detection.isDart ? .green : .red

            // Draw box
            let box = UIBezierPath(rect: rect)
            path.append(box)

            // Draw label
            // ...
        }

        detectionOverlay.path = path.cgPath
        detectionOverlay.strokeColor = UIColor.green.cgColor
        detectionOverlay.lineWidth = 2
        detectionOverlay.fillColor = UIColor.clear.cgColor
    }

    private func convertToViewCoordinates(_ boundingBox: CGRect) -> CGRect {
        // Convert Vision coordinates to view coordinates
        guard let previewLayer = previewLayer else { return .zero }

        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
        let normalizedRect = boundingBox.applying(transform)

        return previewLayer.layerRectConverted(fromMetadataOutputRect: normalizedRect)
    }

    private func trackPerformance(frameStart: Date) {
        let elapsed = Date().timeIntervalSince(frameStart)
        frameTimes.append(elapsed)

        // Keep last 30 frames
        if frameTimes.count > 30 {
            frameTimes.removeFirst()
        }

        // Calculate FPS
        let avgTime = frameTimes.reduce(0, +) / Double(frameTimes.count)
        let fps = 1.0 / avgTime

        // Update FPS label
        // fpsLabel.text = String(format: "FPS: %.1f", fps)
    }
}
```

---

## Part 3: Performance Optimization

### 3.1 Frame Processing Optimization

**Async Processing**:
```swift
class OptimizedDetector {
    private let detectionQueue = DispatchQueue(
        label: "com.darts.detection",
        qos: .userInitiated
    )

    func detectAsync(in image: CIImage, completion: @escaping ([Detection]) -> Void) {
        detectionQueue.async {
            self.dartDetector.detect(in: image) { result in
                if case .success(let detections) = result {
                    DispatchQueue.main.async {
                        completion(detections)
                    }
                }
            }
        }
    }
}
```

**Frame Skipping**:
```swift
var frameCounter = 0
let processEveryNFrames = 2  // Process every 2nd frame for 30 FPS → 15 FPS detection

func shouldProcessFrame() -> Bool {
    frameCounter += 1
    if frameCounter >= processEveryNFrames {
        frameCounter = 0
        return true
    }
    return false
}
```

**Debouncing**:
```swift
var lastDetectionTime = Date()
let minimumDetectionInterval: TimeInterval = 0.1  // 100ms

func shouldDetect() -> Bool {
    let now = Date()
    if now.timeIntervalSince(lastDetectionTime) > minimumDetectionInterval {
        lastDetectionTime = now
        return true
    }
    return false
}
```

---

### 3.2 Memory Optimization

**Image Pooling**:
```swift
class ImagePool {
    private var pool: [CIImage] = []
    private let maxSize = 10

    func getImage() -> CIImage? {
        return pool.popLast()
    }

    func returnImage(_ image: CIImage) {
        if pool.count < maxSize {
            pool.append(image)
        }
    }
}
```

**Autoreleasepool**:
```swift
func captureOutput(...) {
    autoreleasepool {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        processImage(ciImage)
    }
}
```

---

### 3.3 Battery Optimization

**Adaptive Frame Rate**:
```swift
class AdaptiveFrameRate {
    private var currentFPS = 30.0
    private var batteryLevel: Float = 1.0

    func adjustedFrameSkip() -> Int {
        UIDevice.current.isBatteryMonitoringEnabled = true
        batteryLevel = UIDevice.current.batteryLevel

        if batteryLevel < 0.2 {
            return 4  // Process every 4th frame
        } else if batteryLevel < 0.5 {
            return 2  // Process every 2nd frame
        } else {
            return 1  // Process every frame
        }
    }
}
```

---

## Part 4: Testing and Profiling

### 4.1 Performance Benchmarking

**Xcode Instruments**:
```swift
import os.signpost

class PerformanceMonitor {
    private let log = OSLog(subsystem: "com.darts", category: "performance")

    func measureDetection(_ block: () -> Void) {
        let signpostID = OSSignpostID(log: log)
        os_signpost(.begin, log: log, name: "Detection", signpostID: signpostID)

        block()

        os_signpost(.end, log: log, name: "Detection", signpostID: signpostID)
    }
}

// Usage
performanceMonitor.measureDetection {
    dartDetector.detect(in: image) { _ in }
}
```

**Custom Benchmarking**:
```swift
class Benchmark {
    static func measureFPS(iterations: Int = 100, block: (CIImage) -> Void) {
        let testImage = loadTestImage()
        var times: [TimeInterval] = []

        // Warm-up
        for _ in 0..<10 {
            block(testImage)
        }

        // Measure
        for _ in 0..<iterations {
            let start = Date()
            block(testImage)
            let elapsed = Date().timeIntervalSince(start)
            times.append(elapsed)
        }

        let avgTime = times.reduce(0, +) / Double(times.count)
        let fps = 1.0 / avgTime

        print("Average FPS: \(fps)")
        print("Average Latency: \(avgTime * 1000) ms")
        print("Min: \(times.min()! * 1000) ms")
        print("Max: \(times.max()! * 1000) ms")
    }
}
```

---

### 4.2 Accuracy Testing

**Test on Device**:
```swift
class AccuracyTest {
    func testOnTestSet() {
        let testCases = loadTestCases()  // Load labeled test images

        var correctScores = 0
        var totalScores = 0

        for testCase in testCases {
            let image = testCase.image
            let groundTruthScore = testCase.score

            dartDetector.detect(in: image) { result in
                if case .success(let detections) = result {
                    let predictedScore = dartScorer.calculateScore(from: detections)

                    if predictedScore == groundTruthScore {
                        correctScores += 1
                    }
                    totalScores += 1
                }
            }
        }

        let pcs = Double(correctScores) / Double(totalScores) * 100
        print("PCS (Percent Correct Score): \(pcs)%")
    }
}
```

---

## Part 5: Deployment Targets

### 5.1 Device-Specific Optimization

**iPhone Model Recommendations**:

| Device | Model | Input Size | FPS Target | Notes |
|--------|-------|------------|------------|-------|
| iPhone 12 | YOLO11n-INT8 | 320×320 | 40-45 FPS | A14 Bionic |
| iPhone 13 | YOLO11n-INT8 | 416×416 | 45-50 FPS | A15 Bionic |
| iPhone 14 | YOLO11s-INT8 | 416×416 | 40-45 FPS | A15 Bionic |
| iPhone 15 | YOLO11s-INT8 | 416×416 | 50-55 FPS | A16 Bionic |
| iPhone 15 Pro | YOLO11s-W8A8 | 416×416 | 60+ FPS | A17 Pro |

**Adaptive Configuration**:
```swift
class DeviceOptimizer {
    static func getOptimalConfiguration() -> ModelConfiguration {
        let device = UIDevice.current

        // Detect device model
        var systemInfo = utsname()
        uname(&systemInfo)
        let modelCode = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }

        switch modelCode {
        case "iPhone15,2", "iPhone15,3":  // iPhone 15 Pro
            return ModelConfiguration(
                modelName: "yolo11s_w8a8",
                inputSize: 416,
                targetFPS: 60
            )
        case "iPhone14,5", "iPhone14,4":  // iPhone 13
            return ModelConfiguration(
                modelName: "yolo11n_int8",
                inputSize: 416,
                targetFPS: 45
            )
        default:  // Older devices
            return ModelConfiguration(
                modelName: "yolo11n_int8",
                inputSize: 320,
                targetFPS: 35
            )
        }
    }
}

struct ModelConfiguration {
    let modelName: String
    let inputSize: Int
    let targetFPS: Int
}
```

---

### 5.2 App Store Requirements

**Info.plist Configuration**:
```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to detect darts and calculate scores automatically.</string>

<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>armv7</string>
    <string>coreml</string>
</array>
```

**Minimum Requirements**:
- iOS 15.0+
- iPhone 11 or later
- Core ML 5.0+
- Camera access

---

## Expected Performance Summary

### Model Sizes:
- **YOLO11n-INT8**: ~15 MB
- **YOLO11s-INT8**: ~25 MB
- **YOLO11m-FP16**: ~40 MB

### Inference Times (iPhone 13, 416×416):
- **YOLO11n-INT8**: 20-25 ms (40-50 FPS)
- **YOLO11s-INT8**: 25-30 ms (33-40 FPS)
- **End-to-end** (detection + scoring): 30-35 ms (28-33 FPS)

### Accuracy:
- **Face-on**: 95-99% PCS
- **Multi-angle**: 90-95% PCS
- **Per-dart detection**: 95%+ recall

### Memory Usage:
- **Idle**: ~50 MB
- **During detection**: ~80-100 MB
- **Peak**: <150 MB

### Battery Impact:
- **Continuous use**: ~2-3 hours
- **With optimization**: 3-4 hours
- **Power consumption**: Low (Neural Engine efficient)

---

## Troubleshooting Guide

### Common Issues:

**Issue: Slow FPS**
- ✅ Verify INT8 quantization applied
- ✅ Check compute units (should be ALL or NEURAL_ENGINE)
- ✅ Reduce input size (320×320)
- ✅ Implement frame skipping
- ✅ Profile with Instruments

**Issue: High Memory Usage**
- ✅ Use autoreleasepool
- ✅ Implement image pooling
- ✅ Reduce batch size (if using)
- ✅ Clear caches regularly

**Issue: Poor Accuracy on Device**
- ✅ Verify model exported correctly
- ✅ Check preprocessing (normalization)
- ✅ Test with sample images
- ✅ Compare with Python inference

**Issue: App Crashes**
- ✅ Check model file integrity
- ✅ Verify CoreML version compatibility
- ✅ Monitor memory usage
- ✅ Add error handling

---

## Deployment Checklist

- [ ] Export model with INT8 quantization
- [ ] Test model in Python (verify accuracy)
- [ ] Integrate model into Xcode project
- [ ] Implement preprocessing pipeline
- [ ] Add camera capture
- [ ] Implement detection + scoring
- [ ] Profile with Instruments
- [ ] Test on multiple devices
- [ ] Optimize for battery
- [ ] Add error handling
- [ ] Test edge cases
- [ ] Submit to TestFlight
- [ ] Collect user feedback
- [ ] Iterate and improve

---

## Resources

**Apple Documentation**:
- Core ML: https://developer.apple.com/documentation/coreml/
- Vision Framework: https://developer.apple.com/documentation/vision/
- Accelerate: https://developer.apple.com/documentation/accelerate/

**Tools**:
- Xcode Instruments: Performance profiling
- coremltools: Model conversion
- TestFlight: Beta testing

**Community**:
- Ultralytics iOS App: Reference implementation
- Apple Developer Forums: Support

## Conclusion

With proper optimization, YOLO11 can achieve **30-60 FPS** on iPhone 13+ with **95%+ accuracy**, making it suitable for real-time dart detection and scoring. The combination of **INT8 quantization**, **Neural Engine acceleration**, and **efficient preprocessing** enables production-ready mobile deployment.
