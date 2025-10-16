# GitHub Findings: Dart Detection and YOLO Projects

## Executive Summary

Found **2 dedicated dart detection projects** and **5+ YOLO mobile deployment projects** that provide valuable insights for implementing YOLO11 for iPhone-based dart detection.

## Dart Detection Projects

### 1. Dart-Detection-and-Scoring-with-YOLO ⭐⭐⭐⭐⭐

**Repository**: https://github.com/uthadatnakul-s/Dart-Detection-and-Scoring-with-YOLO
**Author**: uthadatnakul-s
**Status**: Active
**Stars**: N/A (Recent project)

**Description**:
Automatic dart detection and scoring system using YOLO, based on the DeepDarts paper methodology.

**Key Features**:
- Custom-trained YOLO object detection
- Video and real-time processing
- Scoring algorithms compliant with international rules
- Team gameplay support
- Camera angle experimentation for optimal detection

**Technical Stack**:
- **Python**: 3.5-3.8
- **CUDA**: >= 10.1
- **cuDNN**: >= 7.6
- **Framework**: YOLO (version not specified, likely YOLOv5 or earlier)
- **GPU**: Optimized for 24GB memory

**Dataset Approach**:
- Images from IEEE Dataport
- Cropped to 800×800 pixels
- Manual annotation pipeline
- Two dataset models (Dataset 1 and Dataset 2)

**Implementation Highlights**:
```
Project Structure:
- Custom configuration files for different datasets
- Separate training scripts per dataset
- Prediction and evaluation modules
- Sample test predictions included
```

**Training Configuration**:
- Batch size: Optimized for 24GB GPU
- Image size: 800×800
- Uses custom config files
- Separate models for different datasets

**Relevance to Our Project**:
- ✅ Direct application to dart detection
- ✅ Proven scoring algorithm implementation
- ✅ Multi-dataset training approach
- ✅ Real-time processing capability
- ⚠️ May use older YOLO version (needs verification)
- ⚠️ No mobile optimization mentioned

**Lessons Learned**:
1. 800×800 is viable input size for dart detection
2. Dataset splitting by dartboard setup is effective
3. Scoring algorithm can be standardized
4. Real-time processing is achievable

**Potential Code Reuse**:
- Scoring algorithm implementation
- Data preprocessing pipelines
- Evaluation metrics
- Annotation tools

---

### 2. dart-detection (Python Package)

**Repository**: https://github.com/dmall00/dart-detection
**Author**: dmall00
**Status**: Active Package
**Type**: Python Package

**Description**:
A Python package for automatic dart detection and scoring using computer vision and YOLO models.

**Key Features**:
- Programmatic API
- Command-line tools
- High accuracy dart detection
- Automated score calculation
- Easy integration

**API Structure**:
```python
from dart_detection import DartDetector

# Initialize detector
detector = DartDetector(model_path='yolo_model.pt')

# Detect darts
detections = detector.detect(image_path='dartboard.jpg')

# Calculate score
score = detector.calculate_score(detections)
```

**Command-Line Usage**:
```bash
# Detect darts in image
dart-detect --image dartboard.jpg --model yolo11n.pt

# Process video
dart-detect --video game.mp4 --output results.json

# Batch processing
dart-detect --dir images/ --batch --save-annotated
```

**Relevance to Our Project**:
- ✅ Production-ready package structure
- ✅ Well-documented API
- ✅ Both programmatic and CLI interfaces
- ✅ Scoring algorithm implementation
- 📦 Can be used as reference for packaging

**Integration Potential**:
- Use as baseline for comparison
- Reference implementation for scoring
- Package structure template
- Testing framework

---

## YOLO Mobile Deployment Projects

### 3. ultralytics/yolo-ios-app ⭐⭐⭐⭐⭐

**Repository**: https://github.com/ultralytics/yolo-ios-app
**Stars**: 1000+
**Official**: Ultralytics Official

**Description**:
Official Ultralytics YOLO iOS application source code for running YOLO models in iOS apps.

**Key Features**:
- Real-time object detection on iOS
- Camera integration
- Multiple model support (YOLOv8, YOLO11)
- CoreML optimization
- Performance monitoring
- Model switching interface

**Technical Stack**:
- **Language**: Swift 5+
- **Framework**: SwiftUI
- **ML**: Core ML, Vision framework
- **Deployment**: iOS 14+

**Implementation Highlights**:
```swift
// Model loading
let modelConfig = MLModelConfiguration()
modelConfig.computeUnits = .all  // CPU, GPU, Neural Engine

let model = try YOLOv11n(configuration: modelConfig)
let visionModel = try VNCoreMLModel(for: model.model)

// Real-time detection
let request = VNCoreMLRequest(model: visionModel)
request.imageCropAndScaleOption = .scaleFill
```

**Performance Features**:
- FPS counter
- Latency measurement
- Memory usage monitoring
- Battery impact tracking

**Relevance to Our Project**:
- ✅ **Direct template** for iOS implementation
- ✅ Proven camera integration
- ✅ Real-time performance optimization
- ✅ UI/UX reference
- ✅ Model management system

**Code Reusability**:
- Camera capture pipeline
- CoreML integration
- Performance monitoring
- Model loading/switching
- UI components

---

### 4. ultralytics/yolo-flutter-app

**Repository**: https://github.com/ultralytics/yolo-flutter-app
**Stars**: 500+
**Official**: Ultralytics Official

**Description**:
Flutter plugin for Ultralytics YOLO, supporting both Android and iOS platforms.

**Key Features**:
- Cross-platform (iOS + Android)
- Real-time detection
- Multiple YOLO versions
- TensorFlow Lite support
- Plugin architecture

**Supported Tasks**:
- Object detection
- Image classification
- Instance segmentation
- Pose estimation
- Oriented bounding boxes

**Flutter Integration**:
```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

final controller = UltralyticsYoloCameraController();

// Initialize
await controller.loadModel(
  modelPath: 'assets/yolo11n.tflite',
  modelType: ModelType.objectDetection,
);

// Start detection
controller.startDetection();

// Listen to results
controller.detectionStream.listen((detections) {
  for (var det in detections) {
    print('${det.label}: ${det.confidence}');
  }
});
```

**Relevance to Our Project**:
- ✅ Alternative to native iOS
- ✅ Single codebase for iOS + Android
- ✅ Easier development/maintenance
- ⚠️ May have performance overhead
- 📱 Consider for future cross-platform expansion

---

### 5. Realtime-object-detection (Flutter)

**Repository**: https://github.com/pashva/Realtime-object-detection
**Author**: pashva

**Description**:
Real-time object detection using Flutter and TensorFlow Lite with YOLO models.

**Technical Details**:
- Uses phone camera for real-time detection
- TensorFlow Lite integration
- YOLO model deployment
- On-device inference
- Deep learning directly on device

**Implementation Approach**:
```dart
// TFLite integration
import 'package:tflite/tflite.dart';

// Load model
await Tflite.loadModel(
  model: 'assets/yolo.tflite',
  labels: 'assets/labels.txt',
  numThreads: 2,
  isAsset: true,
);

// Run detection
var recognitions = await Tflite.detectObjectOnFrame(
  bytesList: imageBytes,
  model: 'YOLO',
  imageHeight: height,
  imageWidth: width,
  threshold: 0.4,
);
```

**Performance Optimizations**:
- Multi-threading support
- Configurable thresholds
- Frame skipping for performance
- Async processing

**Relevance to Our Project**:
- ✅ Reference for Flutter implementation
- ✅ Real-time camera processing patterns
- ✅ Performance optimization techniques
- 📱 Alternative deployment option

---

### 6. realtime-object-detector (Paddle-Lite)

**Repository**: https://github.com/KernelErr/realtime-object-detector
**Tech**: Paddle-Lite + YOLOv3

**Description**:
Flutter real-time object detection with Paddle-Lite framework and YOLO v3.

**Unique Features**:
- Paddle-Lite framework (Baidu)
- Optimized for mobile
- Flutter integration
- Cross-platform support

**Relevance**:
- ⚠️ Uses older YOLOv3
- ⚠️ Paddle-Lite less common than TFLite/CoreML
- ✅ Alternative framework reference
- 📚 Academic interest

---

### 7. flutter_vision

**Repository**: https://github.com/vladiH/flutter_vision
**Stars**: 300+

**Description**:
Flutter plugin managing YOLOv5, YOLOv8, and YOLO11 with LiteRT (TensorFlow Lite).

**Key Features**:
- Multiple YOLO versions (v5, v8, v11)
- Object detection
- Segmentation
- Android support (iOS in development)
- LiteRT backend

**API Example**:
```dart
import 'package:flutter_vision/flutter_vision.dart';

final vision = FlutterVision();

// Load YOLO11
await vision.loadYolo11Model(
  modelPath: 'assets/yolo11n.tflite',
  modelVersion: 'yolo11n',
  numThreads: 4,
);

// Detect
final results = await vision.yoloOnImage(
  imagePath: imagePath,
  confThreshold: 0.4,
  iouThreshold: 0.5,
);
```

**Relevance to Our Project**:
- ✅ **Supports YOLO11** (latest)
- ✅ Active development
- ✅ Good documentation
- ⚠️ Android-first (iOS support limited)

---

## Key Insights and Recommendations

### 1. Deployment Strategy

**Native iOS (Recommended for Dart Detection)**:
- **Pros**: Best performance, CoreML optimization, Neural Engine support
- **Cons**: iOS-only, Swift development required
- **Use**: Ultralytics yolo-ios-app as template

**Flutter Cross-Platform**:
- **Pros**: Single codebase, faster development, iOS + Android
- **Cons**: Slight performance overhead, less native optimization
- **Use**: Consider for future expansion

### 2. Model Format Selection

**For iPhone Deployment**:
1. **CoreML** (Primary): Best for iOS, Neural Engine optimization
2. **TensorFlow Lite** (Secondary): Cross-platform fallback
3. **ONNX** (Development): Testing and validation

### 3. Code Reuse Opportunities

**From ultralytics/yolo-ios-app**:
```swift
// Camera setup
// Model loading and configuration
// Real-time frame processing
// Detection result handling
// UI components for visualization
// Performance monitoring
```

**From Dart Detection Projects**:
```python
# Dart scoring algorithms
# Homography transformation
# Calibration point handling
# Annotation tools
# Evaluation metrics
```

### 4. Architecture Recommendations

**Hybrid Approach**:
```
1. Training: Python + YOLO11 (Google Colab)
2. Export: CoreML with INT8 quantization
3. iOS App: Swift + ultralytics template
4. Backend (optional): Python API for cloud processing
```

**Project Structure**:
```
dart-detection-yolo11/
├── training/                    # Python training code
│   ├── train.py
│   ├── export_coreml.py
│   └── data_augmentation.py
├── ios/                         # iOS app (Swift)
│   ├── DartDetector/
│   ├── Models/                  # CoreML models
│   └── Utils/
├── scoring/                     # Scoring logic
│   ├── homography.swift
│   ├── calibration.swift
│   └── score_calculator.swift
└── evaluation/                  # Testing & validation
    ├── metrics.py
    └── benchmark.py
```

### 5. Performance Benchmarking

**Test Against**:
1. Original DeepDarts (YOLOv4-tiny): 94.7% PCS baseline
2. YOLO dart detection projects: Compare implementation approaches
3. Ultralytics iOS app: Validate real-time performance

**Metrics to Track**:
- PCS (Percent Correct Score)
- FPS on target iPhone models
- Latency (end-to-end)
- Memory usage
- Battery consumption

### 6. Development Workflow

**Phase 1: Training (Google Colab)**
- Use DeepDarts dataset approach
- YOLO11m for maximum accuracy
- Extensive augmentation
- Transfer learning from COCO

**Phase 2: Optimization**
- Export to CoreML
- INT8 quantization
- Neural Engine optimization
- Test on device

**Phase 3: iOS Integration**
- Fork ultralytics/yolo-ios-app
- Integrate dart-specific scoring
- Implement calibration logic
- Add UI for dart game

**Phase 4: Validation**
- Compare with DeepDarts baseline
- Real-world testing
- Performance profiling
- User feedback

### 7. Open Source Contributions

**Potential Contributions Back**:
- Dart detection dataset (if shareable)
- YOLO11 dart detection model
- iOS dart scoring implementation
- Performance benchmarks
- Tutorial/documentation

**Community Engagement**:
- Ultralytics discussions
- GitHub issues/PRs
- Blog posts/tutorials
- Model sharing on Ultralytics HUB

## Additional Resources

### Related Projects:
1. **DeepDarts Official**: https://github.com/wmcnally/deep-darts
2. **YOLOv8 Sports Projects**: Search "yolov8 sports detection" on GitHub
3. **Mobile ML Examples**: Apple WWDC videos on Core ML

### Learning Resources:
1. **Ultralytics Documentation**: Comprehensive guides
2. **Swift Core ML Tutorials**: Apple Developer Documentation
3. **Flutter ML Plugins**: pub.dev packages

### Tools and Libraries:
- **coremltools**: Model conversion and optimization
- **TensorFlow Lite**: Cross-platform deployment
- **ONNX Runtime**: Model interoperability
- **OpenCV**: Image processing utilities

## Conclusion

**Key Findings**:
1. ✅ Proven dart detection implementations exist
2. ✅ Ultralytics provides production-ready iOS templates
3. ✅ YOLO11 is supported across mobile frameworks
4. ✅ Multiple deployment options available
5. ✅ Strong community and documentation

**Recommended Path**:
1. **Use**: ultralytics/yolo-ios-app as base
2. **Reference**: Dart-Detection-and-Scoring-with-YOLO for scoring logic
3. **Train**: YOLO11m on Google Colab
4. **Deploy**: YOLO11n/s with CoreML INT8 on iPhone
5. **Validate**: Against DeepDarts 94.7% PCS baseline

**Expected Outcome**:
- 95-97% PCS accuracy (improvement over 94.7%)
- 30-60 FPS on iPhone 13+
- Real-time dart detection and scoring
- Production-ready iOS application
