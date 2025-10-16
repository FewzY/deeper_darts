# Implementation Plan: YOLO11 Dart Detection for iPhone

## Executive Summary

This implementation plan provides a **step-by-step roadmap** for training a YOLO11 model for dart detection optimized for iPhone deployment, targeting **95-99% accuracy** and **30-60 FPS real-time performance**.

**Timeline**: 4-6 weeks
**Primary Tool**: Google Colab (free tier sufficient)
**Target Device**: iPhone 13+ with iOS 15+

## Project Phases

### Phase 1: Dataset Preparation (Week 1)
### Phase 2: Training Setup (Week 1-2)
### Phase 3: Model Training (Week 2-3)
### Phase 4: Model Optimization (Week 3-4)
### Phase 5: iOS Integration (Week 4-5)
### Phase 6: Testing & Validation (Week 5-6)

---

## Phase 1: Dataset Preparation (Week 1)

### 1.1 Dataset Analysis

**Current Dataset Structure**:
```
datasets/
├── labels.pkl                    # 16,050 entries
│   ├── img_folder                # Session folder
│   ├── img_name                  # Image filename
│   ├── bbox                      # Dartboard bounding box
│   └── xy                        # Keypoints (calibration + darts)
├── images/                       # Organized by session
│   ├── d1_02_04_2020/           # 252 images
│   ├── d1_02_06_2020/           # 527 images
│   └── ...                       # 36+ sessions
└── cropped_images/               # Pre-cropped (if available)
```

**Tasks**:
1. **Analyze labels.pkl structure**
```python
import pandas as pd
import numpy as np

# Load labels
df = pd.read_pickle('datasets/labels.pkl')

# Analyze
print(f"Total images: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check keypoint structure
sample = df.iloc[0]
print(f"Keypoints format: {sample['xy']}")
print(f"Bbox format: {sample['bbox']}")

# Analyze distribution
dart_counts = df['xy'].apply(lambda x: len(x) - 4)  # -4 for calibration points
print(f"Dart distribution:\n{dart_counts.value_counts()}")
```

2. **Verify image availability**
```python
import os
from pathlib import Path

# Count total images
image_folders = Path('datasets/images').glob('*')
total_images = sum(len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.JPG')))
                   for folder in image_folders)

print(f"Total images found: {total_images}")
print(f"Match with labels: {total_images == len(df)}")
```

3. **Quality Check**
```python
# Check for missing images
missing = []
for idx, row in df.iterrows():
    img_path = f"datasets/images/{row['img_folder']}/{row['img_name']}"
    if not os.path.exists(img_path):
        missing.append(img_path)

print(f"Missing images: {len(missing)}")
```

---

### 1.2 Convert to YOLO Format

**Create conversion script** (`scripts/convert_to_yolo.py`):

```python
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

def convert_deepdarts_to_yolo(labels_path, images_dir, output_dir):
    """
    Convert DeepDarts format to YOLO11 format.

    Classes:
    0: calibration_point_5_20
    1: calibration_point_13_6
    2: calibration_point_17_3
    3: calibration_point_8_11
    4: dart_tip
    """

    # Load labels
    df = pd.read_pickle(labels_path)

    # Create output directories
    output_path = Path(output_dir)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'test').mkdir(parents=True, exist_ok=True)

    # Split: 80% train, 10% val, 10% test (session-based)
    sessions = df['img_folder'].unique()
    np.random.shuffle(sessions)

    n_train = int(0.8 * len(sessions))
    n_val = int(0.1 * len(sessions))

    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:n_train + n_val]
    test_sessions = sessions[n_train + n_val:]

    def get_split(session):
        if session in train_sessions:
            return 'train'
        elif session in val_sessions:
            return 'val'
        else:
            return 'test'

    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_folder = row['img_folder']
        img_name = row['img_name']
        bbox = row['bbox']  # [x, y, w, h]
        keypoints = row['xy']  # List of [x, y] coordinates

        # Determine split
        split = get_split(img_folder)

        # Read image to get dimensions
        img_path = Path(images_dir) / img_folder / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Crop to dartboard (if using bbox)
        x, y, w, h = bbox
        cropped = img[y:y+h, x:x+w]

        # Save cropped image
        new_img_name = f"{img_folder}_{img_name}"
        img_out_path = output_path / 'images' / split / new_img_name
        cv2.imwrite(str(img_out_path), cropped)

        # Convert keypoints to YOLO format
        # YOLO format: class x_center y_center width height (all normalized)
        # For keypoints, use small bounding boxes (2.5% of image size)
        bbox_size = 0.025  # 2.5% following DeepDarts paper

        yolo_labels = []

        # First 4 keypoints are calibration points (classes 0-3)
        for i in range(4):
            if i < len(keypoints):
                x_norm, y_norm = keypoints[i]
                # Convert from image coordinates to cropped coordinates
                # (assuming keypoints are relative to full image)
                x_abs = x_norm * img_w
                y_abs = y_norm * img_h

                # Relative to crop
                x_crop = (x_abs - x) / w
                y_crop = (y_abs - y) / h

                # Skip if outside crop
                if not (0 <= x_crop <= 1 and 0 <= y_crop <= 1):
                    continue

                # YOLO format: class x_center y_center width height
                yolo_labels.append(f"{i} {x_crop:.6f} {y_crop:.6f} {bbox_size} {bbox_size}")

        # Remaining keypoints are dart tips (class 4)
        for i in range(4, len(keypoints)):
            x_norm, y_norm = keypoints[i]
            x_abs = x_norm * img_w
            y_abs = y_norm * img_h

            x_crop = (x_abs - x) / w
            y_crop = (y_abs - y) / h

            if not (0 <= x_crop <= 1 and 0 <= y_crop <= 1):
                continue

            yolo_labels.append(f"4 {x_crop:.6f} {y_crop:.6f} {bbox_size} {bbox_size}")

        # Write label file
        label_out_path = output_path / 'labels' / split / f"{Path(new_img_name).stem}.txt"
        with open(label_out_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

    # Create data.yaml
    yaml_content = f"""
path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: calibration_5_20
  1: calibration_13_6
  2: calibration_17_3
  3: calibration_8_11
  4: dart_tip
"""

    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"Conversion complete!")
    print(f"Train: {len(list((output_path / 'images' / 'train').glob('*')))} images")
    print(f"Val: {len(list((output_path / 'images' / 'val').glob('*')))} images")
    print(f"Test: {len(list((output_path / 'images' / 'test').glob('*')))} images")

if __name__ == '__main__':
    convert_deepdarts_to_yolo(
        labels_path='datasets/labels.pkl',
        images_dir='datasets/images',
        output_dir='datasets/yolo_format'
    )
```

**Run conversion**:
```bash
python scripts/convert_to_yolo.py
```

**Expected Output**:
```
datasets/yolo_format/
├── data.yaml
├── images/
│   ├── train/          # ~12,800 images (80%)
│   ├── val/            # ~1,600 images (10%)
│   └── test/           # ~1,650 images (10%)
└── labels/
    ├── train/          # ~12,800 .txt files
    ├── val/            # ~1,600 .txt files
    └── test/           # ~1,650 .txt files
```

---

### 1.3 Verify Dataset

**Verification script** (`scripts/verify_dataset.py`):

```python
from ultralytics import YOLO
import cv2
import yaml

# Load data.yaml
with open('datasets/yolo_format/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Visualize samples
def visualize_sample(image_path, label_path, data_config):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        # Convert to absolute coordinates
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)

        # Draw bounding box
        color = (0, 255, 0) if class_id == 4 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        class_name = data_config['names'][class_id]
        cv2.putText(img, class_name, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# Test visualization
import random
from pathlib import Path

train_images = list(Path('datasets/yolo_format/images/train').glob('*'))
sample_img = random.choice(train_images)
sample_label = Path('datasets/yolo_format/labels/train') / f"{sample_img.stem}.txt"

visualized = visualize_sample(str(sample_img), str(sample_label), data_config)
cv2.imwrite('sample_verification.jpg', visualized)
print("Verification image saved as 'sample_verification.jpg'")
```

---

## Phase 2: Training Setup (Week 1-2)

### 2.1 Google Colab Setup

**Create Colab notebook** (`notebooks/yolo11_dart_training.ipynb`):

```python
# ==================================================
# YOLO11 Dart Detection Training Notebook
# ==================================================

# 1. Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Install dependencies
!pip install ultralytics -q
!pip install roboflow -q  # If using Roboflow

# 4. Import libraries
from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path

# 5. Setup directories
save_dir = '/content/drive/MyDrive/yolo11_darts/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

# 6. Upload or copy dataset
# Option A: Upload from local
# Upload datasets/yolo_format.zip and extract

# Option B: Copy from Drive
!cp -r /content/drive/MyDrive/datasets/yolo_format /content/

# 7. Verify dataset
!ls /content/yolo_format/

# 8. Verify data.yaml
with open('/content/yolo_format/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
    print(data_config)
```

---

### 2.2 Model Selection

**Choose model variant**:

```python
# For training: YOLO11m (balanced)
model_name = 'yolo11m.pt'

# For deployment: YOLO11n or YOLO11s
deploy_model_name = 'yolo11n.pt'  # Will distill later

# Download pretrained weights
!yolo download model={model_name}
```

---

### 2.3 Training Configuration

**Create training config**:

```python
# Training hyperparameters
train_config = {
    'model': model_name,
    'data': '/content/yolo_format/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # Adjust based on GPU memory

    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # Augmentation (built-in)
    'mosaic': 1.0,
    'mixup': 0.2,
    'copy_paste': 0.3,
    'degrees': 10.0,
    'translate': 0.2,
    'scale': 0.5,
    'shear': 2.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,

    # Advanced
    'cache': 'ram',  # Cache images in RAM for faster training
    'workers': 8,
    'patience': 50,
    'save_period': 10,
    'project': save_dir,
    'name': 'yolo11m_darts_v1',

    # Device
    'device': 0,  # GPU 0
}

print("Training configuration:")
for key, value in train_config.items():
    print(f"  {key}: {value}")
```

---

## Phase 3: Model Training (Week 2-3)

### 3.1 Initial Training

```python
# Load model
model = YOLO(model_name)

# Train
results = model.train(**train_config)

# Training will save checkpoints to:
# /content/drive/MyDrive/yolo11_darts/yolo11m_darts_v1/
```

**Monitor training**:
```python
# View training plots
from IPython.display import Image, display

results_dir = Path(save_dir) / 'yolo11m_darts_v1'

# Display training curves
display(Image(filename=str(results_dir / 'results.png')))

# Display sample predictions
display(Image(filename=str(results_dir / 'val_batch0_pred.jpg')))
```

---

### 3.2 Evaluation

```python
# Load best model
best_model = YOLO(str(results_dir / 'weights' / 'best.pt'))

# Validate
val_results = best_model.val(
    data='/content/yolo_format/data.yaml',
    split='val'
)

# Print metrics
print(f"mAP50: {val_results.box.map50}")
print(f"mAP50-95: {val_results.box.map}")
print(f"Precision: {val_results.box.p}")
print(f"Recall: {val_results.box.r}")

# Test on test set
test_results = best_model.val(
    data='/content/yolo_format/data.yaml',
    split='test'
)

print(f"\nTest mAP50: {test_results.box.map50}")
```

---

### 3.3 Custom Augmentation (DeepDarts-style)

**Add task-specific augmentation**:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# DeepDarts augmentation pipeline
deepdarts_augmentation = A.Compose([
    # Dartboard rotation (36° steps)
    A.Rotate(limit=(-180, 180), p=0.5, border_mode=cv2.BORDER_CONSTANT),

    # Perspective warping
    A.Perspective(scale=(0.05, 0.1), p=0.5),

    # Small rotations
    A.Rotate(limit=(-2, 2), p=0.5, border_mode=cv2.BORDER_CONSTANT),

    # Flipping
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Integrate with YOLO training (requires custom data loader)
# Note: This is advanced - start with built-in augmentation first
```

---

### 3.4 Hyperparameter Tuning (Optional)

```python
# Auto-tune hyperparameters
model.tune(
    data='/content/yolo_format/data.yaml',
    epochs=30,  # Shorter tuning
    iterations=300,
    optimizer='AdamW',
    plots=False,
    save=False,
    val=False,
)

# The tuned hyperparameters will be printed
# Use them for final training run
```

---

## Phase 4: Model Optimization (Week 3-4)

### 4.1 Export to CoreML

```python
# Export to CoreML with INT8 quantization
best_model.export(
    format='coreml',
    int8=True,        # INT8 quantization
    nms=True,         # Include NMS
    imgsz=416,        # Mobile-optimized size
    keras=False,
    optimize=True,
    half=False,       # INT8 already applied
    dynamic=False,
    simplify=True,
)

# Output: best_int8.mlpackage
```

**Download CoreML model**:
```python
# Copy to Drive for download
coreml_path = results_dir / 'weights' / 'best_int8.mlpackage'
drive_path = Path(save_dir) / 'models' / 'yolo11_darts_int8.mlpackage'
drive_path.parent.mkdir(parents=True, exist_ok=True)

shutil.copytree(coreml_path, drive_path)
print(f"CoreML model saved to: {drive_path}")
```

---

### 4.2 Distillation (Optional)

**Knowledge distillation** (YOLO11m → YOLO11n):

```python
# Train small model with large model as teacher
# (Not directly supported in YOLO11, but can implement)

# Alternative: Train YOLO11n from scratch
student_model = YOLO('yolo11n.pt')

student_results = student_model.train(
    data='/content/yolo_format/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,  # Larger batch for smaller model
    **train_config  # Reuse other configs
)

# Export student model
student_model.export(
    format='coreml',
    int8=True,
    nms=True,
    imgsz=416,
)
```

---

### 4.3 Benchmarking

```python
# Benchmark on validation set
import time

# Test inference speed
model_path = 'best.pt'
model = YOLO(model_path)

# Warm-up
for _ in range(10):
    model.predict('datasets/yolo_format/images/val/sample.jpg', verbose=False)

# Benchmark
times = []
for img_path in list(Path('datasets/yolo_format/images/val').glob('*'))[:100]:
    start = time.time()
    results = model.predict(img_path, verbose=False)
    times.append(time.time() - start)

print(f"Average inference time: {np.mean(times)*1000:.2f} ms")
print(f"FPS: {1/np.mean(times):.2f}")
```

---

## Phase 5: iOS Integration (Week 4-5)

### 5.1 Setup iOS Project

**Use Ultralytics iOS App as template**:

```bash
# Clone template
git clone https://github.com/ultralytics/yolo-ios-app.git
cd yolo-ios-app

# Open in Xcode
open YOLO.xcodeproj
```

---

### 5.2 Integrate CoreML Model

**Copy model to Xcode project**:
1. Download `yolo11_darts_int8.mlpackage` from Drive
2. Drag into Xcode project (target: YOLO app)
3. Verify model is in "Build Phases" → "Copy Bundle Resources"

**Load model in Swift**:
```swift
import CoreML
import Vision

class DartDetector {
    var model: VNCoreMLModel?

    init() {
        setupModel()
    }

    func setupModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolo11_darts_int8", withExtension: "mlpackage") else {
            fatalError("Model not found")
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU

            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            model = try VNCoreMLModel(for: mlModel)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }

    func detect(in image: CIImage, completion: @escaping ([Detection]) -> Void) {
        guard let model = model else { return }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }

            let detections = results.map { observation -> Detection in
                Detection(
                    label: observation.labels.first?.identifier ?? "",
                    confidence: observation.confidence,
                    boundingBox: observation.boundingBox
                )
            }

            completion(detections)
        }

        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        try? handler.perform([request])
    }
}

struct Detection {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}
```

---

### 5.3 Implement Dart Scoring

**Port DeepDarts scoring logic**:

```swift
import simd

class DartScorer {
    func calculateScore(from detections: [Detection]) -> Int {
        // Separate calibration points and darts
        let calibrationPoints = detections.filter { $0.label.contains("calibration") }
        let darts = detections.filter { $0.label == "dart_tip" }

        guard calibrationPoints.count == 4 else {
            return 0  // Need all 4 calibration points
        }

        // Compute homography
        let homography = computeHomography(from: calibrationPoints)

        // Transform dart points
        let transformedDarts = darts.map { dart in
            transformPoint(dart.boundingBox.center, with: homography)
        }

        // Calculate dartboard center and radius
        let center = computeCenter(from: calibrationPoints)
        let radius = computeRadius(from: calibrationPoints, center: center)

        // Score each dart
        var totalScore = 0
        for dart in transformedDarts {
            let score = classifyDart(dart, center: center, radius: radius)
            totalScore += score
        }

        return totalScore
    }

    private func computeHomography(from calibrationPoints: [Detection]) -> matrix_float3x3 {
        // Implement homography calculation
        // Reference: DeepDarts paper Section 4.2
        // Use known calibration point positions on dartboard
        // ... implementation ...
        return matrix_float3x3()
    }

    private func transformPoint(_ point: CGPoint, with homography: matrix_float3x3) -> CGPoint {
        // Apply homography transformation
        // ... implementation ...
        return point
    }

    private func computeCenter(from calibrationPoints: [Detection]) -> CGPoint {
        // Mean of calibration points
        let sum = calibrationPoints.reduce(CGPoint.zero) { result, point in
            CGPoint(x: result.x + point.boundingBox.midX,
                   y: result.y + point.boundingBox.midY)
        }
        return CGPoint(x: sum.x / CGFloat(calibrationPoints.count),
                      y: sum.y / CGFloat(calibrationPoints.count))
    }

    private func computeRadius(from calibrationPoints: [Detection], center: CGPoint) -> CGFloat {
        // Mean distance from center
        let distances = calibrationPoints.map { point in
            distance(center, point.boundingBox.center)
        }
        return distances.reduce(0, +) / CGFloat(distances.count)
    }

    private func classifyDart(_ dartPoint: CGPoint, center: CGPoint, radius: CGFloat) -> Int {
        // Classify based on distance and angle
        // Reference: Standard dartboard scoring
        // ... implementation ...
        return 0  // Placeholder
    }

    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2))
    }
}

extension CGRect {
    var center: CGPoint {
        CGPoint(x: midX, y: midY)
    }
}
```

---

### 5.4 Real-Time Camera Integration

```swift
import AVFoundation

class CameraViewController: UIViewController {
    let dartDetector = DartDetector()
    let dartScorer = DartScorer()

    var captureSession: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .hd1280x720  // 720p

        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
        let videoInput: AVCaptureDeviceInput

        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }

        if captureSession?.canAddInput(videoInput) == true {
            captureSession?.addInput(videoInput)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))

        if captureSession?.canAddOutput(videoOutput) == true {
            captureSession?.addOutput(videoOutput)
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
        previewLayer?.frame = view.layer.bounds
        previewLayer?.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer!)

        captureSession?.startRunning()
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                      didOutput sampleBuffer: CMSampleBuffer,
                      from connection: AVCaptureConnection) {

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Detect darts
        dartDetector.detect(in: ciImage) { [weak self] detections in
            guard let self = self else { return }

            // Calculate score
            let score = self.dartScorer.calculateScore(from: detections)

            DispatchQueue.main.async {
                // Update UI
                self.updateScore(score)
                self.drawDetections(detections)
            }
        }
    }

    func updateScore(_ score: Int) {
        // Update UI with score
    }

    func drawDetections(_ detections: [Detection]) {
        // Draw bounding boxes on preview
    }
}
```

---

## Phase 6: Testing & Validation (Week 5-6)

### 6.1 Accuracy Testing

**Test on holdout dataset**:

```swift
class ModelValidator {
    func validateOnTestSet() {
        let testImages = loadTestImages()  // From test split

        var correctScores = 0
        var totalImages = testImages.count

        for testCase in testImages {
            let predictedScore = predictScore(testCase.image)
            let actualScore = testCase.groundTruthScore

            if predictedScore == actualScore {
                correctScores += 1
            }
        }

        let pcs = Double(correctScores) / Double(totalImages) * 100
        print("PCS (Percent Correct Score): \(pcs)%")
    }
}
```

---

### 6.2 Performance Testing

**Benchmark on device**:

```swift
class PerformanceBenchmark {
    func measureFPS() {
        var frameTimes: [Double] = []

        // Process 100 frames
        for frame in testFrames {
            let start = Date()
            let _ = dartDetector.detect(in: frame)
            let elapsed = Date().timeIntervalSince(start)
            frameTimes.append(elapsed)
        }

        let avgTime = frameTimes.reduce(0, +) / Double(frameTimes.count)
        let fps = 1.0 / avgTime

        print("Average FPS: \(fps)")
        print("Average Latency: \(avgTime * 1000) ms")
    }

    func measureMemory() {
        // Use Xcode Instruments for detailed profiling
    }
}
```

---

### 6.3 Edge Case Testing

**Test scenarios**:
1. ✅ Occluded darts
2. ✅ Edge of dartboard
3. ✅ Various lighting conditions
4. ✅ Different camera angles
5. ✅ Multiple darts clustered
6. ✅ Missing calibration points (partial)

---

## Expected Results

### Accuracy Targets:
- **Face-on dataset**: 95-99% PCS (vs 94.7% baseline)
- **Multi-angle dataset**: 90-95% PCS (vs 84.0% baseline)
- **Per-dart detection**: >95% recall, >90% precision

### Performance Targets:
- **iPhone 13**: 40-60 FPS with INT8
- **iPhone 14/15**: 50-70 FPS with INT8
- **iPhone 15 Pro**: 60+ FPS with W8A8
- **Latency**: <30ms end-to-end
- **Memory**: <100MB peak

### Model Size:
- **Training model** (YOLO11m): ~80MB
- **Deployment model** (YOLO11n-INT8): ~15-20MB

---

## Troubleshooting Guide

### Common Issues:

**Issue: Low accuracy**
- Check data augmentation settings
- Verify annotation quality
- Increase epochs
- Try larger model (YOLO11m instead of YOLO11n)

**Issue: Slow inference on iPhone**
- Ensure INT8 quantization applied
- Reduce input size (320×320 or 416×416)
- Check Neural Engine utilization
- Implement frame skipping

**Issue: High memory usage**
- Use smaller model (YOLO11n)
- Reduce batch size
- Implement image downsampling
- Use autoreleasepool

**Issue: Poor generalization**
- Add more augmentation
- Collect diverse training data
- Use transfer learning
- Implement domain adaptation

---

## Next Steps After Implementation

1. **App Polish**:
   - UI/UX refinement
   - Game mode selection (501, Cricket, etc.)
   - Player statistics
   - Social features

2. **Advanced Features**:
   - Video recording of games
   - Cloud sync
   - Multiplayer support
   - Achievements/leaderboards

3. **Performance Optimization**:
   - Model pruning
   - Quantization-aware training
   - Custom CoreML operations
   - Hardware-specific tuning

4. **Deployment**:
   - TestFlight beta testing
   - App Store submission
   - Marketing materials
   - User feedback collection

---

## Checklist

### Week 1: Dataset
- [ ] Analyze labels.pkl structure
- [ ] Convert to YOLO format
- [ ] Verify dataset quality
- [ ] Create train/val/test splits

### Week 2-3: Training
- [ ] Setup Google Colab environment
- [ ] Configure training parameters
- [ ] Train YOLO11m model
- [ ] Evaluate on validation set
- [ ] Tune hyperparameters
- [ ] Achieve target accuracy

### Week 3-4: Optimization
- [ ] Export to CoreML with INT8
- [ ] Test quantized model accuracy
- [ ] Train/distill YOLO11n variant
- [ ] Benchmark inference speed

### Week 4-5: iOS Integration
- [ ] Setup Xcode project
- [ ] Integrate CoreML model
- [ ] Implement dart scoring logic
- [ ] Build camera interface
- [ ] Test on simulator
- [ ] Test on device

### Week 5-6: Testing
- [ ] Accuracy testing (PCS metric)
- [ ] Performance benchmarking
- [ ] Edge case testing
- [ ] User testing
- [ ] Bug fixes
- [ ] Documentation

---

## Resources

**Code Templates**:
- Conversion script: `scripts/convert_to_yolo.py`
- Training notebook: `notebooks/yolo11_dart_training.ipynb`
- iOS detector: See Phase 5 code samples

**References**:
- DeepDarts paper: Methodology reference
- Ultralytics docs: Training guide
- Apple CoreML: Optimization guide

**Tools**:
- Google Colab: Training environment
- Xcode: iOS development
- Instruments: Performance profiling

## Timeline Summary

```
Week 1: [========] Dataset Preparation
Week 2: [========] Training Setup & Initial Training
Week 3: [========] Training Completion & Evaluation
Week 4: [========] Model Optimization & Export
Week 5: [========] iOS Integration
Week 6: [========] Testing & Validation
```

**Total: 6 weeks from dataset to production-ready iOS app**
