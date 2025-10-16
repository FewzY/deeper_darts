# Academic Papers: Keypoint Detection and Mobile Optimization

## Executive Summary

Recent academic research (2024-2025) shows significant advancements in **keypoint detection**, **mobile-optimized object detection**, and **efficient neural architectures** that are highly relevant to improving dart detection on iPhone.

## 2024-2025 State-of-the-Art Papers

### 1. DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector

**Publication**: arXiv 2404.08928, April 2024
**Authors**: Multiple contributors
**Link**: https://arxiv.org/abs/2404.08928

**Key Contributions**:
- Addresses keypoint clustering issues through **non-max suppression during training**
- Significant performance improvements: 75.9 → 78.3 mAA on IMC2022 challenge
- Better handling of closely-spaced keypoints
- Improved robustness in challenging scenarios

**Relevance to Dart Detection**:
- ✅ Direct application to dart tip detection (closely-spaced keypoints)
- ✅ Non-max suppression strategy can reduce false positives
- ✅ Improved clustering handling for multiple darts

**Key Techniques**:
```python
# Training-time NMS for better keypoint separation
def training_nms(keypoints, heatmap, radius=3):
    # Apply NMS during training to prevent clustering
    # Improves detection of nearby keypoints
    pass
```

**Applicability**: HIGH
- Dart tips are often clustered together
- NMS strategy can improve detection accuracy
- Training methodology is transferable

---

### 2. XFeat: Accelerated Features for Lightweight Image Matching

**Publication**: CVPR 2024
**Focus**: Fast and accurate image matching for lightweight deployment

**Key Features**:
- Optimized for **real-time performance**
- Suitable for **mobile/edge devices**
- Efficient feature extraction
- Low computational cost

**Relevance to Dart Detection**:
- ✅ Lightweight architecture principles
- ✅ Mobile optimization techniques
- ✅ Real-time processing focus
- 📱 Applicable to iPhone deployment

**Performance Metrics**:
- Real-time on mobile processors
- Maintains accuracy with reduced complexity
- Suitable for edge devices

**Key Insights**:
1. Feature pyramid networks for multi-scale detection
2. Efficient attention mechanisms
3. Optimized memory access patterns
4. Quantization-friendly architecture

---

### 3. High-Resolution Human Keypoint Detection Framework

**Publication**: MDPI Algorithms, August 2025
**Technology**: HRNet (High-Resolution Network)
**Link**: https://www.mdpi.com/1999-4893/18/8/533

**Architecture Innovation**:
- Maintains **high-resolution representations** throughout network
- Effective **multi-scale feature fusion**
- Real-time capability demonstrated
- Suitable for edge deployment

**Applications**:
- Intelligent surveillance
- Sports analysis
- Human-computer interaction
- **Pose estimation** (relevant to keypoint detection)

**Relevance to Dart Detection**:
- ✅ High-resolution feature maps preserve fine details
- ✅ Multi-scale fusion helps with various dart sizes
- ✅ Real-time performance on edge devices
- ✅ Proven in sports analysis domain

**Architecture Principles**:
```
HRNet Design:
- Parallel high-to-low resolution subnetworks
- Repeated multi-scale fusion
- Maintains high resolution throughout
- Better for small object/keypoint detection
```

**Applicability to YOLO11**:
- Inspire backbone modifications
- Multi-scale feature fusion strategies
- High-resolution pathway for dart tips
- Could improve calibration point detection

---

### 4. SphereCraft Dataset for Spherical Keypoint Detection

**Publication**: WACV 2024
**Link**: https://openaccess.thecvf.com/content/WACV2024/

**Key Contribution**:
- Specialized dataset for **spherical keypoint detection**
- Various keypoint detectors evaluated
- Ground truth correspondences provided
- Camera pose estimation integration

**Relevance to Dart Detection**:
- ✅ Circular dartboard shares geometric properties
- ✅ Multiple keypoint detector comparisons
- ✅ Ground truth methodology insights
- 📊 Evaluation framework reference

**Dataset Insights**:
- Importance of geometric constraints
- Multi-view consistency
- Annotation strategies for circular objects
- Evaluation metrics design

**Applicable Techniques**:
1. Geometric constraint enforcement
2. Multi-view consistency checks
3. Circular object annotation methods
4. Specialized evaluation metrics

---

### 5. Enhancement of Speed and Accuracy Trade-Off for Sports Ball Detection

**Publication**: MDPI Sensors 2021, PMC8124271
**Focus**: Fast moving, small objects in real-time
**Application**: Sports ball detection in videos

**Key Findings**:
- **Data augmentation** critical for small objects
- **Copy-paste augmentation** increases object occurrence
- Mosaic augmentation maximizes dataset utility
- Real-time processing achieved

**Relevance to Dart Detection**:
- ✅ Small object detection (dart tips)
- ✅ Real-time processing requirements
- ✅ Data augmentation strategies
- ✅ Sports domain application

**Augmentation Strategies**:
```python
# Copy-paste for small objects
def copy_paste_augmentation(image, darts, n_copies=3):
    # Copy dart instances multiple times
    # Randomize locations and scales
    # Increases training diversity
    return augmented_image

# Mosaic augmentation
def mosaic_augmentation(images, labels):
    # Combine 4 images into one
    # Increases scale variety
    # Better for small datasets
    return mosaic_image
```

**Performance Improvements**:
- Enhanced small object detection
- Better generalization
- Reduced dataset requirements
- Real-time inference maintained

---

### 6. ARSOD-YOLO: Enhancing Small Target Detection for Remote Sensing

**Publication**: PMC11644057, 2024
**Focus**: Aerial/remote sensing with small targets

**Key Innovations**:
- Specialized architecture for **small target detection**
- Multi-scale feature aggregation
- Attention mechanisms for small objects
- Optimized for challenging scenarios

**Relevance to Dart Detection**:
- ✅ Dart tips are relatively small in image
- ✅ Multi-scale detection strategies
- ✅ Attention mechanisms for calibration points
- 📊 Evaluation on small objects

**Architecture Enhancements**:
1. **Small Object Detection Head**
   - Higher resolution feature maps
   - Specialized anchor sizes
   - Enhanced receptive field

2. **Attention Modules**
   - Spatial attention for keypoints
   - Channel attention for features
   - Multi-scale attention fusion

3. **Feature Pyramid Network**
   - Bottom-up pathway
   - Top-down pathway
   - Lateral connections

**Applicability to Dart Detection**:
```
YOLO11 + ARSOD-YOLO concepts:
- Add high-resolution detection head (P2/P3 levels)
- Implement spatial attention for dart tips
- Optimize anchor sizes for dart/calibration detection
- Multi-scale feature aggregation
```

---

## Mobile Optimization Research

### 7. Neural Network Compression Techniques (2024)

**Focus**: Model compression for mobile deployment

**Key Techniques**:

**1. Quantization**
- INT8 quantization: 75% size reduction
- Mixed precision: FP16/INT8 combination
- Quantization-aware training (QAT)
- Post-training quantization (PTQ)

**2. Pruning**
- Structured pruning: Remove entire channels
- Unstructured pruning: Remove individual weights
- Iterative magnitude pruning
- Dynamic network surgery

**3. Knowledge Distillation**
- Teacher-student framework
- Large model → Small model
- Maintain accuracy with reduced size
- Feature-level distillation

**4. Neural Architecture Search (NAS)**
- Automated architecture optimization
- Hardware-aware NAS
- Mobile-specific constraints
- Latency-accuracy trade-off optimization

**Relevance to iPhone Deployment**:
- ✅ All techniques applicable to YOLO11
- ✅ CoreML supports quantization and pruning
- ✅ Knowledge distillation for YOLO11l → YOLO11n
- 📱 Optimizes for Apple Neural Engine

---

### 8. OpenPose Lightweight Variants (2024-2025)

**Publication**: Multiple sources, 2024-2025
**Link**: https://capalearning.com/2024/10/28/the-complete-guide-to-openpose-in-2025/

**Key Features**:
- Optimized for **Edge AI**
- On-device Edge ML Inference
- Real-time pose estimation
- Mobile-friendly architecture

**Relevance**:
- ✅ Keypoint detection optimization
- ✅ Edge deployment strategies
- ✅ Real-time processing techniques
- 📚 Reference architecture

**Mobile Optimization Techniques**:
1. Depthwise separable convolutions
2. Inverted residual blocks
3. Efficient channel shuffling
4. Hardware-aware optimizations

---

## Data Augmentation Research (2024)

### 9. YOLOv8 Data Augmentation for Small Datasets

**Sources**: Multiple GitHub issues, tutorials, 2024

**Best Practices for Small Datasets** (10-200 images):

**1. Mosaic Augmentation**
```python
# YOLOv8/YOLO11 built-in
mosaic_prob = 1.0  # Always apply for small datasets
```
- Combines 4 images
- Increases scale variety
- Maximizes dataset utility
- **Critical for small datasets**

**2. Copy-Paste Augmentation**
```python
# Specifically for small objects
copy_paste_prob = 0.5
```
- Copies object instances
- Randomizes placement
- Increases object count
- Better diversity

**3. Geometric Augmentation**
```python
augmentation_config = {
    'degrees': 10.0,      # Rotation
    'translate': 0.2,     # Translation
    'scale': 0.5,         # Scaling
    'shear': 2.0,         # Shearing
    'flipud': 0.5,        # Vertical flip
    'fliplr': 0.5,        # Horizontal flip
}
```

**4. Color Augmentation**
```python
color_augmentation = {
    'hsv_h': 0.015,       # Hue
    'hsv_s': 0.7,         # Saturation
    'hsv_v': 0.4,         # Value
}
```

**5. Advanced Techniques**
- MixUp: Blend two images
- CutMix: Cut and paste regions
- Auto-augmentation: Learned policies
- Albumentations integration

**Small Dataset Strategy** (DeepDarts scenario: 16k images):
```python
# For 16k images, moderate augmentation
train_config = {
    'mosaic': 1.0,        # Always apply
    'mixup': 0.2,         # Occasionally
    'copy_paste': 0.3,    # For darts
    'augment': True,      # Enable all
}
```

---

### 10. Domain-Specific Augmentation Research

**Finding**: Task-specific augmentation significantly outperforms generic approaches

**DeepDarts Augmentation** (Proven effective):
1. Dartboard rotation (36° steps): +6.8% PCS
2. Perspective warping: +7.1% PCS on multi-angle
3. Small rotations: +4.6% PCS
4. Dartboard flipping: +5.6% PCS

**Lesson**: Combine YOLO11 built-in + task-specific augmentation

**Recommended Strategy**:
```python
# YOLO11 built-in (mosaic, mixup, etc.)
yolo_augmentation = True

# Task-specific (DeepDarts proven)
custom_augmentation = {
    'dartboard_rotation_36deg': 0.5,
    'perspective_warping': 0.5,
    'small_rotations_2deg': 0.5,
    'dartboard_flip': 0.5,
}

# Combined approach
total_augmentation_prob = 0.8
```

---

## Transfer Learning Research

### 11. Few-Shot Keypoint Detection (CVPR 2022)

**Repository**: https://github.com/AlanLuSun/Few-shot-keypoint-detection
**Paper**: Few-shot keypoint detector with uncertainty learning

**Key Contribution**:
- Keypoint detection for **unseen species**
- Few-shot learning approach
- Uncertainty estimation
- Better generalization

**Relevance to Dart Detection**:
- ✅ Transfer learning for new dartboards
- ✅ Few-shot adaptation techniques
- ✅ Uncertainty-aware predictions
- 📊 Cross-domain generalization

**Techniques**:
1. Meta-learning for keypoint detection
2. Support-query framework
3. Uncertainty-based refinement
4. Domain adaptation

**Applicability**:
- Train on Dataset 1 (face-on)
- Few-shot adapt to Dataset 2 (multi-angle)
- Better cross-dartboard generalization
- Reduced annotation requirements

---

## Implementation Recommendations

### Combine Research Findings:

**1. Architecture (from HRNet, ARSOD-YOLO)**:
- High-resolution detection pathway
- Multi-scale feature fusion
- Small object detection head
- Spatial attention for keypoints

**2. Training (from DeDoDe v2, Few-Shot)**:
- Training-time NMS for clustering
- Meta-learning for generalization
- Uncertainty-aware losses
- Few-shot domain adaptation

**3. Augmentation (from Sports Detection, YOLOv8)**:
- Mosaic augmentation (built-in)
- Copy-paste for darts
- Task-specific transformations (DeepDarts)
- Combined strategy

**4. Optimization (from Mobile ML Research)**:
- INT8 quantization
- Knowledge distillation (YOLO11l → YOLO11n)
- Pruning for efficiency
- Hardware-aware optimization

### Proposed YOLO11 Training Pipeline:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolo11m.pt')

# Advanced training configuration
results = model.train(
    data='dart_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,

    # Built-in augmentation
    mosaic=1.0,           # Always apply
    mixup=0.2,            # Occasionally
    copy_paste=0.3,       # For small objects

    # Optimization
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # Advanced features
    patience=50,          # Early stopping
    save_period=10,       # Save checkpoints

    # Device
    device=0,             # GPU
)

# Export optimized model
model.export(
    format='coreml',
    int8=True,            # INT8 quantization
    nms=True,             # Include NMS
    imgsz=416,            # Mobile-optimized size
)
```

---

## Key Research Insights

### Accuracy Improvements Expected:

1. **Architecture**: +2-3% (HRNet principles, multi-scale)
2. **Training**: +1-2% (NMS during training, meta-learning)
3. **Augmentation**: +3-5% (combined strategy)
4. **YOLO11 Base**: +2-3% (vs YOLOv4-tiny)

**Total Expected Improvement**: +8-13% over DeepDarts baseline
- Current: 94.7% PCS (D1), 84.0% PCS (D2)
- Target: **98-99% PCS (D1)**, **92-95% PCS (D2)**

### Mobile Optimization Expected:

1. **Model Size**: 80MB → 15-20MB (INT8 + pruning)
2. **Inference Speed**: 2-3x faster than unoptimized
3. **Memory Usage**: 50% reduction
4. **Battery Impact**: Minimal with Neural Engine

---

## Academic Resources

### Papers with Code:
- Keypoint Detection: https://paperswithcode.com/task/keypoint-detection/
- Object Detection: https://paperswithcode.com/task/object-detection/
- Mobile ML: https://paperswithcode.com/task/model-compression/

### Conferences:
- **CVPR**: Computer Vision and Pattern Recognition
- **ICCV**: International Conference on Computer Vision
- **ECCV**: European Conference on Computer Vision
- **WACV**: Winter Conference on Applications of Computer Vision
- **NeurIPS**: Neural Information Processing Systems

### Journals:
- **TPAMI**: IEEE Transactions on Pattern Analysis and Machine Intelligence
- **IJCV**: International Journal of Computer Vision
- **arXiv**: Pre-print server for latest research

---

## Conclusion

**Key Academic Findings**:
1. ✅ Multi-scale detection critical for keypoints
2. ✅ Training-time NMS improves clustering
3. ✅ Task-specific augmentation outperforms generic
4. ✅ INT8 quantization viable with QAT
5. ✅ Transfer learning highly effective

**Expected Research-Driven Improvements**:
- **95-99% PCS** on face-on dataset (vs 94.7%)
- **90-95% PCS** on multi-angle dataset (vs 84.0%)
- **30-60 FPS** on iPhone 13+ (real-time)
- **15-20 MB** model size (deployable)

**Research Integration Strategy**:
1. YOLO11 base (state-of-the-art 2024)
2. + HRNet multi-scale principles
3. + DeDoDe v2 training strategies
4. + Sports detection augmentation
5. + Mobile optimization techniques
6. = **Best-in-class dart detection system**
