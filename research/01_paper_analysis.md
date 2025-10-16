# DeepDarts Paper Analysis

**Paper**: "DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera"
**Authors**: William McNally, Pascale Walters, Kanav Vats, Alexander Wong, John McPhee
**Institution**: University of Waterloo
**Conference**: CVPR 2021 Workshop
**Code**: https://github.com/wmcnally/deep-darts

## Executive Summary

DeepDarts introduces a novel approach to keypoint detection for automatic dart scoring using a single camera. The key innovation is **modeling keypoints as objects** rather than using traditional heatmap-based regression, enabling detection of multiple keypoints of the same class positioned in close proximity.

## Key Methodology

### 1. Keypoint Detection Approach

**Innovation**: Keypoints as Objects
- Traditional heatmap regression fails when multiple keypoints of the same class appear close together
- Their solution: Use object detection framework with "keypoint bounding boxes"
- Small bounding boxes represent keypoint locations using their centers
- Addresses the overlapping heatmap signals problem

**Network Architecture**:
- Base: YOLOv4-tiny (lightweight for edge deployment)
- Input: RGB image (h×w×3)
- Output: 4 dartboard calibration points + D dart landing positions
- Training: Standard object detection loss (CIoU)
- Classes: 5 total (4 calibration point classes + 1 dart class)

### 2. Three-Stage Pipeline

**Stage 1: Keypoint Detection**
- Deep CNN detects dart tips and 4 calibration points
- Calibration points: Located at intersections of (5&20, 13&6, 17&3, 8&11)
- Keypoint bounding box size: 2.5% of input size (optimal from experiments)

**Stage 2: Homography Transformation**
- Uses detected calibration points to compute homography matrix
- Transforms dart locations from image plane to dartboard plane
- Direct linear transform algorithm for closed-form solution
- Handles any camera angle

**Stage 3: Score Prediction**
- Computes dartboard center as mean of calibration points
- Calculates radius using distance to calibration points
- Classifies dart scores using polar coordinates
- Distance from center + angle from reference direction

## Data Augmentation Strategies

### Task-Specific Augmentation (Critical for Performance)

1. **Dartboard Flipping**
   - Randomly flip horizontally/vertically
   - Keeps calibration points fixed
   - Applies probability: 0.5

2. **Dartboard Rotation**
   - Rotation range: [-180°, 180°]
   - Step size: 36° (best performer, +6.8% PCS on D1)
   - Keeps black section at top
   - Calibration points remain fixed

3. **Small Rotations**
   - Range: [-2°, 2°]
   - Accounts for non-aligned dartboards
   - Applies to all keypoints

4. **Perspective Warping** (Most effective for D2: +7.1% PCS)
   - Randomly perturbs homography matrix H⁻¹
   - Non-diagonal elements scaled by factor ρ ∈ [0, ρ]
   - Generalizes to various camera angles
   - Critical for limited data scenarios

5. **Jitter** (Small Translations)
   - Improvement: +4.2% PCS on D1

## Performance Results

### Dataset 1 (Primary - Face-on Views)
- **Training**: 12,000 images (iPhone XR)
- **Validation**: 1,000 images
- **Test**: 2,000 images
- **Camera**: iPhone XR, face-on view
- **Dartboard**: Winmau Blade 5
- **Test Accuracy**: 94.7% PCS (Percent Correct Score)
- **Inference Speed**:
  - Input 320: 197.3 FPS
  - Input 640: 74.6 FPS
  - Input 800: 55.8 FPS (used for final)

### Dataset 2 (Various Angles - Transfer Learning)
- **Training**: 830 images (DSLR, various angles)
- **Validation**: 70 images
- **Test**: 150 images
- **Camera**: Nikon D3100 DSLR
- **Dartboard**: EastPoint Derbyshire
- **Test Accuracy**: 84.0% PCS
- **Transfer Learning**: +10.0% PCS improvement (67.7% vs 57.7%)

### Key Findings

**Keypoint Bounding Box Size**:
- Too small (<1% or 4.8px): Detrimental to training
- Optimal: 2.5-7.5% of input size
- Final choice: 2.5% at 800px input (20px boxes)

**Input Size vs Speed**:
- 320px: 73.5 PCS, 197.3 FPS
- 480px: 79.8 PCS, 125.0 FPS
- 640px: 82.6 PCS, 74.6 FPS
- 800px: 83.5 PCS, 55.8 FPS (selected)

**Transfer Learning**:
- ImageNet pretraining: +2.4% PCS on D1, +4.0% PCS on D2
- D1→D2 transfer: +10.0% PCS (significant cross-setup knowledge)

## Limitations and Improvement Opportunities

### Identified Limitations:

1. **Occlusion Handling**
   - Most common failure: Missed darts due to occlusion
   - Edge case: Darts obscured by other darts

2. **Edge Scoring Accuracy**
   - Second most common error: Darts on section boundaries
   - Scoring errors when dart is between two sections

3. **Calibration Point Detection**
   - Rare but critical: Missed calibration points due to dart occlusion
   - Recommendation: Train on redundant calibration points

4. **Limited Diversity**
   - Only 2 dartboard setups tested
   - Limited camera angle variety in training
   - Needs "in the wild" deployment testing

5. **Manual Dartboard Cropping**
   - Requires user to manually draw bounding box
   - Not end-to-end automated

### Opportunities for YOLO11 Improvement:

1. **Better Architecture**
   - YOLOv4-tiny (2020) → YOLO11 (2024)
   - 22% fewer parameters with higher accuracy
   - Enhanced feature extraction
   - Better small object detection

2. **Larger Dataset**
   - Current: 16,050 images
   - Target: 50,000+ images with diverse scenarios
   - More camera angles
   - Various dartboard types
   - Different lighting conditions

3. **Mobile Optimization**
   - CoreML export with INT8 quantization
   - Apple Neural Engine acceleration
   - FP16/INT8 mixed precision
   - Model pruning and compression

4. **End-to-End Detection**
   - Remove manual cropping requirement
   - Detect dartboard + keypoints simultaneously
   - Automatic board localization

5. **Redundant Calibration Points**
   - Add more calibration points for robustness
   - Handle partial occlusion better

6. **Advanced Augmentation**
   - Mosaic augmentation (YOLO11 built-in)
   - CutMix, MixUp strategies
   - Synthetic dart generation
   - Domain randomization

## Evaluation Metrics

### Percent Correct Score (PCS)

**Definition**: Percentage of images where predicted total score equals labeled total score

```
PCS = (100/N) × Σ δ(Σ Ŝᵢ - Σ Sᵢ = 0)
```

**Advantages**:
- Easy to interpret (directly relates to game accuracy)
- Accounts for false positives and negatives
- More meaningful than mAP for this application
- User-friendly metric

**Comparison to mAP**:
- mAP is harder to interpret
- PCS directly measures game-level accuracy
- PCS better for end-user understanding

### Additional Metrics to Consider:

1. **Per-Dart Accuracy**
   - Individual dart detection rate
   - Keypoint localization error (pixels)

2. **Calibration Point Accuracy**
   - Detection rate for each calibration point
   - Homography estimation error

3. **Inference Latency**
   - End-to-end processing time
   - FPS on target device (iPhone)

## Technical Implementation Details

### Network Configuration:
- Optimizer: Adam
- Learning rate: 0.001 (initial)
- Schedule: Cosine decay
- Loss: CIoU (Complete IoU)
- Training epochs: 100
- Batch size: 32 (D1), 4 (D2)
- Input size: 800×800

### Inference Configuration:
- IoU threshold: 0.3
- Confidence threshold: 0.25
- Missing calibration handling:
  - 1 missing: Estimate from other 3
  - 2+ missing: Assign score of 0

### Data Collection:
- Games played: 501, Cricket, Around the World
- Lighting: Natural and artificial
- Players: Left and right-handed, beginner to intermediate
- Sessions: 36 (D1), 15 (D2)
- Annotation: Single annotator, custom Python tool
- Annotation accuracy: 97.6% (verified on 1,200 labeled darts)

## Key Takeaways for YOLO11 Implementation

### What to Keep:
1. ✅ Keypoints as objects approach (brilliant innovation)
2. ✅ Homography-based transformation
3. ✅ Task-specific augmentation strategies
4. ✅ PCS evaluation metric
5. ✅ Transfer learning approach

### What to Improve:
1. 🔄 Upgrade to YOLO11 architecture
2. 🔄 Add mobile-specific optimizations (CoreML, quantization)
3. 🔄 Expand dataset size and diversity
4. 🔄 Implement end-to-end detection
5. 🔄 Add redundant calibration points
6. 🔄 Improve occlusion handling

### Novel Opportunities:
1. 🆕 Multi-scale detection for better accuracy
2. 🆕 Attention mechanisms for calibration points
3. 🆕 Temporal consistency (video-based scoring)
4. 🆕 Self-supervised pre-training
5. 🆕 Active learning for data collection

## Citations

McNally, W., Walters, P., Vats, K., Wong, A., & McPhee, J. (2021). DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).

**GitHub**: https://github.com/wmcnally/deep-darts
**Dataset**: Available on IEEE DataPort
