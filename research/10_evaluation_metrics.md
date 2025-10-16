# Evaluation Metrics: Measuring and Improving Performance

## Executive Summary

Comprehensive guide for evaluating YOLO11 dart detection performance using **DeepDarts Percent Correct Score (PCS)** metric and **standard computer vision metrics**, with strategies for continuous improvement.

**Primary Metric**: PCS (Percent Correct Score) - Game-level accuracy
**Secondary Metrics**: mAP, Precision, Recall, Per-dart accuracy
**Target**: 95-99% PCS on face-on, 90-95% on multi-angle

---

## Part 1: Percent Correct Score (PCS)

### 1.1 Definition

**PCS**: Percentage of dartboard images where the predicted total score equals the actual total score.

**Formula**:
```
PCS = (100 / N) × Σ δ(Σ Ŝᵢ - Σ Sᵢ = 0)

Where:
- N = total number of test images
- Ŝᵢ = predicted score for dart i
- Sᵢ = actual score for dart i
- δ = indicator function (1 if true, 0 if false)
```

**Why PCS?**:
- ✅ Easy to interpret (directly measures game accuracy)
- ✅ Accounts for false positives AND false negatives
- ✅ More meaningful than mAP for this application
- ✅ Matches end-user expectations

---

### 1.2 Implementation

**PCS Calculator** (`scripts/calculate_pcs.py`):

```python
"""
Calculate Percent Correct Score (PCS) for dart detection.
"""

import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO

class DartScorer:
    """Calculate dart scores from detections."""

    def __init__(self):
        # Dartboard geometry (standard dimensions)
        self.dartboard_radius = 170  # mm (outer wire)
        self.double_ring_outer = 170
        self.double_ring_inner = 162
        self.treble_ring_outer = 107
        self.treble_ring_inner = 99
        self.bull_outer = 15.9
        self.bull_inner = 6.35

        # Sector angles (degrees from top, clockwise)
        self.sector_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        self.sector_angle = 360 / 20  # 18 degrees per sector

    def calculate_score(self, detections):
        """
        Calculate total score from detections.

        Args:
            detections: List of Detection objects

        Returns:
            total_score: Integer (0 if error)
        """
        # Separate calibration points and darts
        calibration_points = [d for d in detections if d.is_calibration]
        darts = [d for d in detections if d.is_dart]

        # Need all 4 calibration points
        if len(calibration_points) != 4:
            return 0

        # Compute homography
        H = self._compute_homography(calibration_points)

        if H is None:
            return 0

        # Transform dart positions
        transformed_darts = []
        for dart in darts:
            point = self._transform_point(dart.center, H)
            if point is not None:
                transformed_darts.append(point)

        # Calculate board center and radius
        center = self._compute_center(calibration_points, H)
        radius = self._compute_radius(calibration_points, H, center)

        # Score each dart
        total_score = 0
        for dart_point in transformed_darts:
            score = self._score_dart(dart_point, center, radius)
            total_score += score

        return total_score

    def _compute_homography(self, calibration_points):
        """Compute homography matrix from calibration points."""
        # Known positions on dartboard (in mm)
        # Positions at intersections of sections
        known_positions = np.array([
            [0, -self.double_ring_outer],      # Top (5-20)
            [self.double_ring_outer, 0],       # Right (13-6)
            [0, self.double_ring_outer],       # Bottom (17-3)
            [-self.double_ring_outer, 0],      # Left (8-11)
        ], dtype=np.float32)

        # Detected positions (in image coordinates)
        detected_positions = np.array([
            [cp.center[0], cp.center[1]] for cp in sorted(
                calibration_points, key=lambda x: x.class_id
            )
        ], dtype=np.float32)

        # Compute homography
        if len(detected_positions) == 4:
            H, _ = cv2.findHomography(detected_positions, known_positions)
            return H

        return None

    def _transform_point(self, point, H):
        """Transform point using homography."""
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed = H @ point_homogeneous

        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
            return (x, y)

        return None

    def _compute_center(self, calibration_points, H):
        """Compute dartboard center."""
        transformed_points = []
        for cp in calibration_points:
            point = self._transform_point(cp.center, H)
            if point is not None:
                transformed_points.append(point)

        if len(transformed_points) > 0:
            center_x = np.mean([p[0] for p in transformed_points])
            center_y = np.mean([p[1] for p in transformed_points])
            return (center_x, center_y)

        return (0, 0)

    def _compute_radius(self, calibration_points, H, center):
        """Compute dartboard radius."""
        transformed_points = []
        for cp in calibration_points:
            point = self._transform_point(cp.center, H)
            if point is not None:
                transformed_points.append(point)

        if len(transformed_points) > 0:
            distances = [np.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2)
                        for p in transformed_points]
            return np.mean(distances)

        return self.double_ring_outer

    def _score_dart(self, dart_point, center, radius):
        """Score a single dart based on position."""
        # Calculate distance from center
        dx = dart_point[0] - center[0]
        dy = dart_point[1] - center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate angle (0° at top, clockwise)
        angle = np.degrees(np.arctan2(dx, -dy))
        if angle < 0:
            angle += 360

        # Determine sector
        sector_index = int((angle + self.sector_angle/2) / self.sector_angle) % 20
        sector_value = self.sector_numbers[sector_index]

        # Normalize distance by radius
        norm_distance = distance / radius * self.double_ring_outer

        # Determine scoring region
        if norm_distance <= self.bull_inner:
            return 50  # Double bull
        elif norm_distance <= self.bull_outer:
            return 25  # Bull
        elif self.treble_ring_inner <= norm_distance <= self.treble_ring_outer:
            return sector_value * 3  # Treble
        elif self.double_ring_inner <= norm_distance <= self.double_ring_outer:
            return sector_value * 2  # Double
        elif norm_distance < self.treble_ring_inner:
            return sector_value  # Inner single
        elif norm_distance < self.double_ring_inner:
            return sector_value  # Outer single
        else:
            return 0  # Outside dartboard

class PCScalculator:
    """Calculate PCS metric."""

    def __init__(self, model_path, test_data_yaml):
        self.model = YOLO(model_path)
        self.test_data_yaml = test_data_yaml
        self.scorer = DartScorer()

    def calculate_pcs(self):
        """Calculate PCS on test set."""
        import yaml

        # Load test set
        with open(self.test_data_yaml, 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config['path'])
        test_img_dir = data_path / 'images' / 'test'
        test_label_dir = data_path / 'labels' / 'test'

        test_images = list(test_img_dir.glob('*'))

        correct_scores = 0
        total_images = 0
        errors = []

        print(f"Evaluating on {len(test_images)} test images...")

        for img_path in tqdm(test_images):
            # Load ground truth
            label_path = test_label_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            ground_truth_score = self._get_ground_truth_score(label_path)

            # Predict
            results = self.model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.25,
                iou=0.3,
                verbose=False,
            )

            # Convert to detections
            detections = self._parse_results(results[0])

            # Calculate predicted score
            predicted_score = self.scorer.calculate_score(detections)

            # Compare
            total_images += 1
            if predicted_score == ground_truth_score:
                correct_scores += 1
            else:
                errors.append({
                    'image': img_path.name,
                    'predicted': predicted_score,
                    'actual': ground_truth_score,
                    'difference': predicted_score - ground_truth_score
                })

        # Calculate PCS
        pcs = (correct_scores / total_images) * 100 if total_images > 0 else 0

        # Print results
        print(f"\n{'='*60}")
        print(f"PCS (Percent Correct Score): {pcs:.2f}%")
        print(f"Correct: {correct_scores}/{total_images}")
        print(f"Errors: {len(errors)}")
        print(f"{'='*60}")

        # Print error analysis
        if errors:
            print(f"\nError Analysis:")
            print(f"  Mean error: {np.mean([e['difference'] for e in errors]):.2f} points")
            print(f"  Std error: {np.std([e['difference'] for e in errors]):.2f} points")

            # Print worst errors
            errors_sorted = sorted(errors, key=lambda x: abs(x['difference']), reverse=True)
            print(f"\n  Top 5 Worst Errors:")
            for i, error in enumerate(errors_sorted[:5], 1):
                print(f"    {i}. {error['image']}: "
                      f"predicted={error['predicted']}, "
                      f"actual={error['actual']}, "
                      f"diff={error['difference']}")

        return pcs, errors

    def _get_ground_truth_score(self, label_path):
        """Get ground truth score from label file."""
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Count darts (class_id == 4)
        n_darts = sum(1 for line in labels if int(line.split()[0]) == 4)

        # For simplicity, assume average of 20 points per dart
        # In practice, you would need to store actual scores
        # This is a placeholder - actual implementation needs ground truth scores
        return n_darts * 20  # Placeholder

    def _parse_results(self, result):
        """Parse YOLO results to Detection objects."""
        detections = []

        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            # Calculate center
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2

            detection = Detection(
                class_id=class_id,
                confidence=confidence,
                center=(center_x, center_y),
                is_calibration=(0 <= class_id <= 3),
                is_dart=(class_id == 4)
            )

            detections.append(detection)

        return detections

class Detection:
    """Detection object."""
    def __init__(self, class_id, confidence, center, is_calibration, is_dart):
        self.class_id = class_id
        self.confidence = confidence
        self.center = center
        self.is_calibration = is_calibration
        self.is_dart = is_dart

# Usage
if __name__ == '__main__':
    calculator = PCScalculator(
        model_path='runs/yolo11m_darts_v1/weights/best.pt',
        test_data_yaml='datasets/yolo_format/data.yaml'
    )

    pcs, errors = calculator.calculate_pcs()
```

---

## Part 2: Standard Computer Vision Metrics

### 2.1 Mean Average Precision (mAP)

**Definition**: Average precision across all classes and IoU thresholds.

**Metrics**:
- **mAP@0.5**: IoU threshold = 0.5 (lenient)
- **mAP@0.50:0.95**: IoU thresholds from 0.5 to 0.95 (strict)

**Calculation** (built-in with YOLO):
```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Validate
results = model.val(data='data.yaml', split='test')

print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.50:0.95: {results.box.map:.4f}")
```

**Target Values**:
- mAP@0.5: >0.90 (good detection)
- mAP@0.50:0.95: >0.70 (good localization)

---

### 2.2 Precision and Recall

**Precision**: What proportion of positive detections are correct?
```
Precision = TP / (TP + FP)
```

**Recall**: What proportion of actual objects are detected?
```
Recall = TP / (TP + FN)
```

**Per-Class Metrics**:
```python
# Get per-class metrics
class_names = ['calib_5_20', 'calib_13_6', 'calib_17_3', 'calib_8_11', 'dart_tip']

for i, class_name in enumerate(class_names):
    precision = results.box.p[i]
    recall = results.box.r[i]
    ap = results.box.ap[i]

    print(f"{class_name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AP@0.5: {ap[0]:.4f}")
```

**Target Values**:
- Precision: >0.90 (few false positives)
- Recall: >0.95 (few missed detections)

---

### 2.3 F1 Score

**Definition**: Harmonic mean of precision and recall.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Calculation**:
```python
f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1:.4f}")
```

**Target**: F1 > 0.92

---

## Part 3: Task-Specific Metrics

### 3.1 Per-Dart Detection Rate

**Definition**: Percentage of darts correctly detected.

**Calculation**:
```python
def calculate_dart_detection_rate(predictions, ground_truth):
    """
    Calculate per-dart detection rate.

    Args:
        predictions: List of predicted dart positions
        ground_truth: List of actual dart positions

    Returns:
        detection_rate: Float (0-1)
    """
    n_ground_truth = len(ground_truth)
    n_detected = 0

    for gt_dart in ground_truth:
        # Find closest prediction
        min_distance = float('inf')
        for pred_dart in predictions:
            distance = np.linalg.norm(
                np.array(gt_dart) - np.array(pred_dart)
            )
            min_distance = min(min_distance, distance)

        # Consider detected if within threshold
        threshold = 10  # pixels
        if min_distance < threshold:
            n_detected += 1

    detection_rate = n_detected / n_ground_truth if n_ground_truth > 0 else 0
    return detection_rate
```

**Target**: >95% detection rate

---

### 3.2 Calibration Point Detection Rate

**Definition**: Percentage of images with all 4 calibration points detected.

**Calculation**:
```python
def calculate_calibration_rate(test_results):
    """Calculate calibration point detection rate."""
    total_images = 0
    complete_calibration = 0

    for result in test_results:
        total_images += 1

        # Count calibration points (class_id 0-3)
        calibration_points = [
            det for det in result.detections
            if 0 <= det.class_id <= 3
        ]

        if len(calibration_points) == 4:
            complete_calibration += 1

    rate = complete_calibration / total_images if total_images > 0 else 0
    return rate
```

**Target**: >98% (critical for scoring)

---

### 3.3 Localization Error

**Definition**: Average pixel distance between predicted and actual positions.

**Calculation**:
```python
def calculate_localization_error(predictions, ground_truth):
    """Calculate average localization error in pixels."""
    errors = []

    for gt_dart in ground_truth:
        # Find closest prediction
        min_distance = float('inf')
        for pred_dart in predictions:
            distance = np.linalg.norm(
                np.array(gt_dart) - np.array(pred_dart)
            )
            min_distance = min(min_distance, distance)

        errors.append(min_distance)

    mean_error = np.mean(errors) if errors else 0
    std_error = np.std(errors) if errors else 0

    return mean_error, std_error
```

**Target**: <5 pixels mean error

---

## Part 4: Performance Metrics

### 4.1 Inference Speed

**FPS (Frames Per Second)**:
```python
import time

def measure_fps(model, test_images, num_samples=100):
    """Measure inference FPS."""
    times = []

    # Warm-up
    for _ in range(10):
        model.predict(test_images[0], verbose=False)

    # Measure
    for img in test_images[:num_samples]:
        start = time.time()
        model.predict(img, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f"Average FPS: {fps:.2f}")
    print(f"Average latency: {avg_time*1000:.2f} ms")

    return fps
```

**Targets**:
- Desktop (Python): 50-100 FPS
- iPhone (CoreML): 30-60 FPS

---

### 4.2 Model Size

```python
import os

def get_model_size(model_path):
    """Get model size in MB."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)

    print(f"Model size: {size_mb:.2f} MB")
    return size_mb

# PyTorch model
pt_size = get_model_size('best.pt')

# CoreML model (directory)
def get_directory_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_directory_size(entry.path)
    return total

coreml_size = get_directory_size('best_int8.mlpackage') / (1024 * 1024)
print(f"CoreML model size: {coreml_size:.2f} MB")
```

**Targets**:
- PyTorch (YOLO11m): ~80 MB
- CoreML INT8 (YOLO11n): ~15-20 MB

---

## Part 5: Comprehensive Evaluation

### 5.1 Evaluation Script

**Complete Evaluation** (`scripts/comprehensive_evaluation.py`):

```python
"""
Comprehensive evaluation of dart detection model.
"""

class ComprehensiveEvaluator:
    def __init__(self, model_path, test_data_yaml):
        self.model = YOLO(model_path)
        self.test_data_yaml = test_data_yaml
        self.pcs_calculator = PCScalculator(model_path, test_data_yaml)

    def evaluate_all(self):
        """Run all evaluations."""
        results = {}

        print("=" * 80)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 80)

        # 1. Standard metrics (mAP, Precision, Recall)
        print("\n1. Standard Computer Vision Metrics")
        print("-" * 80)
        val_results = self.model.val(data=self.test_data_yaml, split='test')

        results['mAP@0.5'] = float(val_results.box.map50)
        results['mAP@0.50:0.95'] = float(val_results.box.map)
        results['Precision'] = float(val_results.box.mp)
        results['Recall'] = float(val_results.box.mr)
        results['F1'] = 2 * results['Precision'] * results['Recall'] / \
                       (results['Precision'] + results['Recall'])

        print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
        print(f"  mAP@0.50:0.95: {results['mAP@0.50:0.95']:.4f}")
        print(f"  Precision: {results['Precision']:.4f}")
        print(f"  Recall: {results['Recall']:.4f}")
        print(f"  F1 Score: {results['F1']:.4f}")

        # 2. PCS (Primary metric)
        print("\n2. Percent Correct Score (PCS)")
        print("-" * 80)
        pcs, errors = self.pcs_calculator.calculate_pcs()
        results['PCS'] = pcs

        # 3. Inference speed
        print("\n3. Inference Speed")
        print("-" * 80)
        fps = self.measure_inference_speed()
        results['FPS'] = fps

        # 4. Model size
        print("\n4. Model Size")
        print("-" * 80)
        model_size = self.get_model_size()
        results['Model_Size_MB'] = model_size

        # 5. Summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

        # 6. Pass/Fail
        print("\n" + "=" * 80)
        print("TARGETS")
        print("=" * 80)

        targets = {
            'PCS': (95.0, '>'),
            'mAP@0.5': (0.90, '>'),
            'Precision': (0.90, '>'),
            'Recall': (0.95, '>'),
            'F1': (0.92, '>'),
            'FPS': (30.0, '>'),  # For mobile
        }

        all_passed = True
        for metric, (target, comp) in targets.items():
            if metric in results:
                value = results[metric]
                passed = (value > target) if comp == '>' else (value >= target)
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {metric}: {value:.2f} (target: {comp}{target}) - {status}")

                if not passed:
                    all_passed = False

        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL TARGETS MET! Model ready for deployment.")
        else:
            print("⚠️  Some targets not met. Consider further training/optimization.")
        print("=" * 80)

        return results

    def measure_inference_speed(self):
        """Measure inference speed."""
        import yaml
        from pathlib import Path

        # Load test images
        with open(self.test_data_yaml, 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config['path'])
        test_images = list((data_path / 'images' / 'test').glob('*'))[:100]

        times = []

        # Warm-up
        for _ in range(10):
            self.model.predict(test_images[0], verbose=False)

        # Measure
        for img_path in test_images:
            start = time.time()
            self.model.predict(img_path, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        print(f"  Average FPS: {fps:.2f}")
        print(f"  Average latency: {avg_time*1000:.2f} ms")

        return fps

    def get_model_size(self):
        """Get model size."""
        model_path = self.model.ckpt_path
        size_mb = os.path.getsize(model_path) / (1024 * 1024)

        print(f"  Model size: {size_mb:.2f} MB")

        return size_mb

# Usage
if __name__ == '__main__':
    evaluator = ComprehensiveEvaluator(
        model_path='runs/yolo11m_darts_v1/weights/best.pt',
        test_data_yaml='datasets/yolo_format/data.yaml'
    )

    results = evaluator.evaluate_all()
```

---

## Part 6: Continuous Improvement

### 6.1 Error Analysis

**Analyze Failure Cases**:
```python
def analyze_errors(errors):
    """Analyze common error patterns."""

    # Categorize errors
    error_categories = {
        'missed_darts': [],
        'false_positives': [],
        'calibration_issues': [],
        'scoring_errors': [],
    }

    for error in errors:
        if error['n_predicted'] < error['n_actual']:
            error_categories['missed_darts'].append(error)
        elif error['n_predicted'] > error['n_actual']:
            error_categories['false_positives'].append(error)
        # ... more categorization

    # Print analysis
    print("\nError Analysis:")
    for category, cases in error_categories.items():
        print(f"  {category}: {len(cases)} cases")

    return error_categories
```

---

### 6.2 Ablation Studies

**Test Impact of Changes**:
```python
def ablation_study():
    """Test impact of different configurations."""

    configurations = {
        'baseline': {'augment': False, 'mosaic': 0.0},
        'with_augmentation': {'augment': True, 'mosaic': 1.0},
        'larger_model': {'model': 'yolo11l.pt'},
        'higher_resolution': {'imgsz': 800},
    }

    results = {}

    for name, config in configurations.items():
        print(f"\nTesting configuration: {name}")
        # Train and evaluate
        # ... implementation
        results[name] = pcs_score

    # Compare
    print("\nAblation Study Results:")
    for name, pcs in results.items():
        print(f"  {name}: {pcs:.2f}% PCS")

    return results
```

---

### 6.3 Improvement Strategies

**Based on Error Analysis**:

1. **High False Positives**: Increase confidence threshold
2. **Low Recall**: Add more augmentation, collect more data
3. **Edge Scoring Errors**: Refine calibration algorithm
4. **Occlusion Issues**: Add occlusion-specific augmentation
5. **Lighting Issues**: Add color jittering, brightness variation

---

## Part 7: Reporting

### 7.1 Generate Report

**Evaluation Report** (`scripts/generate_report.py`):

```python
def generate_markdown_report(results, output_path='evaluation_report.md'):
    """Generate markdown evaluation report."""

    report = f"""
# YOLO11 Dart Detection Evaluation Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {results.get('model_path', 'N/A')}

## Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **PCS** | {results['PCS']:.2f}% | >95% | {'✅' if results['PCS'] > 95 else '❌'} |
| **mAP@0.5** | {results['mAP@0.5']:.4f} | >0.90 | {'✅' if results['mAP@0.5'] > 0.90 else '❌'} |
| **Precision** | {results['Precision']:.4f} | >0.90 | {'✅' if results['Precision'] > 0.90 else '❌'} |
| **Recall** | {results['Recall']:.4f} | >0.95 | {'✅' if results['Recall'] > 0.95 else '❌'} |
| **F1 Score** | {results['F1']:.4f} | >0.92 | {'✅' if results['F1'] > 0.92 else '❌'} |
| **FPS** | {results['FPS']:.2f} | >30 | {'✅' if results['FPS'] > 30 else '❌'} |

## Detailed Results

### Computer Vision Metrics
- **mAP@0.50:0.95**: {results['mAP@0.50:0.95']:.4f}
- **Model Size**: {results['Model_Size_MB']:.2f} MB
- **Inference Time**: {1000/results['FPS']:.2f} ms

### Error Analysis
[Error analysis details...]

## Recommendations
[Improvement recommendations...]

## Conclusion
[Overall assessment...]
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
```

---

## Summary

### Evaluation Checklist:

- [ ] Calculate PCS (primary metric)
- [ ] Calculate mAP@0.5 and mAP@0.50:0.95
- [ ] Measure Precision and Recall
- [ ] Calculate F1 Score
- [ ] Measure inference FPS
- [ ] Check model size
- [ ] Analyze error patterns
- [ ] Generate evaluation report

### Target Performance:

| Metric | Target | Excellent |
|--------|--------|-----------|
| **PCS** | >95% | >98% |
| **mAP@0.5** | >0.90 | >0.95 |
| **Precision** | >0.90 | >0.95 |
| **Recall** | >0.95 | >0.98 |
| **F1 Score** | >0.92 | >0.96 |
| **FPS (iPhone)** | >30 | >60 |
| **Model Size** | <30 MB | <20 MB |

### Next Steps:

1. Run comprehensive evaluation
2. Analyze failure cases
3. Implement improvements
4. Re-train if needed
5. Repeat until targets met

---

## Resources

**Scripts**:
- `scripts/calculate_pcs.py` - PCS calculator
- `scripts/comprehensive_evaluation.py` - Full evaluation
- `scripts/generate_report.py` - Report generator

**Tools**:
- Ultralytics validation
- Custom metrics calculators
- Error analysis tools
