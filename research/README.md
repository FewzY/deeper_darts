# YOLO11 Dart Detection Research Package

**Version**: 1.0
**Date**: October 16, 2025
**Project**: Training YOLO11 for iPhone-based dart detection
**Target**: 95-99% accuracy with 30-60 FPS real-time performance

## Overview

This research package provides everything needed to train a YOLO11 model for automatic dart detection and scoring on iPhone, improving upon the original **DeepDarts** (94.7% accuracy) paper.

**Key Improvements**:
- ✅ YOLO11 architecture (22% fewer parameters, higher accuracy)
- ✅ CoreML optimization with INT8 quantization
- ✅ Apple Neural Engine acceleration
- ✅ Mobile-optimized deployment (30-60 FPS)
- ✅ Comprehensive augmentation strategies
- ✅ Production-ready iOS integration

## Research Documents

### 📄 Document Overview

| # | Document | Size | Description | Priority |
|---|----------|------|-------------|----------|
| **01** | [Paper Analysis](01_paper_analysis.md) | 8.6 KB | DeepDarts methodology & improvements | ⭐⭐⭐⭐⭐ |
| **02** | [YOLO11 Capabilities](02_yolo11_capabilities.md) | 14 KB | Model features & mobile optimization | ⭐⭐⭐⭐⭐ |
| **03** | [GitHub Findings](03_github_findings.md) | 13 KB | Existing projects & code references | ⭐⭐⭐⭐ |
| **04** | [Academic Papers](04_academic_papers.md) | 15 KB | Recent research & techniques | ⭐⭐⭐ |
| **05** | [Community Insights](05_community_insights.md) | 15 KB | Best practices & tutorials | ⭐⭐⭐⭐ |
| **06** | [Implementation Plan](06_implementation_plan.md) | 29 KB | Step-by-step roadmap (6 weeks) | ⭐⭐⭐⭐⭐ |
| **07** | [Mobile Deployment](07_mobile_deployment.md) | 27 KB | iPhone optimization & CoreML | ⭐⭐⭐⭐⭐ |
| **08** | [Dataset Preparation](08_dataset_preparation.md) | 26 KB | Format conversion & augmentation | ⭐⭐⭐⭐⭐ |
| **09** | [Colab Setup](09_colab_setup.md) | 18 KB | Google Colab training guide | ⭐⭐⭐⭐⭐ |
| **10** | [Evaluation Metrics](10_evaluation_metrics.md) | 26 KB | Performance measurement & PCS | ⭐⭐⭐⭐⭐ |

**Total**: ~192 KB of comprehensive research documentation

---

## Quick Start

### 1. Start Here (Priority Reading)

**For Beginners** (Read in order):
1. **01_paper_analysis.md** - Understand the original approach
2. **06_implementation_plan.md** - Get the complete workflow
3. **08_dataset_preparation.md** - Prepare your data
4. **09_colab_setup.md** - Start training

**For Experienced Users** (Jump to relevant sections):
- Training: **09_colab_setup.md**
- Deployment: **07_mobile_deployment.md**
- Optimization: **02_yolo11_capabilities.md**

---

## Document Summaries

### 01 - Paper Analysis
**DeepDarts: Modeling Keypoints as Objects**

**Key Findings**:
- Novel approach: Keypoints as objects (not heatmaps)
- YOLOv4-tiny based architecture
- 94.7% PCS on face-on dataset, 84.0% on multi-angle
- Homography-based transformation
- Task-specific augmentation (+6.8% improvement)

**Limitations Identified**:
- Occlusion handling needs improvement
- Limited dataset diversity
- Manual dartboard cropping required
- Older YOLO architecture (2020)

**Improvement Opportunities for YOLO11**:
- Better small object detection
- Native mobile optimization
- Advanced augmentation built-in
- End-to-end detection possible

---

### 02 - YOLO11 Capabilities
**Latest YOLO Architecture with Mobile Focus**

**Core Advantages**:
- 22% fewer parameters than YOLOv8
- Higher accuracy on COCO dataset
- Native CoreML support
- INT8 quantization built-in
- Apple Neural Engine optimized

**Model Variants**:
- YOLO11n: 2.6M params, 2.4ms inference
- YOLO11s: 9.4M params, 4.1ms inference
- YOLO11m: 20.1M params, 8.3ms inference

**Mobile Performance** (iPhone 13+):
- YOLO11n-INT8: 40-50 FPS at 416×416
- YOLO11s-INT8: 33-40 FPS at 416×416
- Memory: <100MB peak usage

**Expected Improvements**:
- 95-99% PCS (vs 94.7% baseline)
- 1.5-2x faster inference
- Better occlusion handling
- Production-ready deployment

---

### 03 - GitHub Findings
**Existing Projects & Code References**

**Key Projects Found**:

1. **Dart-Detection-and-Scoring-with-YOLO**
   - Custom YOLO implementation for darts
   - Scoring algorithm implementation
   - Real-time processing capability
   - IEEE DataPort dataset usage

2. **ultralytics/yolo-ios-app** ⭐⭐⭐⭐⭐
   - Official iOS implementation template
   - Real-time camera integration
   - CoreML optimization
   - Performance monitoring
   - **Direct code reuse opportunity**

3. **dart-detection Python Package**
   - Production-ready API
   - Command-line tools
   - High accuracy scoring
   - Reference implementation

**Recommended Path**:
- Use ultralytics/yolo-ios-app as iOS base
- Reference dart detection projects for scoring logic
- Train YOLO11m on Google Colab
- Deploy YOLO11n-INT8 on iPhone

---

### 04 - Academic Papers
**Recent Research (2024-2025)**

**Key Papers**:

1. **DeDoDe v2** (April 2024)
   - Addresses keypoint clustering
   - Training-time NMS strategy
   - 75.9 → 78.3 mAA improvement
   - Applicable to closely-spaced darts

2. **High-Resolution Keypoint Detection** (August 2025)
   - HRNet architecture
   - Multi-scale feature fusion
   - Real-time edge deployment
   - Sports analysis application

3. **Small Object Detection (ARSOD-YOLO)** (2024)
   - Specialized for small targets
   - Attention mechanisms
   - Multi-scale aggregation
   - Relevant for dart tip detection

**Key Techniques Identified**:
- Multi-scale detection for accuracy
- Training-time NMS for clustering
- Task-specific augmentation strategies
- INT8 quantization with calibration
- Knowledge distillation for deployment

**Expected Academic-Driven Improvements**:
- +8-13% PCS improvement potential
- Better handling of edge cases
- Improved mobile performance
- State-of-the-art accuracy

---

### 05 - Community Insights
**Practical Tips from Practitioners**

**YouTube & Tutorials**:
- Official Ultralytics YOLO11 training guides
- Roboflow comprehensive workflows
- Google Colab optimization tips
- Mobile deployment best practices

**Training Best Practices**:
1. Always start with pre-trained weights
2. Use aggressive augmentation for small datasets
3. Google Colab free tier is sufficient
4. Save checkpoints to Google Drive regularly
5. Monitor validation metrics closely
6. 100-200 images minimum for PoC

**Mobile Deployment Tips**:
1. INT8 quantization essential for real-time
2. 416×416 input optimal for mobile
3. Test on actual device, not simulator
4. Neural Engine >> GPU for efficiency
5. Frame skipping acceptable for 30 FPS
6. Profile with Xcode Instruments

**Performance Expectations** (Community Validated):
- iPhone 13: 40-60 FPS with YOLO11n-INT8
- iPhone 15 Pro: 60+ FPS with W8A8
- Model size: 15-20 MB after optimization
- Development time: 2-4 weeks

---

### 06 - Implementation Plan
**6-Week Step-by-Step Roadmap**

**Timeline**:
- **Week 1**: Dataset preparation & format conversion
- **Week 2**: Training setup & initial training
- **Week 3**: Training completion & evaluation
- **Week 4**: Model optimization & CoreML export
- **Week 5**: iOS integration & scoring logic
- **Week 6**: Testing, validation & deployment

**Key Deliverables**:
- ✅ YOLO format dataset (16k+ images)
- ✅ Trained YOLO11m model (95-99% PCS)
- ✅ Optimized YOLO11n-INT8 (15-20 MB)
- ✅ iOS app with real-time detection
- ✅ Dart scoring algorithm integrated
- ✅ Comprehensive evaluation report

**Code Templates Provided**:
- `scripts/convert_to_yolo_format.py`
- `notebooks/yolo11_dart_training.ipynb`
- iOS integration examples (Swift)
- Scoring algorithm implementation

**Expected Results**:
- Accuracy: 95-99% PCS (face-on), 90-95% (multi-angle)
- Speed: 30-60 FPS on iPhone 13+
- Model: 15-20 MB CoreML INT8
- Development: 6 weeks total

---

### 07 - Mobile Deployment
**iPhone Optimization & CoreML Integration**

**Deployment Pipeline**:
```
Camera → Preprocessing → YOLO11 CoreML → Detections →
Homography → Scoring → UI Update
```

**Optimization Techniques**:

1. **Model Export**:
   - CoreML with INT8 quantization
   - Neural Engine configuration
   - NMS included in model

2. **Quantization Options**:
   - FP16: 50% size reduction, 1.5-2x speedup
   - INT8: 75% size reduction, 2-3x speedup
   - W8A8: 75%+ reduction, 3-4x speedup (A17 Pro)

3. **Performance Optimization**:
   - Frame skipping (process every 2nd frame)
   - Async processing on background queue
   - Autoreleasepool for memory management
   - Adaptive frame rate based on battery

**Device-Specific Targets**:
| Device | Model | Input | FPS | Notes |
|--------|-------|-------|-----|-------|
| iPhone 12 | YOLO11n-INT8 | 320×320 | 40-45 | A14 |
| iPhone 13 | YOLO11n-INT8 | 416×416 | 45-50 | A15 |
| iPhone 15 Pro | YOLO11s-W8A8 | 416×416 | 60+ | A17 Pro |

**Complete Swift Implementation Provided**:
- DartDetector class
- CameraManager integration
- Real-time detection pipeline
- Performance monitoring
- Memory optimization

---

### 08 - Dataset Preparation
**Format Conversion & Augmentation**

**Current Dataset**:
- Format: labels.pkl (pandas DataFrame)
- Size: 16,050 images
- Sessions: 36+ different sessions
- Keypoints: 4 calibration + up to 3 darts

**Conversion Process**:
1. Load labels.pkl
2. Split by sessions (80/10/10)
3. Convert to YOLO format (txt annotations)
4. Crop to dartboard
5. Generate data.yaml

**YOLO Format**:
```
class_id x_center y_center width height
```
Classes: 0-3 (calibration), 4 (dart_tip)

**Augmentation Strategies**:

**Built-in YOLO11**:
- Mosaic (1.0 probability)
- MixUp (0.2)
- Copy-paste (0.3)
- Geometric transformations
- Color jittering

**Task-Specific (DeepDarts-inspired)**:
- Dartboard rotation (36° steps): +6.8% PCS
- Perspective warping: +7.1% PCS
- Small rotations (±2°): +4.6% PCS
- Dartboard flipping: +5.6% PCS

**Scripts Provided**:
- `convert_to_yolo_format.py` - Main conversion
- `verify_yolo_format.py` - Quality verification
- `deepdarts_augmentation.py` - Custom augmentation
- `check_dataset_quality.py` - Automated checks

**Expected Output**:
- Train: ~12,800 images
- Val: ~1,600 images
- Test: ~1,650 images
- With 3x augmentation: 48,150+ images total

---

### 09 - Colab Setup
**Google Colab Training Environment**

**Setup Steps**:
1. Check GPU availability (T4 expected)
2. Install ultralytics
3. Mount Google Drive
4. Extract dataset
5. Configure training
6. Train model (6-8 hours)
7. Export to CoreML

**Training Configuration**:
```python
config = {
    'model': 'yolo11m.pt',
    'data': 'data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # T4 GPU optimized
    'optimizer': 'AdamW',
    'mosaic': 1.0,
    'mixup': 0.2,
    'cache': 'ram',  # Fast loading
    'patience': 50,
    'save_period': 10,
}
```

**Best Practices**:
- Save checkpoints to Google Drive regularly
- Use session keep-alive for long training
- Clear memory between runs
- Monitor with TensorBoard
- Resume from checkpoint on timeout

**Complete Notebook Provided**:
- GPU verification
- Dataset extraction
- Training with monitoring
- Validation & testing
- CoreML export
- Results packaging

**Expected Timeline**:
- Setup: 10 minutes
- Training: 6-8 hours (100 epochs)
- Export: 5 minutes
- Total: ~8 hours

---

### 10 - Evaluation Metrics
**Performance Measurement & Analysis**

**Primary Metric: PCS** (Percent Correct Score)
```
PCS = (Correct Total Scores / Total Images) × 100%
```
- Easy to interpret
- Game-level accuracy
- Accounts for FP and FN
- Matches user expectations

**Standard CV Metrics**:
- **mAP@0.5**: Detection quality (target >0.90)
- **mAP@0.50:0.95**: Localization precision (target >0.70)
- **Precision**: False positive rate (target >0.90)
- **Recall**: False negative rate (target >0.95)
- **F1 Score**: Harmonic mean (target >0.92)

**Task-Specific Metrics**:
- Per-dart detection rate (target >95%)
- Calibration point detection (target >98%)
- Localization error (<5 pixels)

**Performance Metrics**:
- Inference FPS (target 30-60 on iPhone)
- Model size (target <30 MB)
- Memory usage (target <150 MB)
- Battery impact (low)

**Comprehensive Evaluation Provided**:
- PCS calculator implementation
- Standard metrics (built-in YOLO)
- Custom scoring algorithm
- Error analysis tools
- Report generation

**Target Performance Summary**:
| Metric | Target | Excellent |
|--------|--------|-----------|
| PCS | >95% | >98% |
| mAP@0.5 | >0.90 | >0.95 |
| Precision | >0.90 | >0.95 |
| Recall | >0.95 | >0.98 |
| FPS (iPhone) | >30 | >60 |
| Model Size | <30 MB | <20 MB |

---

## Expected Results

### Performance Targets

**Accuracy**:
- Face-on dataset: **95-99% PCS** (vs 94.7% baseline)
- Multi-angle dataset: **90-95% PCS** (vs 84.0% baseline)
- Per-dart detection: **95%+ recall**
- Calibration points: **98%+ detection rate**

**Speed (iPhone 13+)**:
- YOLO11n-INT8: **40-50 FPS**
- YOLO11s-INT8: **33-40 FPS**
- End-to-end latency: **<30ms**

**Model Size**:
- Training (YOLO11m): ~80 MB
- Deployment (YOLO11n-INT8): **15-20 MB**
- Memory usage: **<100 MB peak**

**Development Time**:
- Total: **4-6 weeks**
- Training: 6-8 hours (Google Colab)
- iOS integration: 1-2 weeks
- Testing & optimization: 1 week

---

## Project Resources

### Datasets
- **Current**: 16,050 images with labels.pkl
- **Location**: `../datasets/`
- **Format**: DeepDarts (needs conversion)
- **Target**: YOLO format (80/10/10 split)

### Models
- **Training**: YOLO11m (20.1M params)
- **Deployment**: YOLO11n (2.6M params)
- **Format**: PyTorch (.pt) → CoreML (.mlpackage)
- **Optimization**: INT8 quantization

### Tools
- **Training**: Google Colab (free tier)
- **Framework**: Ultralytics YOLO11
- **Mobile**: CoreML + Apple Neural Engine
- **Development**: Xcode, Swift

### Code
- **Python scripts**: Dataset conversion, training, evaluation
- **Swift code**: iOS integration, real-time detection
- **Notebooks**: Google Colab training workflow

---

## Usage Instructions

### For Training:

1. **Prepare Dataset**:
   ```bash
   python scripts/convert_to_yolo_format.py
   ```

2. **Upload to Google Drive**:
   - Create folder: `MyDrive/yolo11_darts/datasets/`
   - Upload: `yolo_format.zip`

3. **Open Colab Notebook**:
   - Use: `notebooks/yolo11_dart_training.ipynb`
   - Follow all cells in order
   - Monitor training progress

4. **Download Results**:
   - Best model: `best.pt`
   - CoreML model: `best_int8.mlpackage`
   - Training curves: `results.png`

### For Deployment:

1. **Setup Xcode Project**:
   - Clone: `ultralytics/yolo-ios-app`
   - Add CoreML model to project

2. **Integrate Dart Scoring**:
   - Use provided Swift code
   - Implement homography calculation
   - Add scoring algorithm

3. **Test on Device**:
   - Run on actual iPhone (not simulator)
   - Profile with Instruments
   - Measure FPS and memory

4. **Optimize if Needed**:
   - Adjust confidence threshold
   - Implement frame skipping
   - Optimize post-processing

### For Evaluation:

1. **Run Comprehensive Evaluation**:
   ```bash
   python scripts/comprehensive_evaluation.py
   ```

2. **Calculate PCS**:
   ```bash
   python scripts/calculate_pcs.py
   ```

3. **Generate Report**:
   ```bash
   python scripts/generate_report.py
   ```

4. **Analyze Errors**:
   - Review error categories
   - Identify improvement opportunities
   - Iterate on weak areas

---

## Key Insights

### What Makes This Approach Better?

1. **Modern Architecture**: YOLO11 (2024) vs YOLOv4-tiny (2020)
   - 22% fewer parameters
   - Higher accuracy
   - Better small object detection

2. **Mobile-First Design**: Native optimization for iPhone
   - CoreML integration
   - INT8 quantization
   - Neural Engine acceleration
   - Real-time performance

3. **Comprehensive Augmentation**: Combined strategies
   - YOLO11 built-in (mosaic, mixup)
   - Task-specific (DeepDarts-inspired)
   - Offline augmentation (3x dataset)

4. **Production-Ready**: End-to-end solution
   - Complete iOS integration code
   - Scoring algorithm implementation
   - Performance monitoring
   - Error handling

5. **Well-Documented**: Everything needed to succeed
   - 192 KB of documentation
   - Code templates and examples
   - Step-by-step instructions
   - Troubleshooting guides

### Critical Success Factors

1. ✅ **Quality Dataset**: 16k+ images, diverse scenarios
2. ✅ **Proper Augmentation**: Task-specific + built-in
3. ✅ **Transfer Learning**: Start from COCO pre-trained
4. ✅ **Mobile Optimization**: INT8 quantization essential
5. ✅ **Real Device Testing**: Simulator not sufficient
6. ✅ **Iterative Improvement**: Measure, analyze, optimize

---

## Next Steps

### Immediate Actions:

1. **Read Core Documents**:
   - [ ] 01_paper_analysis.md (understand baseline)
   - [ ] 06_implementation_plan.md (get roadmap)
   - [ ] 08_dataset_preparation.md (prepare data)

2. **Setup Environment**:
   - [ ] Install Python dependencies
   - [ ] Create Google Colab account
   - [ ] Setup Google Drive structure

3. **Prepare Dataset**:
   - [ ] Run conversion script
   - [ ] Verify YOLO format
   - [ ] Upload to Google Drive

4. **Start Training**:
   - [ ] Open Colab notebook
   - [ ] Configure parameters
   - [ ] Start training (6-8 hours)

### After Training:

1. **Evaluate Model**:
   - [ ] Calculate PCS metric
   - [ ] Measure standard metrics
   - [ ] Analyze errors

2. **Optimize for Mobile**:
   - [ ] Export to CoreML
   - [ ] Apply INT8 quantization
   - [ ] Test on iPhone

3. **Build iOS App**:
   - [ ] Setup Xcode project
   - [ ] Integrate CoreML model
   - [ ] Implement scoring
   - [ ] Test real-time performance

4. **Deploy & Iterate**:
   - [ ] TestFlight beta testing
   - [ ] Collect user feedback
   - [ ] Improve based on data
   - [ ] App Store submission

---

## Support & Resources

### Documentation
- **This Package**: 10 comprehensive markdown documents
- **Ultralytics**: https://docs.ultralytics.com/
- **Apple CoreML**: https://developer.apple.com/documentation/coreml/

### Code Repositories
- **DeepDarts Original**: https://github.com/wmcnally/deep-darts
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **iOS Template**: https://github.com/ultralytics/yolo-ios-app

### Community
- **Ultralytics Discussions**: GitHub discussions forum
- **Stack Overflow**: [yolo11] tag
- **Reddit**: r/computervision

### Citations

**DeepDarts Paper**:
```
McNally, W., Walters, P., Vats, K., Wong, A., & McPhee, J. (2021).
DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping
in Darts using a Single Camera. CVPRW 2021.
```

**YOLO11**:
```
Ultralytics YOLO11 (2024).
https://github.com/ultralytics/ultralytics
```

---

## Conclusion

This research package provides **everything needed** to:
- ✅ Understand the dart detection problem
- ✅ Train a state-of-the-art YOLO11 model
- ✅ Optimize for iPhone deployment
- ✅ Achieve 95-99% accuracy
- ✅ Run at 30-60 FPS in real-time
- ✅ Build a production-ready iOS app

**Expected Outcome**: A superior dart detection system that improves upon DeepDarts (94.7% → 95-99% accuracy) while enabling real-time mobile deployment.

**Time Investment**: 4-6 weeks from start to production app

**Cost**: Free (Google Colab free tier + open source tools)

---

**Start your journey here**: Begin with [01_paper_analysis.md](01_paper_analysis.md) to understand the baseline, then follow the [06_implementation_plan.md](06_implementation_plan.md) for the complete roadmap.

**Questions?** Review the relevant document or check the troubleshooting sections in each guide.

**Ready to train?** Jump straight to [09_colab_setup.md](09_colab_setup.md) to start training on Google Colab!

---

**Last Updated**: October 16, 2025
**Version**: 1.0
**Status**: Complete ✅
