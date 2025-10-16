# 🚀 YOLO11 Dart Detection - Quick Start

**Goal**: Train a dart detection model in ~8 hours
**Result**: 95-99% accuracy, 30-60 FPS on iPhone

---

## ✅ Prerequisites

- [x] Dataset converted (16,050 images) - **DONE!**
- [ ] Google Account (free)
- [ ] 5 GB Google Drive space
- [ ] ~8 hours (mostly automated)

---

## 📋 Step-by-Step

### 1. Package Dataset (5 minutes)

```bash
# Run packaging script
./scripts/package_dataset_for_colab.sh

# Output: datasets/yolo_format.zip (~2-3 GB)
```

**Manual alternative**:
```bash
cd datasets
zip -r yolo_format.zip yolo_format/
```

---

### 2. Upload to Google Drive (30 minutes)

1. Go to https://drive.google.com/
2. Create folder structure:
   ```
   MyDrive/
   └── yolo11_darts/
       └── datasets/
   ```
3. Upload `yolo_format.zip` to `datasets/` folder
4. Wait for upload to complete

---

### 3. Open Google Colab (2 minutes)

1. Go to https://colab.research.google.com/
2. Upload notebook:
   - File → Upload notebook
   - Select: `notebooks/YOLO11_Dart_Detection_Training.ipynb`
3. Change runtime:
   - Runtime → Change runtime type → GPU (T4)

---

### 4. Run Training (6-8 hours)

**Option A: Run All Cells** (Recommended)
```
Runtime → Run all
```

**Option B: Step by Step**
- Cell 1: Check GPU ✅
- Cell 2: Install packages ✅
- Cell 3: Mount Google Drive ✅
- Cell 4: Extract dataset ✅
- Cell 5: Verify dataset ✅
- Cell 6: Initialize model ✅
- Cell 7: **START TRAINING** ⏰ (6-8 hours)
- Cell 8: Evaluate model ✅
- Cell 9: Calculate PCS ✅
- Cell 10: Export to CoreML ✅
- Cell 11: Package results ✅
- Cell 12: Visualize predictions ✅

**Keep the browser tab open!**

---

### 5. Download Results (5 minutes)

From Google Drive:
```
MyDrive/yolo11_darts/results/final_results/
├── best_model_int8.mlpackage  ← **USE THIS FOR IPHONE**
├── best_model.pt              (for further training)
├── training_curves.png        (loss graphs)
├── confusion_matrix.png       (performance)
└── README.txt                 (summary)
```

---

### 6. Integrate with iPhone

**Add to Xcode**:
1. Drag `best_model_int8.mlpackage` into Xcode project
2. Check "Copy items if needed"

**Basic Detection Code**:
```swift
import CoreML
import Vision

// Load model
let model = try! yolo11m_darts()
let visionModel = try! VNCoreMLModel(for: model.model)

// Create request
let request = VNCoreMLRequest(model: visionModel) { request, error in
    guard let results = request.results as? [VNRecognizedObjectObservation] else {
        return
    }

    // Process detections
    for detection in results {
        print("Class: \(detection.labels.first!.identifier)")
        print("Confidence: \(detection.confidence)")
    }
}

// Run on camera frame
let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
try? handler.perform([request])
```

**Full implementation**: See `docs/TRAINING_GUIDE.md` → iPhone Deployment

---

## 🎯 Expected Results

**Accuracy**:
- mAP@0.5: >0.90 ✅
- Precision: >0.90 ✅
- Recall: >0.95 ✅
- PCS: >95% ✅

**iPhone Performance**:
- iPhone 13: 35-40 FPS
- iPhone 14: 40-45 FPS
- iPhone 15: 45-50 FPS
- iPhone 15 Pro: 50-60 FPS

**Model Size**: 15-20 MB (INT8 quantized)

---

## 🆘 Troubleshooting

### GPU not available in Colab
```
Runtime → Change runtime type → GPU → Save
```

### Out of memory during training
```python
# In Cell 6, change batch size:
'batch': 8,  # Instead of 16
```

### Training interrupted
```python
# In Cell 7, resume from checkpoint:
model = YOLO('path/to/last.pt')
results = model.train(resume=True, **train_config)
```

### Slow on iPhone
```swift
// Reduce input size
request.preferredImageResolution = 416  // Instead of 640

// Or skip frames
if frameCount % 2 == 0 { detect() }
```

---

## 📚 Full Documentation

- **Training Guide**: `docs/TRAINING_GUIDE.md` (comprehensive)
- **Research**: `research/README.md` (10 documents, 192 KB)
- **Mobile Deployment**: `research/07_mobile_deployment.md`
- **Dataset Info**: `DATASET_CONVERSION_SUMMARY.md`

---

## ⏱️ Timeline

| Step | Duration | Type |
|------|----------|------|
| Package dataset | 5 min | Manual |
| Upload to Drive | 30 min | Automated |
| Colab setup | 2 min | Manual |
| **Training** | **6-8 hours** | **Automated** |
| Download results | 5 min | Manual |
| **Total** | **~8 hours** | Mostly hands-off |

---

## 🎉 Success Checklist

- [ ] Dataset packaged: `yolo_format.zip`
- [ ] Uploaded to Google Drive
- [ ] Colab notebook running
- [ ] GPU verified (Tesla T4)
- [ ] Training started (Cell 7)
- [ ] Training completed successfully
- [ ] Metrics: mAP >0.90, PCS >95%
- [ ] CoreML model exported
- [ ] Downloaded `best_model_int8.mlpackage`
- [ ] Added to Xcode project
- [ ] Tested on iPhone
- [ ] FPS: 30-60 ✅

---

## 💡 Pro Tips

1. **Keep Colab tab open** during training (prevents timeout)
2. **Check progress** every few hours (metrics should improve)
3. **Save checkpoints** to Google Drive (automatic in notebook)
4. **Test on real iPhone** (not simulator) for accurate FPS
5. **Use INT8 model** for best mobile performance

---

## 🔗 Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **Google Colab**: https://colab.research.google.com/
- **CoreML Guide**: https://developer.apple.com/documentation/coreml/

---

## 🚦 Quick Decision Tree

**Have you converted the dataset?**
- ✅ Yes → Go to Step 1 (Package Dataset)
- ❌ No → Run `python scripts/convert_to_yolo_format_v2.py` first

**Is your dataset packaged?**
- ✅ Yes → Go to Step 2 (Upload to Google Drive)
- ❌ No → Run Step 1

**Is training complete?**
- ✅ Yes → Go to Step 5 (Download Results)
- ⏳ In progress → Wait and monitor (check every 2 hours)
- ❌ Failed → Check troubleshooting section

**Have you downloaded the model?**
- ✅ Yes → Go to Step 6 (Integrate with iPhone)
- ❌ No → Download from Google Drive

---

**Ready to start?**

```bash
# Run this command:
./scripts/package_dataset_for_colab.sh

# Then follow the prompts!
```

---

**Questions?** Read the full guide: `docs/TRAINING_GUIDE.md`

**Good luck!** 🎯🎉

---

**Last Updated**: October 16, 2025
**Status**: Ready to Train ✅
