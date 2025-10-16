# Google Colab Setup: YOLO11 Training Environment

## Executive Summary

Complete guide for setting up and training YOLO11 on Google Colab, including **GPU management**, **dataset handling**, **checkpoint saving**, and **best practices** for free tier optimization.

**Colab Tier Options**:
- **Free**: T4 GPU (16GB), 12-hour sessions
- **Pro** ($9.99/month): Better GPUs, 24-hour sessions
- **Pro+** ($49.99/month): A100 GPU, priority access

**Recommendation**: Free tier is sufficient for this project

---

## Part 1: Initial Setup

### 1.1 Create New Notebook

1. Go to https://colab.research.google.com/
2. Click "New Notebook"
3. Rename: "YOLO11_Dart_Detection_Training.ipynb"
4. Save to Google Drive

---

### 1.2 Check GPU Availability

```python
# Cell 1: Check GPU
import torch
import sys

print("=" * 60)
print("System Information")
print("=" * 60)

# Python version
print(f"Python version: {sys.version}")

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("⚠️ WARNING: GPU not available!")
    print("Go to Runtime → Change runtime type → Hardware accelerator → GPU")

print("=" * 60)
```

**Expected Output (Free Tier)**:
```
============================================================
System Information
============================================================
Python version: 3.10.12
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU name: Tesla T4
GPU memory: 15.90 GB
GPU count: 1
============================================================
```

---

### 1.3 Install Dependencies

```python
# Cell 2: Install Ultralytics
!pip install ultralytics -q

# Verify installation
from ultralytics import YOLO
print(f"Ultralytics version: {YOLO.__version__}")
```

**Additional Dependencies** (if needed):
```python
# Cell 3: Additional packages
!pip install -q \
    albumentations \
    pandas \
    pyyaml \
    matplotlib \
    seaborn \
    tqdm
```

---

### 1.4 Mount Google Drive

```python
# Cell 4: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
import os
drive_path = '/content/drive/MyDrive'
print(f"Drive mounted: {os.path.exists(drive_path)}")
```

---

## Part 2: Dataset Setup

### 2.1 Upload Dataset to Google Drive

**Local Computer** (before Colab):
```bash
# Create directory structure in Google Drive
# Go to Google Drive web interface:
# - Create folder: MyDrive/yolo11_darts/
# - Create folder: MyDrive/yolo11_darts/datasets/

# Upload dataset
# Option 1: Upload via web interface (drag and drop)
# Option 2: Use Google Drive desktop client
# Option 3: Use rclone or gdrive CLI

# Upload: datasets/yolo_format.zip
# Size: ~2-3 GB (16,050 images + labels)
```

---

### 2.2 Extract Dataset in Colab

```python
# Cell 5: Setup directories
import os
from pathlib import Path

# Define paths
DRIVE_BASE = '/content/drive/MyDrive/yolo11_darts'
WORK_DIR = '/content/dart_detection'
DATASET_ZIP = f'{DRIVE_BASE}/datasets/yolo_format.zip'

# Create directories
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Drive base: {DRIVE_BASE}")

# Verify dataset exists
if os.path.exists(DATASET_ZIP):
    print(f"✅ Dataset found: {DATASET_ZIP}")
else:
    print(f"❌ Dataset not found: {DATASET_ZIP}")
    print("Please upload datasets/yolo_format.zip to Google Drive")
```

```python
# Cell 6: Extract dataset
import zipfile
from tqdm import tqdm

print("Extracting dataset...")

# Extract to /content (faster than Drive)
with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
    # Get list of files
    file_list = zip_ref.namelist()
    print(f"Total files: {len(file_list)}")

    # Extract with progress bar
    for file in tqdm(file_list):
        zip_ref.extract(file, WORK_DIR)

print("✅ Extraction complete!")

# Verify
dataset_path = f'{WORK_DIR}/yolo_format'
print(f"\nDataset location: {dataset_path}")
print(f"Exists: {os.path.exists(dataset_path)}")

# List contents
!ls -lh {dataset_path}
```

---

### 2.3 Verify Dataset

```python
# Cell 7: Verify dataset structure
import yaml

data_yaml = f'{dataset_path}/data.yaml'

# Load and print config
with open(data_yaml, 'r') as f:
    config = yaml.safe_load(f)

print("Dataset Configuration:")
print(yaml.dump(config, default_flow_style=False))

# Count files
for split in ['train', 'val', 'test']:
    img_dir = Path(dataset_path) / 'images' / split
    label_dir = Path(dataset_path) / 'labels' / split

    n_images = len(list(img_dir.glob('*')))
    n_labels = len(list(label_dir.glob('*.txt')))

    print(f"\n{split.capitalize()}:")
    print(f"  Images: {n_images}")
    print(f"  Labels: {n_labels}")
    print(f"  Match: {n_images == n_labels}")
```

---

## Part 3: Training Configuration

### 3.1 Configure Training Parameters

```python
# Cell 8: Training configuration
import torch

# GPU batch size estimation
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU memory: {gpu_memory_gb:.2f} GB")

# Estimate optimal batch size
# Rule of thumb: batch_size = GPU_memory_GB * 2
if gpu_memory_gb >= 15:  # T4 or better
    batch_size = 16
elif gpu_memory_gb >= 10:
    batch_size = 8
else:
    batch_size = 4

print(f"Recommended batch size: {batch_size}")

# Training configuration
config = {
    # Model
    'model': 'yolo11m.pt',  # Pre-trained weights

    # Data
    'data': data_yaml,

    # Training
    'epochs': 100,
    'imgsz': 640,
    'batch': batch_size,

    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,          # Initial learning rate
    'lrf': 0.01,           # Final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # Augmentation
    'mosaic': 1.0,         # Mosaic augmentation
    'mixup': 0.2,          # MixUp augmentation
    'copy_paste': 0.3,     # Copy-paste augmentation
    'degrees': 10.0,       # Rotation
    'translate': 0.2,      # Translation
    'scale': 0.5,          # Scaling
    'shear': 2.0,          # Shearing
    'flipud': 0.5,         # Vertical flip
    'fliplr': 0.5,         # Horizontal flip
    'hsv_h': 0.015,        # Hue
    'hsv_s': 0.7,          # Saturation
    'hsv_v': 0.4,          # Value

    # Advanced
    'cache': 'ram',        # Cache images in RAM
    'workers': 8,          # Data loading workers
    'patience': 50,        # Early stopping patience
    'save_period': 10,     # Save checkpoint every N epochs

    # Output
    'project': f'{DRIVE_BASE}/runs',  # Save to Drive
    'name': 'yolo11m_darts_v1',

    # Device
    'device': 0,           # GPU 0

    # Misc
    'verbose': True,
    'seed': 42,            # Reproducibility
}

# Print configuration
print("\nTraining Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
```

---

### 3.2 Download Pre-trained Weights

```python
# Cell 9: Download pre-trained weights
from ultralytics import YOLO

# This will automatically download weights if not present
print("Loading pre-trained YOLO11m model...")
model = YOLO('yolo11m.pt')

print("✅ Model loaded successfully")
print(f"Model summary:")
model.info()
```

---

## Part 4: Training

### 4.1 Start Training

```python
# Cell 10: Train model
print("=" * 60)
print("Starting Training")
print("=" * 60)

try:
    # Train
    results = model.train(**config)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")

except Exception as e:
    print(f"\n❌ Training error: {e}")
    raise
```

**Training Progress**:
```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss    Instances    Size
  1/100      3.2G      1.234       2.456       1.789         123      640
  2/100      3.2G      1.123       2.234       1.567         123      640
  ...
```

---

### 4.2 Monitor Training (Optional)

```python
# Cell 11: Monitor training in parallel (run in separate cell)
import time
from IPython.display import clear_output

def monitor_training(results_dir, interval=30):
    """Monitor training progress."""
    while True:
        clear_output(wait=True)

        # Show latest results
        results_file = Path(results_dir) / 'results.png'
        if results_file.exists():
            from IPython.display import Image, display
            display(Image(filename=str(results_file)))

        print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)

# Usage (run in separate cell while training)
# monitor_training(f'{DRIVE_BASE}/runs/yolo11m_darts_v1')
```

---

### 4.3 Handle Session Timeouts

```python
# Cell 12: Save checkpoint regularly
# This cell should be run periodically (every hour)

import shutil

def save_checkpoint():
    """Save checkpoint to Drive."""
    results_dir = Path(f'{DRIVE_BASE}/runs/yolo11m_darts_v1')

    if not results_dir.exists():
        print("No training results found")
        return

    # Find latest checkpoint
    weights_dir = results_dir / 'weights'
    last_pt = weights_dir / 'last.pt'

    if last_pt.exists():
        # Copy to backup location
        backup_dir = Path(f'{DRIVE_BASE}/checkpoints')
        backup_dir.mkdir(exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'checkpoint_{timestamp}.pt'

        shutil.copy(last_pt, backup_path)
        print(f"✅ Checkpoint saved: {backup_path}")
    else:
        print("No checkpoint found")

# Auto-save every hour (optional)
# while training:
#     time.sleep(3600)  # 1 hour
#     save_checkpoint()
```

---

### 4.4 Resume Training

```python
# Cell 13: Resume training from checkpoint
# Run this if session times out

from ultralytics import YOLO

# Load checkpoint
checkpoint_path = f'{DRIVE_BASE}/runs/yolo11m_darts_v1/weights/last.pt'

if os.path.exists(checkpoint_path):
    print(f"Resuming from: {checkpoint_path}")
    model = YOLO(checkpoint_path)

    # Continue training
    results = model.train(resume=True)
else:
    print("No checkpoint found. Starting from scratch.")
```

---

## Part 5: Evaluation

### 5.1 Validate Model

```python
# Cell 14: Validate on validation set
from ultralytics import YOLO

# Load best model
best_model_path = f'{DRIVE_BASE}/runs/yolo11m_darts_v1/weights/best.pt'
model = YOLO(best_model_path)

print("Validating model on validation set...")
val_results = model.val(
    data=data_yaml,
    split='val',
    imgsz=640,
    batch=8,
    verbose=True,
)

# Print metrics
print("\n" + "=" * 60)
print("Validation Results")
print("=" * 60)
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")
print("=" * 60)
```

---

### 5.2 Test on Test Set

```python
# Cell 15: Test on test set
print("Testing model on test set...")
test_results = model.val(
    data=data_yaml,
    split='test',
    imgsz=640,
    batch=8,
    verbose=True,
)

# Print metrics
print("\n" + "=" * 60)
print("Test Results")
print("=" * 60)
print(f"mAP50: {test_results.box.map50:.4f}")
print(f"mAP50-95: {test_results.box.map:.4f}")
print(f"Precision: {test_results.box.mp:.4f}")
print(f"Recall: {test_results.box.mr:.4f}")
print("=" * 60)
```

---

### 5.3 Visualize Results

```python
# Cell 16: Visualize predictions
from IPython.display import Image, display
import random

# Get random test images
test_img_dir = Path(dataset_path) / 'images' / 'test'
test_images = list(test_img_dir.glob('*'))
samples = random.sample(test_images, min(5, len(test_images)))

print("Sample Predictions:")
print("=" * 60)

for img_path in samples:
    # Predict
    results = model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.25,
        iou=0.3,
        save=True,
        project=f'{WORK_DIR}/predictions',
    )

    # Display
    print(f"\nImage: {img_path.name}")
    result_img = f'{WORK_DIR}/predictions/{img_path.name}'

    if os.path.exists(result_img):
        display(Image(filename=result_img))
```

---

## Part 6: Export Model

### 6.1 Export to CoreML

```python
# Cell 17: Export to CoreML for iPhone
print("Exporting to CoreML (INT8)...")

# Export
coreml_path = model.export(
    format='coreml',
    int8=True,          # INT8 quantization
    nms=True,           # Include NMS
    imgsz=416,          # Mobile-optimized size
    keras=False,
    optimize=True,
    half=False,
    dynamic=False,
    simplify=True,
)

print(f"✅ CoreML model exported: {coreml_path}")

# Copy to Drive
import shutil
drive_model_dir = Path(f'{DRIVE_BASE}/models')
drive_model_dir.mkdir(exist_ok=True)

coreml_drive_path = drive_model_dir / 'yolo11_darts_int8.mlpackage'
shutil.copytree(coreml_path, coreml_drive_path, dirs_exist_ok=True)

print(f"✅ Copied to Drive: {coreml_drive_path}")
```

---

### 6.2 Export to ONNX (Optional)

```python
# Cell 18: Export to ONNX (for cross-platform)
print("Exporting to ONNX...")

onnx_path = model.export(
    format='onnx',
    imgsz=416,
    dynamic=False,
    simplify=True,
)

print(f"✅ ONNX model exported: {onnx_path}")
```

---

## Part 7: Download Results

### 7.1 Package Results

```python
# Cell 19: Package results for download
import shutil
import zipfile

# Create package
package_name = 'yolo11_dart_detection_results.zip'
package_path = f'{DRIVE_BASE}/{package_name}'

print("Packaging results...")

with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add best model
    best_model = f'{DRIVE_BASE}/runs/yolo11m_darts_v1/weights/best.pt'
    zipf.write(best_model, 'models/best.pt')

    # Add CoreML model
    for root, dirs, files in os.walk(coreml_drive_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, coreml_drive_path.parent)
            zipf.write(file_path, f'models/{arcname}')

    # Add training results
    results_png = f'{DRIVE_BASE}/runs/yolo11m_darts_v1/results.png'
    if os.path.exists(results_png):
        zipf.write(results_png, 'results/training_curves.png')

    # Add validation predictions
    val_batch = f'{DRIVE_BASE}/runs/yolo11m_darts_v1/val_batch0_pred.jpg'
    if os.path.exists(val_batch):
        zipf.write(val_batch, 'results/validation_predictions.jpg')

print(f"✅ Package created: {package_path}")
print(f"Download from Google Drive: MyDrive/yolo11_darts/{package_name}")
```

---

## Part 8: Best Practices

### 8.1 Memory Management

```python
# Cell 20: Clear memory
import gc

def clear_memory():
    """Clear GPU and RAM memory."""
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Memory cleared")

# Call when needed
clear_memory()

# Monitor memory
def print_memory_usage():
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

print_memory_usage()
```

---

### 8.2 Automatic Checkpointing

```python
# Cell 21: Auto-checkpoint callback
from ultralytics.utils.callbacks import default_callbacks

def checkpoint_callback(trainer):
    """Save checkpoint to Drive every 10 epochs."""
    epoch = trainer.epoch

    if epoch % 10 == 0:
        checkpoint_dir = Path(f'{DRIVE_BASE}/checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pt'
        shutil.copy(trainer.last, checkpoint_path)

        print(f"✅ Checkpoint saved: {checkpoint_path}")

# Add callback
default_callbacks['on_train_epoch_end'].append(checkpoint_callback)
```

---

### 8.3 Session Keep-Alive

```python
# Cell 22: Keep session alive (run in background)
from IPython.display import Javascript

# Prevent timeout by simulating user activity
display(Javascript('''
function ClickConnect(){
console.log("Keeping session alive...");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
'''))

print("✅ Session keep-alive enabled")
```

---

## Summary

### Complete Workflow:

1. ✅ Check GPU availability
2. ✅ Install dependencies
3. ✅ Mount Google Drive
4. ✅ Extract dataset
5. ✅ Configure training
6. ✅ Train model (100 epochs)
7. ✅ Validate and test
8. ✅ Export to CoreML
9. ✅ Download results

### Expected Timeline:
- Setup: 10 minutes
- Training: 6-8 hours (T4 GPU, 100 epochs)
- Export: 5 minutes
- Total: ~8 hours

### Resources Used:
- **Compute**: ~8 hours GPU time (free tier)
- **Storage**: ~5 GB (dataset + checkpoints)
- **Drive Space**: ~2 GB (results + models)

### Tips:
- ✅ Save checkpoints frequently
- ✅ Monitor training progress
- ✅ Clear memory between runs
- ✅ Use session keep-alive for long training
- ✅ Download results immediately after training

---

## Troubleshooting

**Issue: Session Timeout**
- Solution: Enable session keep-alive
- Solution: Save checkpoints every hour
- Solution: Resume training from checkpoint

**Issue: Out of Memory**
- Solution: Reduce batch size
- Solution: Clear GPU cache
- Solution: Use smaller input size

**Issue: Slow Training**
- Solution: Verify GPU is enabled
- Solution: Use `cache='ram'` for faster loading
- Solution: Increase `workers` for data loading

**Issue: Dataset Not Found**
- Solution: Verify upload to Google Drive
- Solution: Check file paths
- Solution: Re-upload if corrupted

---

## Next Steps

After successful training:
1. Download CoreML model
2. Integrate into iOS app (see 07_mobile_deployment.md)
3. Test on iPhone
4. Iterate based on results

**Notebook Template**: Available in `notebooks/yolo11_dart_training.ipynb`
