#!/usr/bin/env python3
"""
Test script for YOLO model loading and inference.
Tests that the model loads correctly and can perform inference.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_yolo_import():
    """Test that ultralytics YOLO can be imported."""
    print("=" * 60)
    print("TEST: YOLO Import")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        print("  ✓ Successfully imported YOLO from ultralytics")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import YOLO: {e}")
        print("  💡 Install with: pip install ultralytics")
        return False


def test_model_loading():
    """Test that the YOLO model file loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: Model Loading")
    print("=" * 60)

    model_path = '/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt'

    print(f"\n1. Checking model file exists...")
    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
        return False

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  ✓ Model file found ({file_size_mb:.2f} MB)")

    print(f"\n2. Loading model...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"    Model type: {type(model)}")

        # Try to get model info
        if hasattr(model, 'names'):
            print(f"    Model classes: {model.names}")

        return True
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False


def test_inference():
    """Test that the model can perform inference on a sample frame."""
    print("\n" + "=" * 60)
    print("TEST: Model Inference")
    print("=" * 60)

    model_path = '/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt'

    try:
        from ultralytics import YOLO
        import cv2

        print("\n1. Loading model...")
        model = YOLO(model_path)
        print("  ✓ Model loaded")

        print("\n2. Creating test frame...")
        # Create a dummy frame (640x480 RGB)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"  ✓ Test frame created: {test_frame.shape}")

        print("\n3. Running inference...")
        results = model.predict(test_frame, verbose=False)
        print(f"  ✓ Inference completed")
        print(f"    Results type: {type(results)}")
        print(f"    Number of results: {len(results)}")

        if len(results) > 0:
            result = results[0]
            print(f"    Result shape: {result.boxes.shape if hasattr(result, 'boxes') else 'N/A'}")

            # Check if boxes attribute exists
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                print(f"    Number of detections: {len(boxes)}")

                if hasattr(boxes, 'cls'):
                    print(f"    Detection classes: {boxes.cls}")

            return True
        else:
            print("  ⚠️  No results returned (this may be normal for random frame)")
            return True

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_on_real_image():
    """Test inference on an actual image if available."""
    print("\n" + "=" * 60)
    print("TEST: Real Image Inference")
    print("=" * 60)

    # Look for test images
    test_image_paths = [
        '/Users/fewzy/Dev/ai/deeper_darts/datasets/test/images/d1_03_31_2020',
        '/Users/fewzy/Dev/ai/deeper_darts/datasets/cropped_images'
    ]

    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(path, images[0])
                break

    if not test_image:
        print("  ⚠️  No test images found, skipping real image test")
        return None

    try:
        from ultralytics import YOLO
        import cv2

        print(f"\n1. Loading test image: {os.path.basename(test_image)}")
        img = cv2.imread(test_image)
        if img is None:
            print(f"  ✗ Failed to load image")
            return False

        print(f"  ✓ Image loaded: {img.shape}")

        print("\n2. Loading model and running inference...")
        model = YOLO('/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt')
        results = model.predict(img, verbose=False)

        print(f"  ✓ Inference completed")

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                num_detections = len(boxes)
                print(f"    Number of detections: {num_detections}")

                if num_detections > 0:
                    print("    Detection details:")
                    for i, box in enumerate(boxes[:5]):  # Show first 5
                        cls = int(box.cls) if hasattr(box, 'cls') else '?'
                        conf = float(box.conf) if hasattr(box, 'conf') else 0.0
                        print(f"      Detection {i+1}: class={cls}, confidence={conf:.2f}")

        return True

    except Exception as e:
        print(f"  ✗ Real image inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLO MODEL TEST SUITE")
    print("=" * 60)

    # Test 1: Import
    result1 = test_yolo_import()
    if not result1:
        print("\n⚠️  Cannot continue without YOLO import")
        sys.exit(1)

    # Test 2: Model loading
    result2 = test_model_loading()

    # Test 3: Inference on dummy frame
    result3 = test_inference()

    # Test 4: Inference on real image
    result4 = test_inference_on_real_image()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"YOLO import: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Model loading: {'✓ PASS' if result2 else '✗ FAIL'}")
    print(f"Dummy frame inference: {'✓ PASS' if result3 else '✗ FAIL'}")

    if result4 is None:
        print("Real image inference: ⚠️  SKIPPED (no test images)")
    else:
        print(f"Real image inference: {'✓ PASS' if result4 else '✗ FAIL'}")

    print("=" * 60)

    all_pass = result1 and result2 and result3 and (result4 is None or result4)
    sys.exit(0 if all_pass else 1)
