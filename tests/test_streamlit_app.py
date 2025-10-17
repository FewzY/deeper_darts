#!/usr/bin/env python3
"""
Comprehensive test suite for the Streamlit dart detection application.
Tests all components, functions, and integration points.
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TEST: Module Imports")
    print("=" * 60)

    required_modules = [
        ('streamlit', 'st'),
        ('cv2', 'cv2'),
        ('numpy', 'np'),
        ('ultralytics', 'YOLO'),
        ('pathlib', 'Path'),
    ]

    all_success = True
    for module_name, import_name in required_modules:
        try:
            if import_name == 'YOLO':
                from ultralytics import YOLO
                print(f"  ✓ {module_name}.{import_name}")
            else:
                __import__(module_name)
                print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_success = False

    return all_success


def test_dart_detector_module():
    """Test that dart_detector module loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: dart_detector Module")
    print("=" * 60)

    try:
        import dart_detector

        print("\n1. Checking constants...")
        assert hasattr(dart_detector, 'BOARD_DICT'), "Missing BOARD_DICT"
        assert hasattr(dart_detector, 'BOARD_CONFIG'), "Missing BOARD_CONFIG"
        assert hasattr(dart_detector, 'CLASS_NAMES'), "Missing CLASS_NAMES"
        assert hasattr(dart_detector, 'DEFAULT_MODEL_PATH'), "Missing DEFAULT_MODEL_PATH"
        print("  ✓ All constants present")

        print("\n2. Validating BOARD_DICT...")
        assert len(dart_detector.BOARD_DICT) == 20, f"Expected 20 entries, got {len(dart_detector.BOARD_DICT)}"
        print(f"  ✓ BOARD_DICT has 20 entries")

        print("\n3. Validating BOARD_CONFIG...")
        required_keys = ['r_double', 'r_treble', 'r_outer_bull', 'r_inner_bull', 'w_double_treble']
        for key in required_keys:
            assert key in dart_detector.BOARD_CONFIG, f"Missing {key} in BOARD_CONFIG"
            print(f"  ✓ {key}: {dart_detector.BOARD_CONFIG[key]}")

        print("\n4. Checking functions...")
        required_functions = [
            'get_circle', 'board_radii', 'transform', 'get_dart_scores',
            'get_available_cameras', 'get_camera_label', 'setup_page',
            'setup_header', 'setup_sidebar', 'setup_instructions',
            'run_detection', 'main'
        ]
        for func in required_functions:
            assert hasattr(dart_detector, func), f"Missing function: {func}"
            print(f"  ✓ {func}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_functions():
    """Test scoring calculation functions with sample data."""
    print("\n" + "=" * 60)
    print("TEST: Scoring Functions")
    print("=" * 60)

    try:
        import dart_detector

        print("\n1. Testing get_circle()...")
        # Create sample calibration points (square around center)
        xy = np.array([
            [100, 100],  # top-left
            [200, 100],  # top-right
            [100, 200],  # bottom-left
            [200, 200],  # bottom-right
        ], dtype=np.float32)

        c, r = dart_detector.get_circle(xy)
        print(f"  ✓ Center: {c}")
        print(f"  ✓ Radius: {r:.2f}")

        # Validate results
        expected_center = np.array([150, 150])
        assert np.allclose(c, expected_center, atol=0.1), f"Expected center {expected_center}, got {c}"
        print("  ✓ Center calculation correct")

        print("\n2. Testing board_radii()...")
        r_d = 170  # Sample double radius in pixels
        r_t, r_ob, r_ib, w_dt = dart_detector.board_radii(r_d)
        print(f"  ✓ Treble radius: {r_t:.2f}")
        print(f"  ✓ Outer bull radius: {r_ob:.2f}")
        print(f"  ✓ Inner bull radius: {r_ib:.2f}")
        print(f"  ✓ Wire width: {w_dt:.2f}")

        # Validate proportions
        assert r_t < r_d, "Treble radius should be less than double radius"
        assert r_ib < r_ob, "Inner bull should be smaller than outer bull"
        print("  ✓ Radii proportions correct")

        print("\n3. Testing transform()...")
        xy_with_dart = np.array([
            [100, 100],
            [200, 100],
            [100, 200],
            [200, 200],
            [150, 150],  # dart at center
        ], dtype=np.float32)

        xy_transformed, M = dart_detector.transform(xy_with_dart, angle=0)
        print(f"  ✓ Transform successful")
        print(f"    Output shape: {xy_transformed.shape}")
        print(f"    Transform matrix shape: {M.shape}")

        assert xy_transformed.shape == xy_with_dart.shape, "Shape mismatch after transform"
        assert M.shape == (3, 3), f"Expected 3x3 transform matrix, got {M.shape}"
        print("  ✓ Transform dimensions correct")

        print("\n4. Testing get_dart_scores()...")
        # Test with valid calibration and dart
        scores = dart_detector.get_dart_scores(xy_with_dart, numeric=False)
        print(f"  ✓ Scores: {scores}")

        # Test numeric output
        numeric_scores = dart_detector.get_dart_scores(xy_with_dart, numeric=True)
        print(f"  ✓ Numeric scores: {numeric_scores}")

        # Test with insufficient calibration
        xy_incomplete = xy_with_dart[:3]  # Only 3 points
        empty_scores = dart_detector.get_dart_scores(xy_incomplete, numeric=False)
        assert len(empty_scores) == 0, "Should return empty for insufficient calibration"
        print("  ✓ Correctly handles insufficient calibration")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_functions():
    """Test camera enumeration functions."""
    print("\n" + "=" * 60)
    print("TEST: Camera Functions")
    print("=" * 60)

    try:
        import dart_detector
        import cv2

        print("\n1. Testing get_available_cameras()...")
        cameras = dart_detector.get_available_cameras(max_cameras=3)
        print(f"  ✓ Found {len(cameras)} camera(s): {cameras}")

        if len(cameras) == 0:
            print("  ⚠️  Warning: No cameras detected (this may cause issues)")
        else:
            print(f"  ✓ Cameras available for testing")

        print("\n2. Testing get_camera_label()...")
        for i in range(4):
            label = dart_detector.get_camera_label(i)
            print(f"  ✓ Camera {i}: {label}")

        print("\n3. Validating camera access...")
        if cameras:
            test_cam = cameras[0]
            cap = cv2.VideoCapture(test_cam)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                print(f"  ✓ Successfully captured frame from camera {test_cam}")
                print(f"    Frame shape: {frame.shape}")
            else:
                print(f"  ✗ Failed to capture frame from camera {test_cam}")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test YOLO model loading with actual model file."""
    print("\n" + "=" * 60)
    print("TEST: Model Loading")
    print("=" * 60)

    try:
        import dart_detector
        from ultralytics import YOLO
        from pathlib import Path

        print("\n1. Checking model path...")
        model_path = dart_detector.DEFAULT_MODEL_PATH
        print(f"  Model path: {model_path}")

        if not Path(model_path).exists():
            print(f"  ✗ Model file not found")
            return False

        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ Model file exists ({file_size_mb:.2f} MB)")

        print("\n2. Loading model...")
        model = YOLO(model_path)
        print(f"  ✓ Model loaded successfully")

        if hasattr(model, 'names'):
            print(f"  ✓ Model classes: {model.names}")

            # Validate class names match expectations
            expected_classes = {0, 1, 2, 3, 4}  # 5 classes
            actual_classes = set(model.names.keys())
            assert expected_classes == actual_classes, f"Class mismatch: expected {expected_classes}, got {actual_classes}"
            print("  ✓ Class configuration correct")

        print("\n3. Testing inference...")
        # Create dummy frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        results = model.predict(test_frame, verbose=False)

        print(f"  ✓ Inference successful")
        print(f"    Results: {len(results)} frame(s)")

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes'):
                print(f"    Detections: {len(result.boxes)}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements_file():
    """Verify requirements.txt has all necessary dependencies."""
    print("\n" + "=" * 60)
    print("TEST: Requirements File")
    print("=" * 60)

    requirements_path = os.path.join(
        os.path.dirname(__file__), '..', 'demo', 'requirements.txt'
    )

    print(f"\n1. Checking file: {requirements_path}")

    if not os.path.exists(requirements_path):
        print("  ✗ requirements.txt not found")
        return False

    print("  ✓ File exists")

    required_packages = [
        'streamlit',
        'ultralytics',
        'opencv-python',
        'numpy',
        'pillow'
    ]

    print("\n2. Validating required packages...")
    with open(requirements_path, 'r') as f:
        content = f.read().lower()

        all_present = True
        for pkg in required_packages:
            if pkg.lower() in content:
                print(f"  ✓ {pkg}")
            else:
                print(f"  ✗ Missing: {pkg}")
                all_present = False

    if not all_present:
        return False

    print("\n3. Checking installed versions...")
    import pkg_resources

    for pkg in required_packages:
        try:
            # Convert package name for import
            import_name = pkg.replace('-', '_') if pkg != 'opencv-python' else 'cv2'
            if import_name == 'pillow':
                import_name = 'PIL'

            if import_name == 'ultralytics':
                from ultralytics import __version__
                print(f"  ✓ {pkg}: {__version__}")
            elif import_name == 'streamlit':
                import streamlit
                print(f"  ✓ {pkg}: {streamlit.__version__}")
            elif import_name == 'cv2':
                import cv2
                print(f"  ✓ {pkg}: {cv2.__version__}")
            elif import_name == 'numpy':
                import numpy
                print(f"  ✓ {pkg}: {numpy.__version__}")
            elif import_name == 'PIL':
                import PIL
                print(f"  ✓ {pkg}: {PIL.__version__}")
        except Exception as e:
            print(f"  ⚠️  {pkg}: Could not check version ({e})")

    return True


def test_integration():
    """Test integration between components."""
    print("\n" + "=" * 60)
    print("TEST: Component Integration")
    print("=" * 60)

    try:
        import dart_detector
        from ultralytics import YOLO
        from pathlib import Path

        print("\n1. Testing camera + model integration...")

        # Get cameras
        cameras = dart_detector.get_available_cameras(max_cameras=3)
        if not cameras:
            print("  ⚠️  No cameras for integration test")
            return None

        # Load model
        model_path = dart_detector.DEFAULT_MODEL_PATH
        if not Path(model_path).exists():
            print("  ⚠️  Model not available for integration test")
            return None

        print("  ✓ Camera and model both available")

        print("\n2. Testing complete detection pipeline...")
        import cv2

        # Open camera
        cap = cv2.VideoCapture(cameras[0])
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("  ✗ Could not capture frame")
            return False

        print(f"  ✓ Captured frame: {frame.shape}")

        # Run model inference
        model = YOLO(model_path)
        results = model.predict(frame, verbose=False, conf=0.5)

        print(f"  ✓ Inference successful: {len(results[0].boxes)} detections")

        print("\n3. Testing scoring pipeline...")
        # Create sample detections (4 calibration + 1 dart)
        sample_xy = np.array([
            [400, 300],  # cal 1
            [600, 300],  # cal 2
            [400, 500],  # cal 3
            [600, 500],  # cal 4
            [500, 400],  # dart
        ], dtype=np.float32)

        scores = dart_detector.get_dart_scores(sample_xy, numeric=False)
        print(f"  ✓ Scoring successful: {scores}")

        return True

    except Exception as e:
        print(f"  ✗ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STREAMLIT DART DETECTION APP - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    results = {}

    # Run all tests
    results['imports'] = test_imports()
    results['dart_detector'] = test_dart_detector_module()
    results['scoring'] = test_scoring_functions()
    results['camera'] = test_camera_functions()
    results['model'] = test_model_loading()
    results['requirements'] = test_requirements_file()
    results['integration'] = test_integration()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠️  SKIPPED"

        print(f"{test_name.capitalize():20} {status}")

    print("=" * 60)

    # Final verdict
    failures = [name for name, result in results.items() if result is False]
    if failures:
        print(f"\n❌ TESTS FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
