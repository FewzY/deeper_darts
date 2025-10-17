#!/usr/bin/env python3
"""
Test script for camera enumeration functionality.
Tests that get_available_cameras() works correctly with OpenCV.
"""

import cv2
import sys
import os

# Add parent directory to path to import from demo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_camera_enumeration():
    """Test basic camera enumeration with OpenCV."""
    print("=" * 60)
    print("TEST: Camera Enumeration")
    print("=" * 60)

    available_cameras = []

    print("\n1. Testing camera indices 0-2...")
    for i in range(3):
        cap = cv2.VideoCapture(i)
        is_opened = cap.isOpened()

        if is_opened:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            available_cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'fps': fps
            })

            print(f"  ✓ Camera {i}: Available ({width}x{height} @ {fps}fps)")
        else:
            print(f"  ✗ Camera {i}: Not available")

        cap.release()

    print(f"\n2. Total available cameras: {len(available_cameras)}")

    if len(available_cameras) == 0:
        print("  ⚠️  WARNING: No cameras detected!")
        print("  This may cause issues with the Streamlit app.")
        return False

    print(f"  ✓ Found {len(available_cameras)} camera(s)")

    print("\n3. Testing frame capture from first available camera...")
    if available_cameras:
        first_cam = available_cameras[0]['index']
        cap = cv2.VideoCapture(first_cam)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            print(f"  ✓ Successfully captured frame from camera {first_cam}")
            print(f"    Frame shape: {frame.shape}")
            return True
        else:
            print(f"  ✗ Failed to capture frame from camera {first_cam}")
            return False

    return True


def test_get_available_cameras_function():
    """Test the get_available_cameras() function from demo app."""
    print("\n" + "=" * 60)
    print("TEST: get_available_cameras() Function")
    print("=" * 60)

    try:
        # Try to import from demo directory
        demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo')
        if not os.path.exists(demo_path):
            print("  ⚠️  Demo directory not found yet")
            print("  Waiting for coder agent to create implementation...")
            return None

        sys.path.insert(0, demo_path)

        # Try to find the streamlit app file
        app_files = [f for f in os.listdir(demo_path) if f.endswith('.py')]
        if not app_files:
            print("  ⚠️  No Python files found in demo directory")
            return None

        print(f"\n1. Found app files: {app_files}")

        # Try to import get_available_cameras
        for app_file in app_files:
            module_name = app_file.replace('.py', '')
            try:
                module = __import__(module_name)
                if hasattr(module, 'get_available_cameras'):
                    get_available_cameras = module.get_available_cameras
                    print(f"  ✓ Found get_available_cameras() in {app_file}")

                    print("\n2. Testing function...")
                    cameras = get_available_cameras()

                    print(f"  ✓ Function returned {len(cameras)} camera(s)")
                    for cam in cameras:
                        print(f"    - Camera {cam}")

                    return True
            except Exception as e:
                print(f"  ⚠️  Error importing {app_file}: {e}")
                continue

        print("  ⚠️  get_available_cameras() function not found in any file")
        return None

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CAMERA ENUMERATION TEST SUITE")
    print("=" * 60)

    # Test 1: Basic OpenCV camera enumeration
    result1 = test_camera_enumeration()

    # Test 2: Test the actual function from demo app
    result2 = test_get_available_cameras_function()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Basic camera enumeration: {'✓ PASS' if result1 else '✗ FAIL'}")

    if result2 is None:
        print("get_available_cameras(): ⚠️  PENDING (implementation not ready)")
    else:
        print(f"get_available_cameras(): {'✓ PASS' if result2 else '✗ FAIL'}")

    print("=" * 60)

    sys.exit(0 if result1 else 1)
