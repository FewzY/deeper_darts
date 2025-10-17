"""
YOLO11 Dart Detection with iPhone Continuity Camera Support
Production-ready Streamlit application for real-time dart detection and scoring.

Features:
- Camera enumeration and selection (iPhone Continuity Camera support)
- Live YOLO11 inference with best6.pt model
- Automatic dart scoring with calibration
- Real-time score display
- Robust error handling
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import sys

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Dart board number mapping (20 segments, 18° each)
BOARD_DICT = {
    0: '13', 1: '4', 2: '18', 3: '1', 4: '20', 5: '5',
    6: '12', 7: '9', 8: '14', 9: '11', 10: '8', 11: '16',
    12: '7', 13: '19', 14: '3', 15: '17', 16: '2',
    17: '15', 18: '10', 19: '6'
}

# Board dimensions (BDO standard, in meters)
BOARD_CONFIG = {
    'r_double': 0.170,         # Center bull to outside double wire edge
    'r_treble': 0.1074,        # Center bull to outside treble wire edge
    'r_outer_bull': 0.0159,    # Outer bull radius
    'r_inner_bull': 0.00635,   # Inner bull (double bull) radius
    'w_double_treble': 0.01    # Wire width for double and treble rings
}

# Class names from model
CLASS_NAMES = {
    0: 'calibration_5_20',
    1: 'calibration_13_6',
    2: 'calibration_17_3',
    3: 'calibration_8_11',
    4: 'dart_tip'
}

# Default model path
DEFAULT_MODEL_PATH = "/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt"

# ============================================================================
# SCORING FUNCTIONS (Ported from annotate.py)
# ============================================================================

def get_circle(xy):
    """
    Calculate center and radius of dartboard from 4 calibration points.

    Args:
        xy: Array of shape (N, 2) with first 4 points being calibration points

    Returns:
        c: Center coordinates [x, y]
        r: Mean radius from center to calibration points
    """
    c = np.mean(xy[:4], axis=0)
    r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
    return c, r


def board_radii(r_d):
    """
    Calculate pixel radii for dartboard regions from double ring radius.

    Args:
        r_d: Double ring radius in pixels

    Returns:
        r_t: Treble ring radius
        r_ob: Outer bull radius
        r_ib: Inner bull radius
        w_dt: Width of double/treble rings
    """
    r_t = r_d * (BOARD_CONFIG['r_treble'] / BOARD_CONFIG['r_double'])
    r_ib = r_d * (BOARD_CONFIG['r_inner_bull'] / BOARD_CONFIG['r_double'])
    r_ob = r_d * (BOARD_CONFIG['r_outer_bull'] / BOARD_CONFIG['r_double'])
    w_dt = BOARD_CONFIG['w_double_treble'] * (r_d / BOARD_CONFIG['r_double'])
    return r_t, r_ob, r_ib, w_dt


def transform(xy, angle=0):
    """
    Apply perspective transform to normalize dartboard coordinates.

    Args:
        xy: Array of points with first 4 being calibration points
        angle: Rotation angle in degrees (default 0)

    Returns:
        xy_dst: Transformed points
        M: Perspective transformation matrix
    """
    c, r = get_circle(xy)

    # Source points (calibration points)
    src_pts = xy[:4].astype(np.float32)

    # Destination points (perfect square around center)
    dst_pts = np.array([
        [c[0] - r * np.sin(np.deg2rad(angle)), c[1] - r * np.cos(np.deg2rad(angle))],
        [c[0] + r * np.sin(np.deg2rad(angle)), c[1] + r * np.cos(np.deg2rad(angle))],
        [c[0] - r * np.cos(np.deg2rad(angle)), c[1] + r * np.sin(np.deg2rad(angle))],
        [c[0] + r * np.cos(np.deg2rad(angle)), c[1] - r * np.sin(np.deg2rad(angle))]
    ]).astype(np.float32)

    # Calculate perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply transform to all points
    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    return xy_dst, M


def get_dart_scores(xy, numeric=False):
    """
    Calculate dart scores from detection coordinates.

    Requires 4 calibration points (first 4 in xy) and at least 1 dart tip.
    Uses perspective transform to normalize coordinates, then calculates
    distance and angle from board center to determine score.

    Args:
        xy: Array of shape (N, 2) where first 4 are calibration points,
            rest are dart tips
        numeric: If True, return numeric scores; else string labels

    Returns:
        List of scores (e.g., ['D20', '13', 'T5', 'DB'] or [40, 13, 15, 50])
    """
    # Validate input
    if xy.shape[0] <= 4:
        return []

    # Check if we have 4 valid calibration points
    valid_cal = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
    if valid_cal.shape[0] < 4:
        return []

    try:
        # Transform to normalized dartboard coordinates
        xy_transformed, _ = transform(xy.copy(), angle=0)
        c, r_d = get_circle(xy_transformed)

        # Calculate radii for different board regions (in pixels)
        r_t, r_ob, r_ib, w_dt = board_radii(r_d)

        # Get dart positions relative to center
        dart_positions = xy_transformed[4:] - c

        # Calculate angles (0-360°) and distances from center
        angles = np.arctan2(-dart_positions[:, 1], dart_positions[:, 0]) / np.pi * 180
        angles = [a + 360 if a < 0 else a for a in angles]
        distances = np.linalg.norm(dart_positions, axis=-1)

        # Calculate scores
        scores = []
        for angle, dist in zip(angles, distances):
            if dist > r_d:
                # Outside double ring = miss
                scores.append(0 if numeric else 'Miss')
            elif dist <= r_ib:
                # Inner bull (double bull)
                scores.append(50 if numeric else 'DB')
            elif dist <= r_ob:
                # Outer bull (bull)
                scores.append(25 if numeric else 'B')
            else:
                # Determine dartboard segment (20 segments, 18° each)
                segment = int(angle / 18) % 20
                number = BOARD_DICT[segment]

                # Check if in double ring
                if r_d - w_dt <= dist <= r_d:
                    scores.append(int(number) * 2 if numeric else f'D{number}')
                # Check if in treble ring
                elif r_t - w_dt <= dist <= r_t:
                    scores.append(int(number) * 3 if numeric else f'T{number}')
                # Single scoring area
                else:
                    scores.append(int(number) if numeric else number)

        return scores

    except Exception as e:
        st.error(f"Error calculating scores: {str(e)}")
        return []


# ============================================================================
# CAMERA UTILITIES
# ============================================================================

def get_available_cameras(max_cameras=10):
    """
    Enumerate available camera devices.

    Tests camera indices 0-9 to find available devices.
    Continuity Camera typically appears at index 1 when active.

    Args:
        max_cameras: Maximum number of camera indices to test

    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def get_camera_label(index):
    """Get friendly label for camera index."""
    camera_labels = {
        0: "📱 MacBook Camera",
        1: "📱 iPhone (Continuity Camera)",
        2: "📹 External Camera 1",
        3: "📹 External Camera 2",
    }
    return camera_labels.get(index, f"📹 Camera {index}")


# ============================================================================
# STREAMLIT UI SETUP
# ============================================================================

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🎯 YOLO11 Dart Detection",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def setup_header():
    """Display page header."""
    st.title("🎯 YOLO11 Dart Detection - Continuity Camera")
    st.markdown("**Model Performance**: mAP@0.5: 99.0% | Precision: 99.29% | Recall: 98.14%")


def setup_sidebar():
    """
    Setup sidebar with configuration options.

    Returns:
        Dictionary with configuration values
    """
    st.sidebar.header("⚙️ Configuration")

    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value=DEFAULT_MODEL_PATH,
        help="Path to YOLO11 model weights file"
    )

    # Check if model exists
    if not Path(model_path).exists():
        st.sidebar.error(f"❌ Model not found: {model_path}")
        st.sidebar.info("Please check the model path and ensure the file exists.")

    # Camera selection
    st.sidebar.subheader("📷 Camera Selection")
    available_cameras = get_available_cameras()

    if not available_cameras:
        st.sidebar.error("❌ No cameras detected!")
        st.sidebar.info("Please ensure:\n- Camera is connected\n- iPhone is in landscape mode (for Continuity Camera)\n- Camera permissions are enabled")
        return None

    st.sidebar.success(f"✅ Found {len(available_cameras)} camera(s)")

    selected_camera = st.sidebar.selectbox(
        "Select Camera Source",
        options=available_cameras,
        index=1 if 1 in available_cameras else 0,  # Default to iPhone if available
        format_func=get_camera_label,
        help="iPhone Continuity Camera typically appears at index 1"
    )

    st.sidebar.info(f"Using camera index: **{selected_camera}**")

    # Detection parameters
    st.sidebar.subheader("🎛️ Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.50, 0.05,
        help="Minimum confidence for detections (higher = fewer false positives)"
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        0.0, 1.0, 0.45, 0.05,
        help="Intersection over Union threshold for NMS"
    )
    img_size = st.sidebar.selectbox(
        "Image Size",
        [640, 800],
        index=1,
        help="Input size for model (larger = more accurate but slower)"
    )

    # Display options
    st.sidebar.subheader("🖼️ Display Options")
    show_conf = st.sidebar.checkbox("Show Confidence", value=True)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_fps = st.sidebar.checkbox("Show FPS", value=True)

    return {
        'model_path': model_path,
        'selected_camera': selected_camera,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'img_size': img_size,
        'show_conf': show_conf,
        'show_labels': show_labels,
        'show_fps': show_fps
    }


def setup_instructions():
    """Display usage instructions in sidebar."""
    with st.sidebar.expander("📖 Instructions", expanded=False):
        st.markdown("""
        ### How to Use

        1. **Camera Setup**:
           - Place iPhone in landscape mode near MacBook
           - Wait for Continuity Camera to activate
           - Select "iPhone (Continuity Camera)" from dropdown

        2. **Detection**:
           - Click "▶️ Start Detection"
           - Point camera at dartboard
           - Ensure all 4 calibration points are visible

        3. **Scoring**:
           - Scores appear automatically when:
             - 4 calibration points detected (green boxes)
             - At least 1 dart tip detected (colored boxes)

        ### Tips
        - Good lighting improves accuracy
        - Keep dartboard centered in frame
        - Adjust confidence threshold if needed
        - Use 800px image size for best accuracy
        - Calibration points: 5-20, 13-6, 17-3, 8-11

        ### Troubleshooting
        - **No cameras detected**: Check iPhone is in landscape mode
        - **Scores not calculating**: Ensure 4/4 calibration points visible
        - **Low FPS**: Reduce image size to 640px
        - **Model not found**: Check model path in configuration
        """)


# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================

def run_detection(config):
    """
    Main detection loop with live inference and scoring.

    Args:
        config: Configuration dictionary from setup_sidebar()
    """
    # Load model
    try:
        with st.spinner("Loading YOLO11 model..."):
            model = YOLO(config['model_path'])
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

    # Open camera
    cap = cv2.VideoCapture(config['selected_camera'])

    if not cap.isOpened():
        st.error(f"❌ Failed to open camera {config['selected_camera']}")
        st.info("Please check camera connection and try again.")
        st.stop()

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📹 Live Detection")
        frame_placeholder = st.empty()
        fps_display = st.empty() if config['show_fps'] else None

    with col2:
        st.subheader("🎯 Dart Scores")
        calib_status = st.empty()
        scores_container = st.container()
        total_score_display = st.empty()

    # FPS tracking
    fps_counter = []

    # Main inference loop
    try:
        while st.session_state.running:
            start_time = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from camera")
                break

            # Run inference
            results = model.predict(
                source=frame,
                conf=config['conf_threshold'],
                iou=config['iou_threshold'],
                imgsz=config['img_size'],
                verbose=False
            )

            # Extract detections
            boxes = results[0].boxes
            detections_xy = []

            # Process each detection
            for box in boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Get center of bounding box (keypoint location)
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                detections_xy.append([center_x, center_y, cls])

            # Sort by class (calibration points first, then darts)
            detections_xy = sorted(detections_xy, key=lambda x: x[2])

            # Count detections
            num_calibration = sum(1 for d in detections_xy if d[2] < 4)
            num_darts = sum(1 for d in detections_xy if d[2] == 4)

            # Calculate scores if calibrated
            dart_scores = []
            total_score = 0

            if num_calibration >= 4 and num_darts > 0:
                try:
                    xy_array = np.array([[d[0], d[1]] for d in detections_xy])
                    dart_scores = get_dart_scores(xy_array, numeric=False)
                    numeric_scores = get_dart_scores(xy_array, numeric=True)
                    total_score = sum(numeric_scores)
                except Exception as e:
                    st.warning(f"⚠️ Score calculation error: {str(e)}")

            # Annotate frame
            annotated_frame = results[0].plot(
                conf=config['show_conf'],
                labels=config['show_labels'],
                line_width=2
            )

            # Display frame
            with frame_placeholder:
                # Convert BGR to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(annotated_frame_rgb, use_column_width=True)

            # Update FPS
            if fps_display:
                fps_counter.append(1 / (time.time() - start_time))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = sum(fps_counter) / len(fps_counter)
                with fps_display:
                    st.metric("FPS", f"{avg_fps:.1f}")

            # Update calibration status
            with calib_status:
                if num_calibration >= 4:
                    st.success(f"✅ Calibration: {num_calibration}/4")
                else:
                    st.warning(f"⚠️ Calibration: {num_calibration}/4")
                st.info(f"🎯 Darts detected: {num_darts}")

            # Display scores
            with scores_container:
                if dart_scores:
                    st.markdown("**Individual Scores:**")
                    for i, score in enumerate(dart_scores, 1):
                        # Color code based on score type
                        if 'DB' in str(score):
                            st.markdown(f"🎯 **Dart {i}:** `{score}` (Double Bull!)")
                        elif 'B' in str(score):
                            st.markdown(f"🎯 **Dart {i}:** `{score}` (Bull)")
                        elif 'T' in str(score):
                            st.markdown(f"🎯 **Dart {i}:** `{score}` (Triple)")
                        elif 'D' in str(score):
                            st.markdown(f"🎯 **Dart {i}:** `{score}` (Double)")
                        elif score == 'Miss':
                            st.markdown(f"❌ **Dart {i}:** `{score}`")
                        else:
                            st.markdown(f"**Dart {i}:** `{score}`")
                else:
                    if num_calibration < 4:
                        st.warning("⚠️ Need 4 calibration points")
                        st.info("Point camera at dartboard to detect calibration points (green boxes)")
                    elif num_darts == 0:
                        st.info("🎯 No darts detected yet")

            with total_score_display:
                if dart_scores:
                    st.metric("Total Score", total_score)

    finally:
        # Cleanup
        cap.release()
        st.success("✅ Detection stopped")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Page setup
    setup_page()
    setup_header()

    # Sidebar configuration
    config = setup_sidebar()
    setup_instructions()

    # Check if configuration is valid
    if config is None:
        st.error("❌ Cannot start detection without a valid camera")
        st.stop()

    if not Path(config['model_path']).exists():
        st.error(f"❌ Model not found: {config['model_path']}")
        st.info("Please check the model path in the sidebar configuration.")
        st.stop()

    # Control buttons
    col_start, col_stop = st.columns(2)
    start_button = col_start.button("▶️ Start Detection", use_container_width=True, type="primary")
    stop_button = col_stop.button("⏹️ Stop", use_container_width=True)

    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False

    if start_button:
        st.session_state.running = True

    if stop_button:
        st.session_state.running = False

    # Run detection loop
    if st.session_state.running:
        run_detection(config)
    else:
        st.info("👆 Click 'Start Detection' to begin")

        # Show example information when not running
        with st.expander("ℹ️ About This Application", expanded=False):
            st.markdown("""
            This application uses YOLO11 for real-time dart detection and automatic scoring.

            **Features:**
            - 🎯 Real-time dart detection with 99% accuracy
            - 📱 iPhone Continuity Camera support
            - 🎲 Automatic score calculation (D20, T19, Bull, etc.)
            - 📊 Live calibration status
            - ⚡ High-performance inference (20-30 FPS)

            **Model Details:**
            - Architecture: YOLO11
            - Input Size: 800x800 pixels
            - Classes: 5 (4 calibration points + dart tips)
            - Performance: mAP@0.5: 99.0%

            **Requirements:**
            - macOS with Continuity Camera support (macOS Ventura+)
            - iPhone with iOS 16+ (for Continuity Camera)
            - Good lighting conditions
            - Clear view of dartboard with all 4 calibration points
            """)


if __name__ == "__main__":
    main()
