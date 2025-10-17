#!/bin/bash
#
# Quick Start Script for YOLO11 Dart Detection
# Automatically checks dependencies and launches Streamlit app
#

set -e  # Exit on error

echo "🎯 YOLO11 Dart Detection - Quick Start"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo ""
    echo "📦 Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Verify model exists
MODEL_PATH="../models/best6.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Please ensure the model file exists."
    exit 1
fi
echo "✅ Model found: $MODEL_PATH"

# Check cameras
echo ""
echo "📷 Checking cameras..."
CAMERAS=$(python3 -c "
import cv2
available = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            available.append(i)
        cap.release()
print(f'{len(available)} camera(s) detected: {available}')
" 2>/dev/null)

echo "✅ $CAMERAS"

# Check for iPhone camera
if python3 -c "import cv2; cap = cv2.VideoCapture(1); result = cap.isOpened() and cap.read()[0]; cap.release(); exit(0 if result else 1)" 2>/dev/null; then
    echo "✅ iPhone Continuity Camera detected at index 1"
else
    echo "⚠️  iPhone Continuity Camera not detected"
    echo "   Place iPhone in landscape mode near MacBook to enable Continuity Camera"
fi

# Launch Streamlit
echo ""
echo "🚀 Launching Streamlit application..."
echo "   Browser will open automatically at http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the application"
echo ""
echo "========================================"
echo ""

# Run Streamlit
streamlit run dart_detector.py
