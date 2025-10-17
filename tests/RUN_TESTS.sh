#!/bin/bash
# Comprehensive test runner for YOLO11 Dart Detection App
# Run all tests and generate summary report

echo "=========================================="
echo "🎯 YOLO11 DART DETECTION - TEST RUNNER"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo "Running: $test_name..."
    echo "----------------------------------------"
    
    if python3 "$test_file"; then
        echo "✅ PASSED: $test_name"
        ((PASSED_TESTS++))
    else
        echo "❌ FAILED: $test_name"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
    echo ""
}

# Run all tests
echo "Starting test suite..."
echo ""

run_test "Camera Enumeration" "tests/test_camera_enumeration.py"
run_test "YOLO Model" "tests/test_yolo_model.py"
run_test "Scoring Logic" "tests/test_scoring_logic.py"
run_test "Full Integration" "tests/test_streamlit_app.py"

# Summary
echo "=========================================="
echo "📊 TEST SUMMARY"
echo "=========================================="
echo "Total Tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED_TESTS ✅"
echo "Failed:       $FAILED_TESTS ❌"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ Application is ready for production use."
    exit 0
else
    echo "⚠️  SOME TESTS FAILED"
    echo "❌ Please review errors above."
    exit 1
fi
