#!/bin/bash

################################################################################
# Package Dataset for Google Colab
################################################################################
#
# This script packages the YOLO format dataset for upload to Google Colab.
#
# Usage:
#   ./scripts/package_dataset_for_colab.sh
#
# Output:
#   datasets/yolo_format.zip (~2-3 GB)
#
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════"
echo "📦 PACKAGING DATASET FOR GOOGLE COLAB"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if dataset exists
if [ ! -d "datasets/yolo_format" ]; then
    echo -e "${RED}❌ ERROR: Dataset not found at datasets/yolo_format${NC}"
    echo ""
    echo "Please run the conversion script first:"
    echo "  python scripts/convert_to_yolo_format_v2.py"
    exit 1
fi

echo "🔍 Verifying dataset structure..."
echo ""

# Count files
TRAIN_IMAGES=$(find datasets/yolo_format/images/train -type f | wc -l | tr -d ' ')
VAL_IMAGES=$(find datasets/yolo_format/images/val -type f | wc -l | tr -d ' ')
TEST_IMAGES=$(find datasets/yolo_format/images/test -type f | wc -l | tr -d ' ')
TOTAL_IMAGES=$((TRAIN_IMAGES + VAL_IMAGES + TEST_IMAGES))

echo "📊 Dataset Statistics:"
echo "  Train images: ${TRAIN_IMAGES}"
echo "  Val images:   ${VAL_IMAGES}"
echo "  Test images:  ${TEST_IMAGES}"
echo "  Total:        ${TOTAL_IMAGES}"
echo ""

# Expected counts
EXPECTED_TOTAL=16050

if [ "$TOTAL_IMAGES" -ne "$EXPECTED_TOTAL" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Expected ${EXPECTED_TOTAL} images, found ${TOTAL_IMAGES}${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Dataset structure verified!"
echo ""

# Check if zip already exists
if [ -f "datasets/yolo_format.zip" ]; then
    echo -e "${YELLOW}⚠️  yolo_format.zip already exists${NC}"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm datasets/yolo_format.zip
        echo "  Removed old zip file"
    else
        echo "  Keeping existing zip file"
        exit 0
    fi
fi

echo "📦 Creating zip file..."
echo "   This may take 2-3 minutes..."
echo ""

# Create zip with progress
cd datasets
zip -r -q yolo_format.zip yolo_format/ &
ZIP_PID=$!

# Show spinner while zipping
spin='-\|/'
i=0
while kill -0 $ZIP_PID 2>/dev/null; do
    i=$(( (i+1) %4 ))
    printf "\r   Compressing... ${spin:$i:1}"
    sleep .1
done

printf "\r   Compressing... ✅\n"
cd ..

echo ""
echo "✅ Zip file created successfully!"
echo ""

# Show file size
ZIP_SIZE=$(ls -lh datasets/yolo_format.zip | awk '{print $5}')
echo "📁 File Information:"
echo "  Location: datasets/yolo_format.zip"
echo "  Size:     ${ZIP_SIZE}"
echo ""

# Calculate upload time estimate
ZIP_SIZE_MB=$(ls -l datasets/yolo_format.zip | awk '{print $5/1024/1024}')
UPLOAD_TIME_MIN=$(echo "scale=0; $ZIP_SIZE_MB / 10" | bc)  # Assuming 10 MB/s

echo "⏱️  Estimated upload time:"
echo "  10 Mbps:  ~${UPLOAD_TIME_MIN} minutes"
echo "  50 Mbps:  ~$((UPLOAD_TIME_MIN / 5)) minutes"
echo "  100 Mbps: ~$((UPLOAD_TIME_MIN / 10)) minutes"
echo ""

echo "════════════════════════════════════════════════════════════════════════"
echo "✅ PACKAGING COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "📤 Next Steps:"
echo ""
echo "1. Upload to Google Drive:"
echo "   - Go to https://drive.google.com/"
echo "   - Create folder: MyDrive/yolo11_darts/datasets/"
echo "   - Upload: datasets/yolo_format.zip"
echo ""
echo "2. Open Google Colab:"
echo "   - Upload: notebooks/YOLO11_Dart_Detection_Training.ipynb"
echo "   - Run all cells"
echo ""
echo "3. Wait for training (~6-8 hours)"
echo ""
echo "4. Download results:"
echo "   - best_model_int8.mlpackage (for iPhone)"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Optional: Open Google Drive in browser
read -p "Open Google Drive in browser? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "https://drive.google.com/drive/my-drive"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "https://drive.google.com/drive/my-drive"
    else
        echo "Please open: https://drive.google.com/drive/my-drive"
    fi
fi

echo ""
echo "🎉 Ready to train! Good luck!"
echo ""
