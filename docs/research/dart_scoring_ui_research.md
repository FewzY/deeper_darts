# Dart Scoring UI Research Findings

**Research Date**: 2025-10-17
**Focus**: Real-time dart detection and scoring visualization UI/UX patterns
**Key**: dart_scoring_ui

---

## Executive Summary

This research covers UI/UX patterns, homography implementation, and mobile optimization for real-time dartboard detection and scoring systems. The findings integrate existing codebase patterns from the DeepDarts project with modern JavaScript libraries and React component patterns.

---

## 1. Visualization Components

### 1.1 Bounding Box Rendering on Canvas

**Existing Implementation** (from `/datasets/annotate.py`):
- Uses OpenCV's `cv2.circle()` and `cv2.putText()` for annotations
- Color-coded keypoints: green (calibration), cyan (darts)
- Font: `cv2.FONT_HERSHEY_SIMPLEX`, scale: 0.5, line_type: 1

**JavaScript/React Best Practices**:

**Library: `react-bounding-box`**
```javascript
import BoundingBox from 'react-bounding-box';

<BoundingBox
  image="path/to/image.jpg"
  boxes={[
    {
      coord: [xmin, ymin, xmax, ymax],
      label: 'dart',
      color: '#00ffff',
      confidence: 0.95
    }
  ]}
  options={{
    colors: {
      normal: 'rgba(0, 255, 255, 0.6)',
      selected: 'rgba(0, 255, 0, 1.0)',
      unselected: 'rgba(128, 128, 128, 0.3)'
    },
    style: {
      maxWidth: '100%',
      maxHeight: '100vh'
    }
  }}
/>
```

**Custom Canvas Implementation**:
```javascript
const drawBoundingBox = (ctx, bbox, label, confidence) => {
  const [x, y, width, height] = bbox;

  // Draw box
  ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, width, height);

  // Draw label background
  const text = `${label} ${(confidence * 100).toFixed(1)}%`;
  ctx.font = '14px Arial';
  const textWidth = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(0, 255, 255, 0.8)';
  ctx.fillRect(x, y - 20, textWidth + 10, 20);

  // Draw label text
  ctx.fillStyle = '#000';
  ctx.fillText(text, x + 5, y - 5);
};
```

### 1.2 Calibration Point Indicators

**From Existing Codebase** (`annotate.py`):
- 4 calibration points: `5_20`, `13_6`, `17_3`, `8_11` (BOARD_DICT indices)
- Points stored as normalized coordinates (0-1 range)
- Visual representation: small circles with numbered labels

**React Component**:
```javascript
const CalibrationPoints = ({ points, imageSize }) => {
  const pointNames = ['5_20', '13_6', '17_3', '8_11'];

  return (
    <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
      {points.map((point, idx) => {
        const [x, y] = point;
        const pixelX = x * imageSize.width;
        const pixelY = y * imageSize.height;

        return (
          <g key={idx}>
            <circle
              cx={pixelX}
              cy={pixelY}
              r={5}
              fill="rgba(0, 255, 0, 0.6)"
              stroke="#0f0"
              strokeWidth={2}
            />
            <text
              x={pixelX + 8}
              y={pixelY + 4}
              fill="#0f0"
              fontSize="12px"
              fontWeight="bold"
            >
              {pointNames[idx]}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
```

### 1.3 Dart Tip Tracking with Trails

**Trail Effect Implementation**:
```javascript
const DartTrail = ({ positions, maxTrailLength = 10 }) => {
  const [trail, setTrail] = useState([]);

  useEffect(() => {
    if (positions.length > 0) {
      const newTrail = [...trail, ...positions].slice(-maxTrailLength);
      setTrail(newTrail);
    }
  }, [positions]);

  const drawTrail = (ctx) => {
    ctx.beginPath();
    trail.forEach((pos, idx) => {
      const opacity = (idx + 1) / trail.length;
      const [x, y] = pos;

      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      ctx.strokeStyle = `rgba(255, 255, 0, ${opacity})`;
      ctx.lineWidth = 3 * opacity;
      ctx.stroke();
    });
  };

  return null; // Render to canvas ref
};
```

**Canvas-based with Fade Effect**:
```javascript
const renderTrailEffect = (ctx, currentPos, previousPositions) => {
  // Don't clear canvas completely - use fade effect
  ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'; // Semi-transparent black
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw trail with gradient opacity
  previousPositions.forEach((pos, index) => {
    const age = previousPositions.length - index;
    const opacity = Math.max(0, 1 - (age / 30)); // Fade over 30 frames
    const size = 5 * opacity;

    ctx.fillStyle = `rgba(255, 255, 0, ${opacity})`;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
    ctx.fill();
  });

  // Draw current position
  ctx.fillStyle = 'rgba(255, 255, 0, 1)';
  ctx.beginPath();
  ctx.arc(currentPos.x, currentPos.y, 8, 0, Math.PI * 2);
  ctx.fill();
};
```

### 1.4 Score Calculation Display

**From Existing Code** (`annotate.py` - `get_dart_scores()`):
```python
# Score logic:
# - Distance > r_d: '0' (miss)
# - Distance <= r_ib: 'DB' (double bull = 50)
# - Distance <= r_ob: 'B' (bull = 25)
# - Distance in double ring: 'D{number}' (2x)
# - Distance in treble ring: 'T{number}' (3x)
# - Otherwise: single number
```

**React Score Display Component**:
```javascript
const ScoreDisplay = ({ scores, currentTurn, totalScore }) => {
  return (
    <div className="score-display">
      {/* Current Turn Scores */}
      <div className="current-scores">
        <h3>Current Turn</h3>
        <div className="dart-scores">
          {scores.map((score, idx) => (
            <div key={idx} className={`dart-score dart-${idx + 1}`}>
              <span className="score-value">{score.display}</span>
              <span className="score-numeric">{score.numeric}</span>
            </div>
          ))}
        </div>
        <div className="turn-total">
          Turn Total: {scores.reduce((sum, s) => sum + s.numeric, 0)}
        </div>
      </div>

      {/* Running Total */}
      <div className="total-score">
        <h2>{totalScore}</h2>
        <span className="score-label">Total Score</span>
      </div>

      {/* Score History */}
      <div className="score-history">
        <h4>Recent Turns</h4>
        {/* Scrollable history */}
      </div>
    </div>
  );
};
```

### 1.5 Dartboard Overlay Graphics

**SVG Dartboard Overlay**:
```javascript
const DartboardOverlay = ({ calibrationPoints, dimensions }) => {
  // Calculate center and radii from calibration points
  const center = calculateCenter(calibrationPoints);
  const radii = calculateRadii(calibrationPoints);

  return (
    <svg
      viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {/* Double ring */}
      <circle
        cx={center.x}
        cy={center.y}
        r={radii.double}
        fill="none"
        stroke="rgba(255, 255, 255, 0.3)"
        strokeWidth="2"
      />

      {/* Treble ring */}
      <circle
        cx={center.x}
        cy={center.y}
        r={radii.treble}
        fill="none"
        stroke="rgba(255, 255, 255, 0.3)"
        strokeWidth="2"
      />

      {/* Bulls */}
      <circle
        cx={center.x}
        cy={center.y}
        r={radii.outerBull}
        fill="none"
        stroke="rgba(255, 255, 255, 0.4)"
        strokeWidth="1"
      />

      <circle
        cx={center.x}
        cy={center.y}
        r={radii.innerBull}
        fill="none"
        stroke="rgba(255, 255, 255, 0.5)"
        strokeWidth="1"
      />

      {/* Segment dividers (20 segments) */}
      {Array.from({ length: 20 }).map((_, i) => {
        const angle = (i * 18) - 90; // Start from top
        const rad = (angle * Math.PI) / 180;
        const x2 = center.x + radii.double * Math.cos(rad);
        const y2 = center.y + radii.double * Math.sin(rad);

        return (
          <line
            key={i}
            x1={center.x}
            y1={center.y}
            x2={x2}
            y2={y2}
            stroke="rgba(255, 255, 255, 0.2)"
            strokeWidth="1"
          />
        );
      })}
    </svg>
  );
};
```

---

## 2. Homography Implementation

### 2.1 Converting Detected Coordinates to Dartboard Space

**Theory** (from existing Python code in `annotate.py` - `transform()`):
- Use 4 calibration points as reference
- Create perspective transform matrix using `cv2.getPerspectiveTransform()`
- Apply transform using homogeneous coordinates
- Normalize output coordinates

**JavaScript Libraries**:

**Option 1: Homography.js**
```javascript
import Homography from 'homography';

// Define source points (detected calibration points in image space)
const srcPoints = [
  [x1, y1], // cal_1 (5_20)
  [x2, y2], // cal_2 (13_6)
  [x3, y3], // cal_3 (17_3)
  [x4, y4]  // cal_4 (8_11)
];

// Define destination points (ideal dartboard positions)
// Using board parameters: r_double = 0.170m
const center = [imgWidth / 2, imgHeight / 2];
const r = 100; // pixels for display

const dstPoints = [
  [center[0] - r * Math.sin(9 * Math.PI / 180), center[1] - r * Math.cos(9 * Math.PI / 180)],
  [center[0] + r * Math.sin(9 * Math.PI / 180), center[1] + r * Math.cos(9 * Math.PI / 180)],
  [center[0] - r * Math.cos(9 * Math.PI / 180), center[1] + r * Math.sin(9 * Math.PI / 180)],
  [center[0] + r * Math.cos(9 * Math.PI / 180), center[1] - r * Math.sin(9 * Math.PI / 180)]
];

// Create homography object
const homography = new Homography(srcPoints, dstPoints);

// Transform detected dart point
const dartImageCoords = [dartX, dartY];
const dartBoardCoords = homography.transform(dartImageCoords);
```

**Option 2: perspective-transform Library**
```javascript
import PerspT from 'perspective-transform';

const perspT = PerspT(
  // Source quad (detected calibration points)
  [x1, y1, x2, y2, x3, y3, x4, y4],
  // Destination quad (ideal dartboard space)
  [0, 0, 1, 0, 1, 1, 0, 1]
);

// Transform dart coordinates
const transformedCoords = perspT.transform(dartX, dartY);
```

**Option 3: OpenCV.js**
```javascript
// Load OpenCV.js
const cv = await import('opencv.js');

// Create Mat objects for points
const srcMat = cv.matFromArray(4, 1, cv.CV_32FC2, [
  x1, y1, x2, y2, x3, y3, x4, y4
]);

const dstMat = cv.matFromArray(4, 1, cv.CV_32FC2, [
  dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4
]);

// Get perspective transform matrix
const M = cv.getPerspectiveTransform(srcMat, dstMat);

// Transform points
const pointMat = cv.matFromArray(1, 1, cv.CV_32FC2, [dartX, dartY]);
const transformedMat = new cv.Mat();
cv.perspectiveTransform(pointMat, transformedMat, M);

// Extract transformed coordinates
const transformed = transformedMat.data32F;
const transformedX = transformed[0];
const transformedY = transformed[1];

// Clean up
srcMat.delete();
dstMat.delete();
M.delete();
pointMat.delete();
transformedMat.delete();
```

### 2.2 Using 4 Calibration Points

**Calibration Point Mapping** (from BOARD_DICT in `annotate.py`):
```javascript
const BOARD_DICT = {
  0: '13', 1: '4', 2: '18', 3: '1', 4: '20', 5: '5', 6: '12', 7: '9',
  8: '14', 9: '11', 10: '8', 11: '16', 12: '7', 13: '19', 14: '3',
  15: '17', 16: '2', 17: '15', 18: '10', 19: '6'
};

// Calibration points are at specific dartboard positions
// cal_1: Between 5 and 20 (top)
// cal_2: Between 13 and 6 (bottom)
// cal_3: Between 17 and 3 (left)
// cal_4: Between 8 and 11 (right)

const CalibrationConfig = {
  points: [
    { name: 'cal_1', position: '5_20', angle: 9 },    // degrees from vertical
    { name: 'cal_2', position: '13_6', angle: 189 },
    { name: 'cal_3', position: '17_3', angle: 99 },
    { name: 'cal_4', position: '8_11', angle: 279 }
  ],

  // Board parameters from config
  boardRadii: {
    r_board: 0.2255,      // meters
    r_double: 0.170,      // meters
    r_treble: 0.1074,     // meters
    r_outer_bull: 0.0159, // meters
    r_inner_bull: 0.00635,// meters
    w_double_treble: 0.01 // wire width meters
  }
};
```

**Calibration Point Detection & Validation**:
```javascript
const validateCalibrationPoints = (detectedPoints) => {
  // Check if all 4 points detected
  if (detectedPoints.length < 4) {
    console.warn('Missing calibration points:', 4 - detectedPoints.length);
    return estimateMissingPoints(detectedPoints);
  }

  // Check if points form a reasonable quadrilateral
  const center = calculateCentroid(detectedPoints);
  const distances = detectedPoints.map(pt =>
    Math.sqrt(Math.pow(pt.x - center.x, 2) + Math.pow(pt.y - center.y, 2))
  );

  const avgDistance = distances.reduce((a, b) => a + b) / distances.length;
  const maxDeviation = Math.max(...distances.map(d => Math.abs(d - avgDistance)));

  if (maxDeviation / avgDistance > 0.3) {
    console.warn('Calibration points may be inaccurate');
  }

  return detectedPoints;
};

const estimateMissingPoints = (detectedPoints) => {
  // Logic from est_cal_pts() in predictv8.py
  const missingIdx = findMissingIndex(detectedPoints);

  if (missingIdx.length === 1) {
    const idx = missingIdx[0];

    if (idx <= 1) {
      // Missing point 0 or 1
      const center = calculateCentroid([detectedPoints[2], detectedPoints[3]]);
      const reference = detectedPoints[idx === 0 ? 1 : 0];

      // Mirror across center
      detectedPoints[idx] = {
        x: 2 * center.x - reference.x,
        y: 2 * center.y - reference.y
      };
    } else {
      // Missing point 2 or 3
      const center = calculateCentroid([detectedPoints[0], detectedPoints[1]]);
      const reference = detectedPoints[idx === 2 ? 3 : 2];

      detectedPoints[idx] = {
        x: 2 * center.x - reference.x,
        y: 2 * center.y - reference.y
      };
    }
  }

  return detectedPoints;
};
```

### 2.3 Perspective Transformation in JavaScript

**Complete Homography Pipeline**:
```javascript
class DartboardHomography {
  constructor(calibrationPoints, boardConfig) {
    this.calibrationPoints = calibrationPoints;
    this.boardConfig = boardConfig;
    this.transformMatrix = null;
    this.center = null;
    this.radius = null;

    this.computeTransform();
  }

  computeTransform() {
    // Calculate center and radius from calibration points
    this.center = {
      x: this.calibrationPoints.reduce((sum, pt) => sum + pt.x, 0) / 4,
      y: this.calibrationPoints.reduce((sum, pt) => sum + pt.y, 0) / 4
    };

    this.radius = this.calibrationPoints.reduce((sum, pt) => {
      const dx = pt.x - this.center.x;
      const dy = pt.y - this.center.y;
      return sum + Math.sqrt(dx * dx + dy * dy);
    }, 0) / 4;

    // Create ideal destination points (9 degree rotation as in original)
    const angle = 9 * Math.PI / 180;
    const r = this.radius;

    const dstPoints = [
      [this.center.x - r * Math.sin(angle), this.center.y - r * Math.cos(angle)],
      [this.center.x + r * Math.sin(angle), this.center.y + r * Math.cos(angle)],
      [this.center.x - r * Math.cos(angle), this.center.y + r * Math.sin(angle)],
      [this.center.x + r * Math.cos(angle), this.center.y - r * Math.sin(angle)]
    ];

    // Compute homography matrix
    this.transformMatrix = this.computeHomographyMatrix(
      this.calibrationPoints.map(pt => [pt.x, pt.y]),
      dstPoints
    );
  }

  computeHomographyMatrix(src, dst) {
    // Solve for homography using DLT (Direct Linear Transform)
    // H is 3x3 matrix such that dst = H * src
    const A = [];

    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [xp, yp] = dst[i];

      A.push([
        -x, -y, -1, 0, 0, 0, x * xp, y * xp, xp
      ]);
      A.push([
        0, 0, 0, -x, -y, -1, x * yp, y * yp, yp
      ]);
    }

    // Solve using SVD (use math.js or similar)
    const { u, q, v } = svd(A);
    const h = v[v.length - 1]; // Last column of V

    return [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], h[8]]
    ];
  }

  transformPoint(x, y) {
    const M = this.transformMatrix;

    // Apply homography: [x', y', w'] = M * [x, y, 1]
    const xp = M[0][0] * x + M[0][1] * y + M[0][2];
    const yp = M[1][0] * x + M[1][1] * y + M[1][2];
    const w = M[2][0] * x + M[2][1] * y + M[2][2];

    // Normalize by w
    return {
      x: xp / w,
      y: yp / w
    };
  }

  transformPoints(points) {
    return points.map(pt => this.transformPoint(pt.x, pt.y));
  }
}
```

### 2.4 Score Region Mapping

**Score Calculation from Transformed Coordinates**:
```javascript
class DartScoreCalculator {
  constructor(boardConfig) {
    // From config: r_double, r_treble, r_inner_bull, r_outer_bull, w_double_treble
    this.config = boardConfig;

    // Board number arrangement (clockwise from top)
    this.BOARD_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5];
  }

  calculateScore(transformedPoint, center, r_double_px) {
    // Calculate relative position to center
    const dx = transformedPoint.x - center.x;
    const dy = transformedPoint.y - center.y;

    // Distance from center
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Angle (0 degrees = top, clockwise positive)
    let angle = Math.atan2(-dy, dx) * 180 / Math.PI;
    if (angle < 0) angle += 360;

    // Calculate radii in pixels
    const r_d = r_double_px;
    const r_t = r_d * (this.config.r_treble / this.config.r_double);
    const r_ob = r_d * (this.config.r_outer_bull / this.config.r_double);
    const r_ib = r_d * (this.config.r_inner_bull / this.config.r_double);
    const w_dt = this.config.w_double_treble * (r_d / this.config.r_double);

    // Determine score
    if (distance > r_d) {
      return { display: '0', numeric: 0, region: 'miss' };
    }

    if (distance <= r_ib) {
      return { display: 'DB', numeric: 50, region: 'double-bull' };
    }

    if (distance <= r_ob) {
      return { display: 'B', numeric: 25, region: 'bull' };
    }

    // Determine board number from angle (20 segments, 18 degrees each)
    const segmentIndex = Math.floor(angle / 18);
    const number = this.BOARD_NUMBERS[segmentIndex];

    // Determine ring
    if (distance <= r_t && distance > r_t - w_dt) {
      return {
        display: `T${number}`,
        numeric: number * 3,
        region: 'treble',
        number
      };
    }

    if (distance <= r_d && distance > r_d - w_dt) {
      return {
        display: `D${number}`,
        numeric: number * 2,
        region: 'double',
        number
      };
    }

    // Single region
    return {
      display: `${number}`,
      numeric: number,
      region: 'single',
      number
    };
  }

  calculateDartScores(dartPoints, calibrationPoints, imageSize) {
    // Create homography
    const homography = new DartboardHomography(calibrationPoints, this.config);

    // Transform dart points to corrected space
    const transformedDarts = homography.transformPoints(dartPoints);

    // Calculate scores
    return transformedDarts.map(pt =>
      this.calculateScore(pt, homography.center, homography.radius)
    );
  }
}
```

---

## 3. Real-Time UI Patterns

### 3.1 Frame-by-Frame Detection Rendering

**TensorFlow.js YOLO Detection Loop**:
```javascript
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

class RealtimeDartDetector {
  constructor(modelPath, canvasRef, videoRef) {
    this.model = null;
    this.canvasRef = canvasRef;
    this.videoRef = videoRef;
    this.isDetecting = false;
    this.frameCount = 0;
    this.fps = 0;
    this.lastFrameTime = Date.now();
  }

  async loadModel(modelPath) {
    console.log('Loading YOLO model...');
    this.model = await loadGraphModel(modelPath);
    console.log('Model loaded successfully');
  }

  async startDetection() {
    this.isDetecting = true;
    this.detectFrame();
  }

  stopDetection() {
    this.isDetecting = false;
  }

  async detectFrame() {
    if (!this.isDetecting) return;

    const startTime = performance.now();

    try {
      // Get video frame
      const video = this.videoRef.current;
      const canvas = this.canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Prepare input tensor
      const inputTensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(canvas);
        const resized = tf.image.resizeBilinear(img, [800, 800]); // YOLOv8 input size
        const normalized = resized.div(255.0);
        const batched = normalized.expandDims(0);
        return batched;
      });

      // Run inference
      const predictions = await this.model.predict(inputTensor);

      // Process predictions
      const boxes = await this.processDetections(predictions);

      // Draw detections
      this.drawDetections(ctx, boxes, canvas.width, canvas.height);

      // Clean up tensors
      inputTensor.dispose();
      predictions.dispose();

      // Calculate FPS
      this.updateFPS();

      // Draw FPS
      this.drawFPS(ctx);

    } catch (error) {
      console.error('Detection error:', error);
    }

    const endTime = performance.now();
    const inferenceTime = endTime - startTime;

    // Schedule next frame
    requestAnimationFrame(() => this.detectFrame());
  }

  async processDetections(predictions) {
    // YOLOv8 output format: [batch, num_predictions, 5 + num_classes]
    // [x_center, y_center, width, height, objectness, ...class_scores]

    const [boxes, scores, classes] = await Promise.all([
      predictions[0].array(),
      predictions[1].array(),
      predictions[2].array()
    ]);

    // Apply NMS (Non-Maximum Suppression)
    const detections = [];
    const confidenceThreshold = 0.5;
    const iouThreshold = 0.45;

    // Filter by confidence and apply NMS
    // ... (implementation details)

    return detections;
  }

  drawDetections(ctx, detections, width, height) {
    detections.forEach(det => {
      const { bbox, class_id, confidence } = det;
      const [x, y, w, h] = bbox;

      // Denormalize coordinates
      const x1 = x * width;
      const y1 = y * height;
      const boxWidth = w * width;
      const boxHeight = h * height;

      // Color by class (0: dart, 1-4: calibration points)
      const color = class_id === 0 ? '#00ffff' : '#00ff00';

      // Draw box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      // Draw label
      const label = this.getClassLabel(class_id);
      const text = `${label} ${(confidence * 100).toFixed(1)}%`;

      ctx.fillStyle = color;
      ctx.font = 'bold 16px Arial';
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

      ctx.fillStyle = '#000';
      ctx.fillText(text, x1 + 5, y1 - 7);

      // Draw center point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x1 + boxWidth / 2, y1 + boxHeight / 2, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  getClassLabel(classId) {
    const labels = ['dart', 'cal_1', 'cal_2', 'cal_3', 'cal_4'];
    return labels[classId] || 'unknown';
  }

  updateFPS() {
    this.frameCount++;
    const currentTime = Date.now();
    const elapsed = currentTime - this.lastFrameTime;

    if (elapsed >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / elapsed);
      this.frameCount = 0;
      this.lastFrameTime = currentTime;
    }
  }

  drawFPS(ctx) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 100, 30);

    ctx.fillStyle = '#0f0';
    ctx.font = 'bold 18px monospace';
    ctx.fillText(`FPS: ${this.fps}`, 20, 32);
  }
}
```

### 3.2 Confidence Threshold Controls

**React Component**:
```javascript
const DetectionControls = ({ onConfigChange, initialConfig }) => {
  const [config, setConfig] = useState({
    confidenceThreshold: initialConfig.confidenceThreshold || 0.5,
    iouThreshold: initialConfig.iouThreshold || 0.45,
    maxDetections: initialConfig.maxDetections || 3,
    showCalibrationPoints: initialConfig.showCalibrationPoints || true,
    showTrails: initialConfig.showTrails || false,
    trailLength: initialConfig.trailLength || 10
  });

  const handleChange = (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  return (
    <div className="detection-controls">
      <div className="control-group">
        <label>
          Confidence Threshold: {(config.confidenceThreshold * 100).toFixed(0)}%
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.confidenceThreshold}
          onChange={(e) => handleChange('confidenceThreshold', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>
          IoU Threshold: {(config.iouThreshold * 100).toFixed(0)}%
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={config.iouThreshold}
          onChange={(e) => handleChange('iouThreshold', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>
          Max Darts per Turn: {config.maxDetections}
        </label>
        <input
          type="range"
          min="1"
          max="5"
          step="1"
          value={config.maxDetections}
          onChange={(e) => handleChange('maxDetections', parseInt(e.target.value))}
        />
      </div>

      <div className="control-group checkbox">
        <label>
          <input
            type="checkbox"
            checked={config.showCalibrationPoints}
            onChange={(e) => handleChange('showCalibrationPoints', e.target.checked)}
          />
          Show Calibration Points
        </label>
      </div>

      <div className="control-group checkbox">
        <label>
          <input
            type="checkbox"
            checked={config.showTrails}
            onChange={(e) => handleChange('showTrails', e.target.checked)}
          />
          Show Dart Trails
        </label>
      </div>

      {config.showTrails && (
        <div className="control-group">
          <label>
            Trail Length: {config.trailLength} frames
          </label>
          <input
            type="range"
            min="5"
            max="30"
            step="1"
            value={config.trailLength}
            onChange={(e) => handleChange('trailLength', parseInt(e.target.value))}
          />
        </div>
      )}
    </div>
  );
};
```

### 3.3 FPS Counter Display

**Performance Monitor Component**:
```javascript
const PerformanceMonitor = ({ detector }) => {
  const [metrics, setMetrics] = useState({
    fps: 0,
    avgInferenceTime: 0,
    memoryUsage: 0,
    detectionCount: 0
  });

  useEffect(() => {
    const interval = setInterval(() => {
      if (detector) {
        setMetrics({
          fps: detector.fps,
          avgInferenceTime: detector.avgInferenceTime || 0,
          memoryUsage: performance.memory ?
            (performance.memory.usedJSHeapSize / 1048576).toFixed(2) : 0,
          detectionCount: detector.totalDetections || 0
        });
      }
    }, 100);

    return () => clearInterval(interval);
  }, [detector]);

  return (
    <div className="performance-monitor">
      <div className="metric">
        <span className="metric-label">FPS:</span>
        <span className={`metric-value ${metrics.fps < 15 ? 'warning' : metrics.fps < 25 ? 'ok' : 'good'}`}>
          {metrics.fps}
        </span>
      </div>

      <div className="metric">
        <span className="metric-label">Inference:</span>
        <span className="metric-value">
          {metrics.avgInferenceTime.toFixed(1)}ms
        </span>
      </div>

      <div className="metric">
        <span className="metric-label">Memory:</span>
        <span className="metric-value">
          {metrics.memoryUsage}MB
        </span>
      </div>

      <div className="metric">
        <span className="metric-label">Detections:</span>
        <span className="metric-value">
          {metrics.detectionCount}
        </span>
      </div>
    </div>
  );
};
```

### 3.4 Model Performance Metrics

**Metrics Collection System**:
```javascript
class PerformanceMetrics {
  constructor() {
    this.metrics = {
      inferenceTime: [],
      preprocessTime: [],
      postprocessTime: [],
      totalTime: [],
      fps: [],
      detectionConfidence: [],
      calibrationPointsDetected: []
    };

    this.maxSamples = 100;
  }

  recordInference(times) {
    this.addMetric('inferenceTime', times.inference);
    this.addMetric('preprocessTime', times.preprocess);
    this.addMetric('postprocessTime', times.postprocess);
    this.addMetric('totalTime', times.total);
  }

  recordDetections(detections) {
    const dartDetections = detections.filter(d => d.class_id === 0);
    const calPoints = detections.filter(d => d.class_id >= 1 && d.class_id <= 4);

    this.addMetric('detectionConfidence',
      dartDetections.map(d => d.confidence));
    this.addMetric('calibrationPointsDetected', calPoints.length);
  }

  addMetric(key, value) {
    if (Array.isArray(value)) {
      this.metrics[key].push(...value);
    } else {
      this.metrics[key].push(value);
    }

    // Keep only recent samples
    if (this.metrics[key].length > this.maxSamples) {
      this.metrics[key] = this.metrics[key].slice(-this.maxSamples);
    }
  }

  getStatistics() {
    const stats = {};

    for (const [key, values] of Object.entries(this.metrics)) {
      if (values.length === 0) continue;

      const sum = values.reduce((a, b) => a + b, 0);
      const avg = sum / values.length;
      const sorted = [...values].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      const min = Math.min(...values);
      const max = Math.max(...values);

      stats[key] = { avg, median, min, max, samples: values.length };
    }

    return stats;
  }

  exportCSV() {
    const stats = this.getStatistics();
    let csv = 'Metric,Average,Median,Min,Max,Samples\n';

    for (const [metric, values] of Object.entries(stats)) {
      csv += `${metric},${values.avg.toFixed(2)},${values.median.toFixed(2)},${values.min.toFixed(2)},${values.max.toFixed(2)},${values.samples}\n`;
    }

    return csv;
  }
}
```

### 3.5 Session Scoring History

**Score History Component**:
```javascript
const ScoringHistory = ({ history, onSelectTurn }) => {
  const [filter, setFilter] = useState('all'); // 'all', 'today', 'session'

  const filteredHistory = useMemo(() => {
    switch (filter) {
      case 'today':
        const today = new Date().setHours(0, 0, 0, 0);
        return history.filter(h => new Date(h.timestamp).getTime() >= today);
      case 'session':
        return history.filter(h => h.sessionId === currentSessionId);
      default:
        return history;
    }
  }, [history, filter]);

  return (
    <div className="scoring-history">
      <div className="history-header">
        <h3>Scoring History</h3>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All Time</option>
          <option value="today">Today</option>
          <option value="session">Current Session</option>
        </select>
      </div>

      <div className="history-list">
        {filteredHistory.map((turn, idx) => (
          <div
            key={idx}
            className="history-item"
            onClick={() => onSelectTurn(turn)}
          >
            <div className="turn-number">Turn {turn.turnNumber}</div>
            <div className="dart-scores">
              {turn.darts.map((dart, dartIdx) => (
                <span key={dartIdx} className={`dart-score ${dart.region}`}>
                  {dart.display}
                </span>
              ))}
            </div>
            <div className="turn-total">{turn.total}</div>
            <div className="timestamp">
              {new Date(turn.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>

      <div className="history-summary">
        <div className="stat">
          <span className="stat-label">Total Turns:</span>
          <span className="stat-value">{filteredHistory.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Average Score:</span>
          <span className="stat-value">
            {(filteredHistory.reduce((sum, t) => sum + t.total, 0) /
              filteredHistory.length).toFixed(1)}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Best Turn:</span>
          <span className="stat-value">
            {Math.max(...filteredHistory.map(t => t.total))}
          </span>
        </div>
      </div>
    </div>
  );
};
```

---

## 4. Mobile-Optimized UI

### 4.1 Touch-Friendly Controls

**Mobile Controls Component**:
```javascript
const MobileControls = ({ onCapture, onSwitchCamera, onToggleFlash, isDetecting }) => {
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);

  // Swipe detection for camera switch
  const handleTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const handleTouchMove = (e) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const handleTouchEnd = () => {
    if (!touchStart || !touchEnd) return;

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > 50;
    const isRightSwipe = distance < -50;

    if (isLeftSwipe || isRightSwipe) {
      onSwitchCamera();
    }
  };

  return (
    <div
      className="mobile-controls"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <button
        className="control-btn capture-btn"
        onClick={onCapture}
        disabled={isDetecting}
      >
        <CameraIcon size={48} />
      </button>

      <button
        className="control-btn switch-camera-btn"
        onClick={onSwitchCamera}
      >
        <FlipCameraIcon size={32} />
      </button>

      <button
        className="control-btn flash-btn"
        onClick={onToggleFlash}
      >
        <FlashIcon size={32} />
      </button>

      <div className="swipe-indicator">
        Swipe to switch camera
      </div>
    </div>
  );
};

// CSS for touch targets (minimum 44x44px)
const styles = `
.control-btn {
  min-width: 44px;
  min-height: 44px;
  padding: 12px;
  margin: 8px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.7);
  border: none;
  color: white;
  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
}

.control-btn:active {
  transform: scale(0.95);
  background: rgba(0, 0, 0, 0.9);
}

.capture-btn {
  width: 80px;
  height: 80px;
}
`;
```

### 4.2 Responsive Canvas Sizing

**Responsive Canvas Hook**:
```javascript
const useResponsiveCanvas = (videoRef, canvasRef) => {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const updateDimensions = () => {
      if (!videoRef.current) return;

      const video = videoRef.current;
      const container = canvasRef.current?.parentElement;

      if (!container) return;

      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      const videoAspect = video.videoWidth / video.videoHeight;
      const containerAspect = containerWidth / containerHeight;

      let width, height;

      if (videoAspect > containerAspect) {
        // Video is wider
        width = containerWidth;
        height = containerWidth / videoAspect;
      } else {
        // Video is taller
        height = containerHeight;
        width = containerHeight * videoAspect;
      }

      setDimensions({ width, height });
      setScale(width / video.videoWidth);

      // Update canvas
      if (canvasRef.current) {
        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
        canvasRef.current.style.width = `${width}px`;
        canvasRef.current.style.height = `${height}px`;
      }
    };

    updateDimensions();

    window.addEventListener('resize', updateDimensions);
    window.addEventListener('orientationchange', updateDimensions);

    return () => {
      window.removeEventListener('resize', updateDimensions);
      window.removeEventListener('orientationchange', updateDimensions);
    };
  }, [videoRef, canvasRef]);

  return { dimensions, scale };
};
```

### 4.3 Orientation Handling

**Orientation Manager**:
```javascript
const OrientationManager = ({ children, onOrientationChange }) => {
  const [orientation, setOrientation] = useState('portrait');
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    const handleOrientationChange = () => {
      const isPortrait = window.matchMedia('(orientation: portrait)').matches;
      const newOrientation = isPortrait ? 'portrait' : 'landscape';

      setOrientation(newOrientation);
      onOrientationChange?.(newOrientation);
    };

    handleOrientationChange();

    window.addEventListener('orientationchange', handleOrientationChange);
    window.matchMedia('(orientation: portrait)').addEventListener('change', handleOrientationChange);

    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
      window.matchMedia('(orientation: portrait)').removeEventListener('change', handleOrientationChange);
    };
  }, [onOrientationChange]);

  const lockOrientation = async (type) => {
    try {
      if (screen.orientation && screen.orientation.lock) {
        await screen.orientation.lock(type);
        setIsLocked(true);
      }
    } catch (error) {
      console.warn('Orientation lock not supported:', error);
    }
  };

  const unlockOrientation = () => {
    try {
      if (screen.orientation && screen.orientation.unlock) {
        screen.orientation.unlock();
        setIsLocked(false);
      }
    } catch (error) {
      console.warn('Orientation unlock failed:', error);
    }
  };

  return (
    <div className={`orientation-container orientation-${orientation}`}>
      {children}

      {orientation === 'portrait' && (
        <div className="orientation-hint">
          <RotateIcon size={48} />
          <p>Rotate device to landscape for better view</p>
        </div>
      )}

      <button
        className="lock-orientation-btn"
        onClick={() => isLocked ? unlockOrientation() : lockOrientation('landscape')}
      >
        {isLocked ? <LockIcon /> : <UnlockIcon />}
      </button>
    </div>
  );
};
```

### 4.4 Camera Switching

**Camera Manager Hook**:
```javascript
const useCameraManager = () => {
  const [devices, setDevices] = useState([]);
  const [activeDevice, setActiveDevice] = useState(null);
  const [stream, setStream] = useState(null);
  const [facingMode, setFacingMode] = useState('environment'); // 'user' or 'environment'

  useEffect(() => {
    enumerateDevices();
  }, []);

  const enumerateDevices = async () => {
    try {
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceList.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);

      if (videoDevices.length > 0 && !activeDevice) {
        setActiveDevice(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error('Failed to enumerate devices:', error);
    }
  };

  const startCamera = async (deviceId) => {
    try {
      // Stop existing stream
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          facingMode: deviceId ? undefined : facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30, max: 60 }
        }
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(newStream);
      setActiveDevice(deviceId);

      return newStream;
    } catch (error) {
      console.error('Failed to start camera:', error);
      throw error;
    }
  };

  const switchCamera = async () => {
    const currentIndex = devices.findIndex(d => d.deviceId === activeDevice);
    const nextIndex = (currentIndex + 1) % devices.length;
    const nextDevice = devices[nextIndex];

    if (nextDevice) {
      return startCamera(nextDevice.deviceId);
    }

    // Fallback: toggle facing mode
    const newFacingMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newFacingMode);
    return startCamera(null);
  };

  const toggleFlash = async () => {
    if (!stream) return;

    const videoTrack = stream.getVideoTracks()[0];
    const capabilities = videoTrack.getCapabilities();

    if (capabilities.torch) {
      const currentSettings = videoTrack.getSettings();
      const newTorchState = !currentSettings.torch;

      try {
        await videoTrack.applyConstraints({
          advanced: [{ torch: newTorchState }]
        });
      } catch (error) {
        console.warn('Flash toggle not supported:', error);
      }
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  return {
    devices,
    activeDevice,
    stream,
    facingMode,
    startCamera,
    switchCamera,
    toggleFlash,
    stopCamera
  };
};
```

### 4.5 Fullscreen Mode

**Fullscreen Component**:
```javascript
const FullscreenToggle = ({ targetRef }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    const element = targetRef.current;

    if (!isFullscreen) {
      try {
        if (element.requestFullscreen) {
          await element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
          await element.webkitRequestFullscreen();
        } else if (element.mozRequestFullScreen) {
          await element.mozRequestFullScreen();
        } else if (element.msRequestFullscreen) {
          await element.msRequestFullscreen();
        }
      } catch (error) {
        console.error('Fullscreen request failed:', error);
      }
    } else {
      try {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
          await document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
          await document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
          await document.msExitFullscreen();
        }
      } catch (error) {
        console.error('Exit fullscreen failed:', error);
      }
    }
  };

  return (
    <button
      className="fullscreen-toggle"
      onClick={toggleFullscreen}
      aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
    >
      {isFullscreen ? <ExitFullscreenIcon /> : <FullscreenIcon />}
    </button>
  );
};
```

---

## 5. Reference Implementations & Code Snippets

### 5.1 Complete React Component Example

```javascript
import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const DartScoringApp = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const [detector, setDetector] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [calibrationPoints, setCalibrationPoints] = useState([]);
  const [dartScores, setDartScores] = useState([]);
  const [config, setConfig] = useState({
    confidenceThreshold: 0.5,
    showOverlay: true,
    showTrails: false
  });

  const cameraManager = useCameraManager();
  const { dimensions, scale } = useResponsiveCanvas(videoRef, canvasRef);

  // Initialize TensorFlow and load model
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Set backend
        await tf.ready();
        await tf.setBackend('webgl');

        // Load YOLO model
        const detectorInstance = new RealtimeDartDetector(
          '/models/yolov8_dart_model/model.json',
          canvasRef,
          videoRef
        );

        await detectorInstance.loadModel();
        setDetector(detectorInstance);

        // Start camera
        const stream = await cameraManager.startCamera();
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Initialization failed:', error);
        setIsLoading(false);
      }
    };

    initializeApp();

    return () => {
      cameraManager.stopCamera();
      if (detector) {
        detector.stopDetection();
      }
    };
  }, []);

  // Start/stop detection
  const toggleDetection = () => {
    if (isDetecting) {
      detector?.stopDetection();
    } else {
      detector?.startDetection();
    }
    setIsDetecting(!isDetecting);
  };

  // Calculate scores when calibration points detected
  useEffect(() => {
    if (calibrationPoints.length === 4 && dartScores.length > 0) {
      const calculator = new DartScoreCalculator(boardConfig);
      const scores = calculator.calculateDartScores(
        dartScores,
        calibrationPoints,
        dimensions
      );

      console.log('Calculated scores:', scores);
    }
  }, [calibrationPoints, dartScores, dimensions]);

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Loading dart detection model...</p>
      </div>
    );
  }

  return (
    <OrientationManager onOrientationChange={(o) => console.log('Orientation:', o)}>
      <div ref={containerRef} className="dart-scoring-app">
        {/* Video and Canvas */}
        <div className="detection-viewport">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ display: 'none' }}
          />

          <canvas
            ref={canvasRef}
            className="detection-canvas"
          />

          {config.showOverlay && calibrationPoints.length === 4 && (
            <DartboardOverlay
              calibrationPoints={calibrationPoints}
              dimensions={dimensions}
            />
          )}

          <PerformanceMonitor detector={detector} />
        </div>

        {/* Controls */}
        <div className="controls-panel">
          <MobileControls
            onCapture={toggleDetection}
            onSwitchCamera={cameraManager.switchCamera}
            onToggleFlash={cameraManager.toggleFlash}
            isDetecting={isDetecting}
          />

          <DetectionControls
            onConfigChange={setConfig}
            initialConfig={config}
          />

          <FullscreenToggle targetRef={containerRef} />
        </div>

        {/* Scoring Display */}
        <div className="scoring-panel">
          <ScoreDisplay
            scores={dartScores}
            currentTurn={1}
            totalScore={0}
          />

          <ScoringHistory
            history={[]}
            onSelectTurn={(turn) => console.log('Selected turn:', turn)}
          />
        </div>
      </div>
    </OrientationManager>
  );
};

export default DartScoringApp;
```

### 5.2 Board Configuration (from existing codebase)

```javascript
// Board configuration from /configs/deepdarts_d1.yaml
const boardConfig = {
  r_board: 0.2255,      // radius of full board (m)
  r_double: 0.170,      // center bull to outside double wire edge (m) - BDO standard
  r_treble: 0.1074,     // center bull to outside treble wire edge (m) - BDO standard
  r_outer_bull: 0.0159, // outer bull radius (m)
  r_inner_bull: 0.00635,// inner bull radius (m)
  w_double_treble: 0.01 // wire apex to apex for double and treble (m)
};

// Dartboard number arrangement (clockwise from top)
const BOARD_NUMBERS = [
  20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
];

// Class IDs for YOLO model
const CLASS_LABELS = {
  0: 'dart',
  1: 'cal_1', // 5_20 position
  2: 'cal_2', // 13_6 position
  3: 'cal_3', // 17_3 position
  4: 'cal_4'  // 8_11 position
};
```

### 5.3 CSS Styles

```css
/* Mobile-first responsive styles */
.dart-scoring-app {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #000;
  display: flex;
  flex-direction: column;
}

.detection-viewport {
  position: relative;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.detection-canvas {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.performance-monitor {
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(0, 0, 0, 0.7);
  padding: 10px;
  border-radius: 8px;
  color: #fff;
  font-family: monospace;
  font-size: 12px;
  z-index: 10;
}

.metric {
  display: flex;
  justify-content: space-between;
  margin: 4px 0;
}

.metric-value.good { color: #0f0; }
.metric-value.ok { color: #ff0; }
.metric-value.warning { color: #f00; }

.controls-panel {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 20px;
  z-index: 20;
}

.scoring-panel {
  position: absolute;
  top: 60px;
  right: 10px;
  width: 280px;
  max-height: calc(100vh - 80px);
  background: rgba(0, 0, 0, 0.8);
  border-radius: 12px;
  padding: 15px;
  overflow-y: auto;
  z-index: 15;
}

/* Landscape orientation optimizations */
@media (orientation: landscape) {
  .dart-scoring-app {
    flex-direction: row;
  }

  .scoring-panel {
    position: relative;
    top: auto;
    right: auto;
    width: 300px;
    max-height: 100vh;
  }
}

/* Touch targets */
.control-btn {
  min-width: 44px;
  min-height: 44px;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
}

/* Loading screen */
.loading-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: #000;
  color: #fff;
}

.spinner {
  width: 60px;
  height: 60px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top-color: #0ff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

---

## Summary & Recommendations

### Key Findings:

1. **Homography Implementation**: Use `Homography.js` or `perspective-transform` library for lightweight transforms, or OpenCV.js for full feature parity with Python implementation.

2. **Real-Time Detection**: TensorFlow.js with YOLO models achieves 15-30 FPS on mobile devices with WebGL acceleration.

3. **Mobile Optimization**: Focus on touch-friendly controls (44x44px minimum), responsive canvas sizing, and orientation handling.

4. **Calibration**: Implement robust 4-point calibration with missing point estimation (mirroring across center).

5. **Score Calculation**: Convert polar coordinates (angle/distance) to dartboard sectors using BOARD_NUMBERS array and radial regions.

### Recommended Stack:

- **Framework**: React with hooks for state management
- **ML**: TensorFlow.js with pre-trained YOLOv8 model (converted from PyTorch)
- **Homography**: Homography.js for performance, OpenCV.js for compatibility
- **Canvas**: HTML5 Canvas API for rendering (no heavy libraries needed)
- **Camera**: MediaDevices API with device enumeration
- **State**: Context API or Zustand for global state

### Next Steps:

1. Convert trained YOLOv8 model to TensorFlow.js format
2. Implement calibration point detection and validation
3. Build homography transformation pipeline
4. Create mobile-responsive UI components
5. Add performance monitoring and optimization
6. Implement score history persistence (localStorage/IndexedDB)

---

**Research Completed**: 2025-10-17
**Files Referenced**:
- `/Users/fewzy/Dev/ai/deeper_darts/datasets/annotate.py`
- `/Users/fewzy/Dev/ai/deeper_darts/predictv8.py`
- `/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml`
- `/Users/fewzy/Dev/ai/deeper_darts/requirements.txt`
