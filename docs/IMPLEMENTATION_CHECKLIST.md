# 🎯 Streamlit Demo Implementation Checklist

**Purpose**: Track implementation progress for YOLO11 dart detection Streamlit app
**Created**: October 17, 2025
**Status**: NOT STARTED

---

## 📦 Phase 1: Core Implementation (2-3 hours)

**Priority**: 🔴 CRITICAL | **Blockers**: None

### Setup
- [ ] Create `demo/` directory
- [ ] Create `demo/requirements.txt` with dependencies:
  - [ ] streamlit>=1.28.0
  - [ ] ultralytics>=8.0.0
  - [ ] opencv-python>=4.8.0
  - [ ] numpy>=1.24.0

### Camera Enumeration
- [ ] Implement `get_available_cameras()` function
- [ ] Test camera detection (0-9 range)
- [ ] Verify Camera 0 (MacBook) detected
- [ ] Verify Camera 1 (iPhone) detected

### Basic UI Structure
- [ ] Streamlit page config (title, icon, layout)
- [ ] Sidebar with configuration options
- [ ] Main layout (2 columns: video + scores)
- [ ] Camera selection dropdown
- [ ] Start/Stop buttons

### Model Loading
- [ ] Model path input field (default: models/best6.pt)
- [ ] YOLO model loading with error handling
- [ ] Verify model loads without errors

### Video Capture
- [ ] Open camera with `cv2.VideoCapture(selected_camera)`
- [ ] Set camera resolution (1280×720)
- [ ] Display video feed in Streamlit
- [ ] Handle camera open failures

### Session State
- [ ] Initialize `st.session_state.running`
- [ ] Start button sets running = True
- [ ] Stop button sets running = False
- [ ] Proper cleanup on stop

### Basic Testing
- [ ] App launches without errors
- [ ] Camera dropdown appears
- [ ] Video feed displays
- [ ] Start/Stop works
- [ ] No crashes on camera switch

**Phase 1 Completion Criteria**: App runs and displays live video feed

---

## 🎯 Phase 2: Scoring Integration (1-2 hours)

**Priority**: 🟡 HIGH | **Blockers**: Phase 1 complete

### Constants
- [ ] Define `BOARD_DICT` (20 segments)
- [ ] Define `BOARD_CONFIG` (radii from deepdarts_d1.yaml)
- [ ] Define `CLASS_NAMES` (5 classes: 4 calibration + dart_tip)

### Helper Functions
- [ ] Port `get_circle(xy)` from annotate.py
- [ ] Port `transform(xy, angle)` from annotate.py
- [ ] Implement `board_radii(r_d, cfg)` calculation
- [ ] Port `get_dart_scores(xy, numeric)` from annotate.py

### YOLO Integration
- [ ] Run inference: `model.predict(frame, conf, iou, imgsz)`
- [ ] Extract bounding boxes
- [ ] Get center points of boxes
- [ ] Separate calibration points (classes 0-3) and dart tips (class 4)
- [ ] Sort by class (calibration first)

### Score Calculation
- [ ] Count calibration points detected
- [ ] Count dart tips detected
- [ ] Create xy_array from detections
- [ ] Call `get_dart_scores()` for string labels
- [ ] Call `get_dart_scores(numeric=True)` for numbers
- [ ] Sum numeric scores for total

### UI Updates
- [ ] Calibration status indicator (✅ or ⚠️)
- [ ] Display "X/4" calibration count
- [ ] Display "Y darts detected"
- [ ] List individual dart scores (Dart 1: D20, etc.)
- [ ] Display total score in metric

### Visualization
- [ ] Annotate frame with `results[0].plot()`
- [ ] Show confidence if enabled
- [ ] Show labels if enabled
- [ ] Display annotated frame in Streamlit

### Testing
- [ ] Verify 4 calibration points detected → ✅
- [ ] Verify dart tips detected
- [ ] Test scoring for:
  - [ ] Regular numbers (1-20)
  - [ ] Doubles (D1-D20, 2× points)
  - [ ] Triples (T1-T20, 3× points)
  - [ ] Bull (B = 25 points)
  - [ ] Double Bull (DB = 50 points)
  - [ ] Miss (0 points)
- [ ] Verify total score sums correctly
- [ ] Test with < 4 calibration points (should not score)

**Phase 2 Completion Criteria**: Scores display accurately when conditions met

---

## 📚 Phase 3: Documentation (30 min)

**Priority**: 🟢 MEDIUM | **Blockers**: Phase 1 complete

### README.md
- [ ] Create `demo/README.md`
- [ ] Add title and description
- [ ] Prerequisites section (Python, dependencies, hardware)
- [ ] Installation instructions
- [ ] Usage instructions (step-by-step)
- [ ] Camera setup guide (Continuity Camera)
- [ ] Expected behavior section
- [ ] Troubleshooting common issues
- [ ] Performance expectations

### Inline Documentation
- [ ] Add docstrings to functions
- [ ] Add comments for complex logic
- [ ] Document constants with units
- [ ] Add type hints where appropriate

### User Instructions (in app)
- [ ] Sidebar expander with "How to Use"
- [ ] Camera setup instructions
- [ ] Detection requirements
- [ ] Scoring explanation
- [ ] Tips for best results

**Phase 3 Completion Criteria**: User can follow README and run app successfully

---

## 🧪 Phase 4: Testing & Validation (2-3 hours)

**Priority**: 🟢 MEDIUM | **Blockers**: Phases 1-2 complete

### Camera Tests
- [ ] Test Camera 0 (MacBook) selection and capture
- [ ] Test Camera 1 (iPhone) selection and capture
- [ ] Test camera switching without crashes
- [ ] Verify error handling for unavailable cameras
- [ ] Test with no cameras available

### Detection Tests
- [ ] Point camera at dartboard
- [ ] Verify 4 calibration points detected (green boxes)
- [ ] Verify dart tips detected (colored boxes)
- [ ] Test different lighting conditions (bright, dim)
- [ ] Test different camera angles
- [ ] Test various distances from dartboard

### Scoring Accuracy Tests
- [ ] Test with real dartboard and darts
- [ ] Verify scores match physical dart positions
- [ ] Test edge cases:
  - [ ] Dart on wire (boundary)
  - [ ] Dart very close to bullseye
  - [ ] Dart at segment boundary
  - [ ] Multiple darts in same segment
  - [ ] Dart outside board (miss)
- [ ] Compare with manual scoring (95%+ accuracy)

### Performance Tests
- [ ] Measure FPS on MacBook Pro
- [ ] Measure FPS on MacBook Air
- [ ] Target: FPS > 15 minimum
- [ ] Measure inference latency (< 100ms)
- [ ] Test with different image sizes (640, 800)
- [ ] Test with different confidence thresholds

### UI/UX Tests
- [ ] FPS counter updates correctly
- [ ] Calibration status updates in real-time
- [ ] Dart scores display without lag
- [ ] Total score updates correctly
- [ ] Start/Stop buttons are responsive
- [ ] Camera dropdown works smoothly
- [ ] Sliders update settings correctly

### Error Handling Tests
- [ ] Test with missing model file
- [ ] Test with corrupted model file
- [ ] Test with invalid camera index
- [ ] Test with camera disconnect during use
- [ ] Test with insufficient permissions
- [ ] Verify error messages are helpful

### Edge Cases
- [ ] Run for extended period (30+ min)
- [ ] Rapid start/stop cycles
- [ ] Camera switch during inference
- [ ] Extreme lighting (very bright/dark)
- [ ] Obstructed dartboard view
- [ ] Partial calibration points visible

**Phase 4 Completion Criteria**: All tests pass, no crashes, accurate scoring

---

## 🎨 Phase 5: Polish & Optimization (1-2 hours)

**Priority**: 🔵 LOW | **Blockers**: Phases 1-4 complete

### UI Improvements
- [ ] Add app icon/logo
- [ ] Improve color scheme
- [ ] Better button styling
- [ ] Add loading spinners
- [ ] Improve score display formatting
- [ ] Add dartboard diagram (optional)

### Performance Optimization
- [ ] Optimize camera resolution settings
- [ ] Implement frame skipping if FPS < 15
- [ ] Cache model loading
- [ ] Optimize scoring calculations
- [ ] Reduce Streamlit refresh overhead

### Code Quality
- [ ] Remove debug print statements
- [ ] Add proper logging (optional)
- [ ] Consistent code style
- [ ] Remove unused imports
- [ ] Add error recovery mechanisms

### Documentation Polish
- [ ] Add screenshots to README
- [ ] Create usage GIF/video (optional)
- [ ] Update troubleshooting with learnings
- [ ] Add FAQ section
- [ ] Document performance benchmarks

### User Experience
- [ ] Add keyboard shortcuts (optional)
- [ ] Save settings to config file (optional)
- [ ] Add session history (optional)
- [ ] Export scores to CSV (optional)

**Phase 5 Completion Criteria**: Professional, polished, production-ready app

---

## 📊 Overall Progress Tracker

### Completion Status
- [ ] Phase 1: Core Implementation (0%)
- [ ] Phase 2: Scoring Integration (0%)
- [ ] Phase 3: Documentation (0%)
- [ ] Phase 4: Testing & Validation (0%)
- [ ] Phase 5: Polish & Optimization (0%)

**Overall Completion**: 0/5 phases (0%)

### Time Tracking
- **Estimated Total**: 7-11 hours
- **Actual Time Spent**: ___ hours
- **Started**: ___________
- **Completed**: ___________

---

## ✅ Production Readiness Checklist

### Before First User
- [ ] All Phase 1-4 items complete
- [ ] README.md is comprehensive
- [ ] All critical tests pass
- [ ] FPS > 15 on target hardware
- [ ] Scoring accuracy > 95%
- [ ] Error messages are helpful
- [ ] Camera detection works reliably

### Before Public Release
- [ ] All Phase 1-5 items complete
- [ ] Extended testing (multiple sessions)
- [ ] Documentation includes screenshots
- [ ] Performance benchmarks documented
- [ ] Known issues documented
- [ ] Support/contact info added

---

## 🎯 Success Metrics

Application is production-ready when:

- ✅ Launches without errors
- ✅ Detects both MacBook and iPhone cameras
- ✅ Displays live video at 15+ FPS
- ✅ Detects 4 calibration points reliably
- ✅ Detects dart tips accurately
- ✅ Calculates scores correctly (95%+ accuracy)
- ✅ UI is responsive and intuitive
- ✅ Error handling is comprehensive
- ✅ Documentation is complete
- ✅ User can run app following README only

---

## 📝 Notes & Issues

Use this section to track issues, ideas, or deviations from plan:

### Issues Found
_None yet - implementation not started_

### Deviations from Plan
_None yet - implementation not started_

### Improvements Implemented
_None yet - implementation not started_

---

## 🔗 Quick Links

- **Plan**: `/Users/fewzy/Dev/ai/deeper_darts/research/13_continuity_camera_streamlit_plan.md`
- **Review**: `/Users/fewzy/Dev/ai/deeper_darts/docs/PRODUCTION_READINESS_REVIEW.md`
- **Summary**: `/Users/fewzy/Dev/ai/deeper_darts/docs/REVIEW_SUMMARY.md`
- **Reference Code**: `/Users/fewzy/Dev/ai/deeper_darts/datasets/annotate.py`
- **Config**: `/Users/fewzy/Dev/ai/deeper_darts/configs/deepdarts_d1.yaml`
- **Model**: `/Users/fewzy/Dev/ai/deeper_darts/models/best6.pt`

---

**Last Updated**: October 17, 2025
**Status**: Ready for implementation to begin
**Next Action**: Start Phase 1 - Core Implementation
