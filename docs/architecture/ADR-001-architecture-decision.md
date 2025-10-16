# ADR-001: Web-Based YOLO11 Dart Detection Architecture

**Date**: 2025-10-17
**Status**: ACCEPTED
**Deciders**: System Architecture Designer
**Technical Story**: Browser-based dart detection with iPhone camera access

---

## Context and Problem Statement

We need to implement a web-based YOLO11 dart detection system that:
1. Works on iPhone Safari with camera access
2. Provides real-time or near-real-time detection
3. Calculates dart scores automatically
4. Allows rapid iteration and testing
5. Can be deployed for production use

Three architectural approaches were evaluated:
- **Option A**: Streamlit + ngrok (Python backend, simple UI)
- **Option B**: Full client-side (ONNX.js/TensorFlow.js)
- **Option C**: Hybrid (FastAPI backend + React frontend)

---

## Decision Drivers

### Functional Requirements
- Camera access on iPhone Safari (requires HTTPS)
- YOLO11 model inference (best.pt trained model)
- Bounding box visualization
- Score calculation with dartboard geometry
- Homography transformation for perspective correction

### Non-Functional Requirements
- **Performance**: Target 15-30 FPS on iPhone 13+
- **Latency**: < 250ms end-to-end for real-time feel
- **Model Size**: < 50 MB for web delivery
- **Development Speed**: MVP in 1 week, production in 5 weeks
- **Cost**: < $20/month for hosting
- **Scalability**: Support 10+ concurrent users initially

### Constraints
- Must work on iPhone Safari (WebKit limitations)
- HTTPS required for camera API access
- Limited budget for hosting infrastructure
- Single developer working on implementation
- Existing YOLO11 model trained in PyTorch

---

## Considered Options

### Option 1: Streamlit + ngrok

**Architecture**: Python backend with Streamlit UI, ngrok for HTTPS tunneling

**Pros**:
- ✅ Fastest development (2-3 hours to MVP)
- ✅ No model conversion (use best.pt directly)
- ✅ Full Python ecosystem (Ultralytics, OpenCV, NumPy)
- ✅ Simple debugging and iteration
- ✅ Minimal code (~200 lines)

**Cons**:
- ❌ Poor performance (1-3 FPS, 500-1000ms latency)
- ❌ No continuous video (frame-by-frame upload)
- ❌ ngrok free tier limitations (tunnel expires)
- ❌ Not production-ready
- ❌ High network overhead

**Use Case**: MVP testing and rapid prototyping

---

### Option 2: Full Client-Side (ONNX.js)

**Architecture**: React frontend with ONNX Runtime Web, static hosting

**Pros**:
- ✅ Best performance (20-30 FPS, 40-70ms latency)
- ✅ Zero server costs (static hosting)
- ✅ Offline capable (PWA with Service Workers)
- ✅ Scalable (no backend bottleneck)
- ✅ Best user experience

**Cons**:
- ❌ Complex model conversion (PyTorch → ONNX)
- ❌ iOS Safari WebGL limitations
- ❌ Browser memory constraints
- ❌ Difficult debugging (device-specific issues)
- ❌ Longer development time (2-3 weeks)

**Use Case**: Production PWA with offline support (future enhancement)

---

### Option 3: Hybrid (FastAPI + React)

**Architecture**: Python backend for inference, React frontend for UI, WebSocket for streaming

**Pros**:
- ✅ Good performance (15-30 FPS, 100-250ms latency)
- ✅ Full Python backend (no model conversion)
- ✅ Production-ready architecture
- ✅ Flexible deployment options
- ✅ Server-side analytics and logging
- ✅ Progressive enhancement possible

**Cons**:
- ❌ Backend hosting costs ($5-20/month)
- ❌ Network dependency (no offline mode)
- ❌ More complex than Streamlit (frontend + backend)
- ❌ Longer development (4-5 weeks)

**Use Case**: Production deployment with real-time requirements

---

## Decision Outcome

**Chosen Option**: **Phased Approach**

### Phase 1 (Week 1): Streamlit + ngrok - MVP
**Rationale**:
- Validate detection accuracy and user experience fastest
- Test model performance on real dart games
- Identify issues early with minimal investment
- Gather user feedback before committing to production architecture

**Acceptance Criteria**:
- Camera works on iPhone Safari via HTTPS
- Detection accuracy > 85% on test images
- Score calculation produces correct results
- UI is usable on mobile (responsive)
- Can capture and process 1-3 frames per second

### Phase 2 (Week 2-5): Hybrid (FastAPI + React) - Production
**Rationale**:
- Best balance of performance and development complexity
- Real-time experience (15-30 FPS) acceptable for dart games
- Backend flexibility for future features (analytics, multiplayer)
- Can add client-side ONNX in Phase 3 for offline mode
- Proven technology stack with good documentation

**Acceptance Criteria**:
- Real-time video streaming (15-30 FPS)
- End-to-end latency < 250ms
- Stable WebSocket connections
- Deployed with HTTPS on production domain
- Error handling and reconnection logic

### Phase 3 (Future): Client-Side Enhancement - PWA
**Rationale**:
- Add offline capability for venues without internet
- Reduce server costs for high-traffic scenarios
- Improve latency to 40-70ms
- Learn from Phase 1 & 2 experience before complex conversion

---

## Consequences

### Positive
1. **Fast Validation**: MVP in 1 week validates core assumptions
2. **Risk Mitigation**: Test iPhone Safari compatibility early
3. **Progressive Enhancement**: Each phase builds on previous learning
4. **Flexibility**: Can stop at any phase if requirements change
5. **Cost-Effective**: Only pay for hosting when needed (Phase 2+)

### Negative
1. **Throwaway Code**: Streamlit code not reused in production
2. **Time Investment**: Total 5-6 weeks vs 2-3 weeks direct implementation
3. **Context Switching**: Multiple technology stacks
4. **Delayed Production**: Real-time features not available until Week 5

### Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model accuracy insufficient | High | Medium | Test extensively in Phase 1 before committing to production |
| iOS Safari WebSocket issues | High | Low | Research Safari compatibility, add fallback to polling |
| Backend costs exceed budget | Medium | Low | Start with Railway ($5/mo), optimize before scaling |
| ONNX conversion fails | Low | Medium | Phase 3 is optional, keep hybrid as fallback |
| Development timeline overruns | Medium | Medium | Fixed time boxes per phase, cut features if needed |

---

## Technical Architecture

### Phase 1: Streamlit MVP
```
iPhone Safari → HTTPS (ngrok) → Streamlit Server → YOLO11 (best.pt)
                                     ↓
                              Score Calculation
                                     ↓
                              Annotated Image Display
```

**Stack**: Python, Streamlit, Ultralytics, ngrok
**Hosting**: Local development machine
**Cost**: $0

### Phase 2: Hybrid Production
```
iPhone Safari → WSS → FastAPI Backend → YOLO11 Inference
     ↑                                         ↓
     |                                   Score Calculation
     |                                         ↓
     └──────────── JSON Response ─────────────┘

React Frontend (Vercel) ← WebSocket → FastAPI Backend (Railway)
```

**Stack**:
- Frontend: React, TypeScript, Socket.IO, Vite
- Backend: Python, FastAPI, Ultralytics, OpenCV
- Deployment: Vercel (frontend) + Railway (backend)

**Cost**: $5-20/month

### Phase 3: PWA Enhancement (Future)
```
iPhone Safari (Offline)
     ↓
React PWA + ONNX Runtime Web
     ↓
Local Inference (WebGL/WASM)
     ↓
Canvas Rendering + Score Display
```

**Stack**: React, ONNX.js, Web Workers, Service Workers
**Hosting**: GitHub Pages / Cloudflare Pages
**Cost**: $0

---

## Performance Comparison

| Metric | Streamlit | Hybrid | Client-Side |
|--------|-----------|--------|-------------|
| FPS | 1-3 | 15-30 | 20-30 |
| Latency | 500-1000ms | 100-250ms | 40-70ms |
| Initial Load | 2s | 3-5s | 5-8s |
| Offline Support | ❌ | ❌ | ✅ |
| Server Cost | $0 | $5-20/mo | $0 |
| Development Time | 2-3 days | 4-5 weeks | 2-3 weeks |

---

## Implementation Roadmap

### Week 1: Streamlit MVP
- [ ] Day 1-2: Basic detection and UI
- [ ] Day 3-4: Score calculation logic
- [ ] Day 5: iPhone testing and refinement
- [ ] Deliverable: Working prototype for user testing

### Week 2-3: Model Optimization
- [ ] Week 2: ONNX conversion experiments
- [ ] Week 3: Performance benchmarking
- [ ] Deliverable: Optimized model ready for production

### Week 4-5: Hybrid Production
- [ ] Week 4: Backend development (FastAPI + WebSocket)
- [ ] Week 5: Frontend integration and deployment
- [ ] Deliverable: Production app on public URL

### Week 6+: Refinement and Phase 3 Planning
- [ ] User testing and bug fixes
- [ ] Analytics integration
- [ ] Evaluate Phase 3 (PWA) necessity

---

## Review and Retrospective

**Review Date**: End of Week 1 (after MVP testing)

**Key Questions**:
1. Is detection accuracy sufficient for real games?
2. Are there any iPhone Safari compatibility issues?
3. Is the score calculation logic correct?
4. Do users find the interface intuitive?
5. Should we proceed to Phase 2 or iterate on Phase 1?

**Success Criteria for Phase 2 Go/No-Go**:
- ✅ Detection accuracy > 85%
- ✅ No major Safari compatibility blockers
- ✅ Positive user feedback on UI/UX
- ✅ Score calculation validated against manual scoring
- ✅ Performance acceptable for testing purposes

---

## References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [MediaStream API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_API)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [iOS Safari WebKit Features](https://webkit.org/web-inspector/)

---

## Appendix: Alternative Options Rejected

### Django + Celery
**Reason**: Too complex for single developer, overkill for initial requirements

### Flask + Socket.IO
**Reason**: FastAPI has better async support and modern tooling

### TensorFlow.js (instead of ONNX.js)
**Reason**: ONNX has better YOLO support, smaller bundle size

### AWS Lambda
**Reason**: Cold starts unacceptable for real-time inference

### WebRTC for Streaming
**Reason**: WebSocket simpler for MVP, can add WebRTC in Phase 3
