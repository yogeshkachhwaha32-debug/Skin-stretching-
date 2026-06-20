// public/app.js — Skin Stretching Simulation (Optimized & Tear-Free)

// ─── Landmark Indices ───
const EYE_LEFT_OUTER = 33;
const EYE_RIGHT_OUTER = 263;

// ─── Global State ───
let showDebug = false;
let latestHands = [];
let latestFaces = [];
let activeLocks = new Map();
let luffyStretches = new Map();
let faceMouthTrackers = new Map();
let lastSpringTime = performance.now() / 1000.0;
let frameCounter = 0;
let lastFpsTime = performance.now();
let browserFps = 0.0;

// Persistent Entity ID Trackers
let prevHandCenters = new Map();
let nextHandId = 0;
let prevFaceCenters = new Map();
let nextFaceId = 0;
let pinchInactiveFrames = new Map();

// Smoothers cache
let faceSmoothers = new Map();
let handSmoothers = new Map();

// ─── DOM Elements ───
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const ctx = canvasElement.getContext('2d', { willReadFrequently: false });
const loadingBanner = document.getElementById('loading-banner');

// Offscreen canvas for downscaled MediaPipe inference input
const mpCanvas = document.createElement('canvas');
mpCanvas.width = 320;
mpCanvas.height = 180;
const mpCtx = mpCanvas.getContext('2d');

// Reusable offscreen canvas for skin-only warp mask
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');

// Pre-allocated reusable canvases for mouth warp (avoid per-frame allocation)
const _mouthCropCanvas = document.createElement('canvas');
const _mouthCropCtx = _mouthCropCanvas.getContext('2d');
const _mouthStretchCanvas = document.createElement('canvas');
const _mouthStretchCtx = _mouthStretchCanvas.getContext('2d');
const _mouthMaskCanvas = document.createElement('canvas');
const _mouthMaskCtx = _mouthMaskCanvas.getContext('2d');

// ─── Async MediaPipe Pipeline ───
let mpBusy = false;
let mpFrameQueued = false;

console.log("Initializing MediaPipe Holistic...");
const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
});
console.log("MediaPipe Holistic initialized successfully.");

// Callback Handlers
holistic.onResults((results) => {
    // 1. Populate latestFaces (expects face landmarks inside an array)
    latestFaces = [];
    if (results.faceLandmarks) {
        latestFaces.push(results.faceLandmarks);
    }
    
    // 2. Populate latestHands (expects [{ landmarks, label }])
    latestHands = [];
    if (results.leftHandLandmarks) {
        latestHands.push({
            landmarks: results.leftHandLandmarks,
            label: 'Left'
        });
    }
    if (results.rightHandLandmarks) {
        latestHands.push({
            landmarks: results.rightHandLandmarks,
            label: 'Right'
        });
    }
    
    mpBusy = false;
    // If a frame was queued while we were processing, send it now
    if (mpFrameQueued) {
        mpFrameQueued = false;
        sendFrameToMediaPipe();
    }
});

function sendFrameToMediaPipe() {
    if (mpBusy) {
        mpFrameQueued = true;
        return;
    }
    if (videoElement.paused || videoElement.ended || videoElement.readyState < 2) return;
    mpBusy = true;
    mpCtx.drawImage(videoElement, 0, 0, mpCanvas.width, mpCanvas.height);
    holistic.send({ image: mpCanvas }).catch(err => {
        console.error("MediaPipe send error:", err);
        mpBusy = false;
    });
}

// ─── Helper Functions ───
function calculateDistance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

// ─── EMA Smoothing Filter ───
class EMASmoother {
    constructor(alpha = 0.3) {
        this.alpha = alpha;
        this.value = null;
    }
    
    filter(nextValue) {
        if (this.value === null) {
            this.value = Array.isArray(nextValue) ? [...nextValue] : nextValue;
        } else {
            if (Array.isArray(nextValue)) {
                const a = this.alpha, b = 1.0 - a;
                for (let i = 0; i < nextValue.length; i++) {
                    this.value[i] = a * nextValue[i] + b * this.value[i];
                }
            } else {
                this.value = this.alpha * nextValue + (1.0 - this.alpha) * this.value;
            }
        }
        return this.value;
    }
    
    reset() {
        this.value = null;
    }
}

// Smooth list of coordinates (2D landmarks array)
class PointListSmoother {
    constructor(size, alpha = 0.3) {
        this.smoothers = Array.from({ length: size }, () => new EMASmoother(alpha));
    }
    
    filter(points) {
        if (!points) return points;
        while (points.length > this.smoothers.length) {
            this.smoothers.push(new EMASmoother(this.smoothers[0].alpha));
        }
        return points.map((p, idx) => {
            const val = this.smoothers[idx].filter([p.x, p.y]);
            return { x: val[0], y: val[1] };
        });
    }
    
    reset() {
        this.smoothers.forEach(s => s.reset());
    }
}

// ─── 1D Damped Spring-Mass Physics System ───
class Spring1D {
    constructor(restPos = 0.0, stiffness = 0.20, damping = 0.12, mass = 1.0) {
        this.restPos = restPos;
        this.stiffness = stiffness;
        this.damping = damping;
        this.mass = mass;
        this.pos = restPos;
        this.vel = 0.0;
    }
    
    update(timeStep = 1.0) {
        const displacement = this.pos - this.restPos;
        const springForce = -this.stiffness * displacement;
        const dampingForce = -this.damping * this.vel;
        const force = springForce + dampingForce;
        const accel = force / this.mass;
        
        this.vel += accel * timeStep;
        this.pos += this.vel * timeStep;
        return this.pos;
    }
}

// ─── Mouth Metrics Engine ───
class MouthTracker {
    constructor() {
        this.calibrationFrames = 0;
        this.calibrationLimit = 60;
        this.restNormalizedWidth = 0.5;
        this.restWidths = [];
    }
    
    getMouthMetrics(faceCoords) {
        if (!faceCoords || faceCoords.length < 300) return null;
        
        const pLeft = faceCoords[61];
        const pRight = faceCoords[291];
        const pUpper = faceCoords[0];
        const pLower = faceCoords[17];
        
        const pInnerUpper = faceCoords[13];
        const pInnerLower = faceCoords[14];
        
        const pEyeLeft = faceCoords[33];
        const pEyeRight = faceCoords[263];
        
        const width = calculateDistance(pLeft, pRight);
        const height = calculateDistance(pInnerUpper, pInnerLower);
        
        const centerX = Math.round((pLeft.x + pRight.x) / 2);
        const centerY = Math.round((pUpper.y + pLower.y) / 2);
        const center = { x: centerX, y: centerY };
        
        let eyeDistance = calculateDistance(pEyeLeft, pEyeRight);
        if (eyeDistance === 0) eyeDistance = 1.0;
        
        const normalizedWidth = width / eyeDistance;
        
        if (this.calibrationFrames < this.calibrationLimit) {
            this.restWidths.push(normalizedWidth);
            this.calibrationFrames++;
            if (this.calibrationFrames === this.calibrationLimit) {
                this.restNormalizedWidth = this.restWidths.reduce((a, b) => a + b, 0) / this.restWidths.length;
                console.log(`Mouth Calibration complete. Base rest normalized width: ${this.restNormalizedWidth.toFixed(4)}`);
            }
        }
        
        const stretchRatio = normalizedWidth / this.restNormalizedWidth;
        const isStretching = stretchRatio >= 1.35;
        const isOpen = height >= 25;
        
        return {
            width,
            height,
            center,
            stretchRatio,
            isStretching,
            isOpen,
            leftCorner: pLeft,
            rightCorner: pRight,
            upperLip: pUpper,
            lowerLip: pLower
        };
    }
}

// ─── Track entities frame-to-frame to assign persistent IDs ───
function trackEntities(handsList, facesList, w, h) {
    const trackedHands = new Map();
    const newHandCenters = new Map();
    
    handsList.forEach(hand => {
        const landmarks = getPixelCoords(hand.landmarks, w, h);
        const wrist = landmarks[0];
        const mcp = landmarks[9];
        const cx = Math.round((wrist.x + mcp.x) / 2);
        const cy = Math.round((wrist.y + mcp.y) / 2);
        
        let matchedId = null;
        let minDist = Infinity;
        for (let [hid, prevC] of prevHandCenters) {
            const d = Math.hypot(cx - prevC.x, cy - prevC.y);
            if (d < minDist && d < 180) {
                minDist = d;
                matchedId = hid;
            }
        }
        
        let handId;
        if (matchedId !== null && !trackedHands.has(matchedId)) {
            handId = matchedId;
        } else {
            handId = nextHandId++;
        }
        
        trackedHands.set(handId, { landmarks, label: hand.label });
        newHandCenters.set(handId, { x: cx, y: cy });
    });
    prevHandCenters = newHandCenters;
    
    const trackedFaces = new Map();
    const newFaceCenters = new Map();
    
    facesList.forEach(face => {
        const coords = getPixelCoords(face, w, h);
        const nose = coords[4];
        const cx = nose.x;
        const cy = nose.y;
        
        let matchedId = null;
        let minDist = Infinity;
        for (let [fid, prevC] of prevFaceCenters) {
            const d = Math.hypot(cx - prevC.x, cy - prevC.y);
            if (d < minDist && d < 200) {
                minDist = d;
                matchedId = fid;
            }
        }
        
        let faceId;
        if (matchedId !== null && !trackedFaces.has(matchedId)) {
            faceId = matchedId;
        } else {
            faceId = nextFaceId++;
        }
        
        trackedFaces.set(faceId, coords);
        newFaceCenters.set(faceId, { x: cx, y: cy });
    });
    prevFaceCenters = newFaceCenters;
    
    return { trackedHands, trackedFaces };
}

// Convert normalized landmarks to pixel coords
function getPixelCoords(landmarks, width, height) {
    return landmarks.map(lm => ({
        x: Math.round(lm.x * width),
        y: Math.round(lm.y * height)
    }));
}

// ─── Hysteresis pinch check ───
function detectPinch(hand, handId) {
    const landmarks = hand.landmarks;
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const pixelDist = calculateDistance(thumbTip, indexTip);
    
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];
    const handSize = calculateDistance(wrist, middleMcp);
    const handSizeVal = handSize === 0 ? 1.0 : handSize;
    
    const normalizedPinch = pixelDist / handSizeVal;
    const threshold = activeLocks.has(handId) ? 0.35 : 0.22;
    return normalizedPinch < threshold;
}

// ─── Canvas Resize Handler ───
function resizeCanvas() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Use a lower internal resolution for better performance while CSS scales it up
    const scale = Math.min(1, 1280 / vw);
    const cw = Math.round(vw * scale);
    const ch = Math.round(vh * scale);
    if (canvasElement.width !== cw || canvasElement.height !== ch) {
        canvasElement.width = cw;
        canvasElement.height = ch;
    }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ─── Main Frame Processing Loop ───
function updateAndDraw() {
    frameCounter++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        browserFps = (frameCounter * 1000) / (now - lastFpsTime);
        frameCounter = 0;
        lastFpsTime = now;
        const fpsEl = document.getElementById('fps-val');
        if (fpsEl) {
            fpsEl.innerText = browserFps.toFixed(1);
        }
    }
    
    const w = canvasElement.width;
    const h = canvasElement.height;
    
    // Draw original webcam frame
    ctx.drawImage(videoElement, 0, 0, w, h);
    
    // Track faces and hands with persistent ID classification mapping
    let { trackedHands, trackedFaces } = trackEntities(latestHands, latestFaces, w, h);
    
    // Hide loading banner on first successful frame
    if (trackedHands.size > 0 || trackedFaces.size > 0) {
        if (loadingBanner.style.display !== 'none') {
            loadingBanner.style.opacity = '0';
            setTimeout(() => { loadingBanner.style.display = 'none'; }, 500);
        }
    }
    
    // 1. Smooth Face Landmarks
    const smoothedFaces = new Map();
    for (let [fid, coords] of trackedFaces) {
        if (!faceSmoothers.has(fid)) {
            faceSmoothers.set(fid, new PointListSmoother(coords.length, 0.15));
        }
        smoothedFaces.set(fid, faceSmoothers.get(fid).filter(coords));
    }
    trackedFaces = smoothedFaces;
    
    for (let fid of faceSmoothers.keys()) {
        if (!trackedFaces.has(fid)) faceSmoothers.delete(fid);
    }
    
    // 2. Smooth Hand Landmarks
    const smoothedHands = new Map();
    for (let [hid, hand] of trackedHands) {
        if (!handSmoothers.has(hid)) {
            handSmoothers.set(hid, new PointListSmoother(hand.landmarks.length, 0.12));
        }
        const smoothedCoords = handSmoothers.get(hid).filter(hand.landmarks);
        smoothedHands.set(hid, {
            landmarks: smoothedCoords.map(pt => ({ x: Math.round(pt.x), y: Math.round(pt.y) })),
            label: hand.label
        });
    }
    trackedHands = smoothedHands;
    
    for (let hid of handSmoothers.keys()) {
        if (!trackedHands.has(hid)) handSmoothers.delete(hid);
    }
    
    // Debounce hand tracking loss for locks
    const currentHandIds = new Set(trackedHands.keys());
    for (let hid of activeLocks.keys()) {
        if (!currentHandIds.has(hid)) {
            const count = pinchInactiveFrames.get(hid) || 0;
            pinchInactiveFrames.set(hid, count + 1);
            if (count > 10) {
                activeLocks.delete(hid);
                pinchInactiveFrames.delete(hid);
            }
        }
    }
    
    // Update lock targets
    const activeStretches = [];
    const allProcessingHandIds = new Set([...trackedHands.keys(), ...activeLocks.keys()]);
    
    for (let handId of allProcessingHandIds) {
        const hand = trackedHands.get(handId);
        
        let isPinching = false;
        if (hand) {
            isPinching = detectPinch(hand, handId);
        }
        
        if (isPinching) {
            pinchInactiveFrames.set(handId, 0);
        } else {
            const count = pinchInactiveFrames.get(handId) || 0;
            pinchInactiveFrames.set(handId, count + 1);
        }
        
        const isPinchActive = isPinching || (activeLocks.has(handId) && pinchInactiveFrames.get(handId) <= 10);
        
        if (isPinchActive) {
            let currentPinchPos = null;
            if (hand) {
                const thumbTip = hand.landmarks[4];
                const indexTip = hand.landmarks[8];
                currentPinchPos = {
                    x: Math.round((thumbTip.x + indexTip.x) / 2),
                    y: Math.round((thumbTip.y + indexTip.y) / 2)
                };
            } else {
                currentPinchPos = activeLocks.get(handId).lastPinchPos;
            }
            
            if (!currentPinchPos) continue;
            
            let lock = activeLocks.get(handId);
            
            if (!lock && hand) {
                let bestType = null;
                let bestId = null;
                let bestLmIdx = -1;
                let minDist = Infinity;
                
                for (let [fid, coords] of trackedFaces) {
                    for (let idx = 0; idx < Math.min(coords.length, 468); idx++) {
                        const pt = coords[idx];
                        const d = calculateDistance(currentPinchPos, pt);
                        if (d < minDist) {
                            minDist = d;
                            bestType = 'face';
                            bestId = fid;
                            bestLmIdx = idx;
                        }
                    }
                }
                
                for (let [hid, handInfo] of trackedHands) {
                    if (hid === handId) continue;
                    for (let idx = 0; idx < handInfo.landmarks.length; idx++) {
                        const pt = handInfo.landmarks[idx];
                        const d = calculateDistance(currentPinchPos, pt);
                        if (d < minDist) {
                            minDist = d;
                            bestType = 'hand';
                            bestId = hid;
                            bestLmIdx = idx;
                        }
                    }
                }
                
                if (minDist < 180 && bestLmIdx !== -1) {
                    let initAnchor = null;
                    if (bestType === 'face') {
                        initAnchor = trackedFaces.get(bestId)[bestLmIdx];
                    } else {
                        initAnchor = trackedHands.get(bestId).landmarks[bestLmIdx];
                    }
                    
                    lock = {
                        anchorType: bestType,
                        anchorId: bestId,
                        landmarkIdx: bestLmIdx,
                        initialPinchPos: currentPinchPos,
                        initialAnchorPos: initAnchor,
                        initialDist: minDist,
                        lastPinchPos: currentPinchPos
                    };
                    activeLocks.set(handId, lock);
                }
            }
            
            if (lock) {
                lock.lastPinchPos = currentPinchPos;
                let targetExists = false;
                let anchorPos = null;
                let nominalLen = 50.0;
                let targetCoords = null;
                
                if (lock.anchorType === 'face') {
                    if (trackedFaces.has(lock.anchorId)) {
                        targetCoords = trackedFaces.get(lock.anchorId);
                        anchorPos = targetCoords[lock.landmarkIdx];
                        targetExists = true;
                        
                        const pEyeLeft = targetCoords[EYE_LEFT_OUTER];
                        const pEyeRight = targetCoords[EYE_RIGHT_OUTER];
                        const eyeDist = calculateDistance(pEyeLeft, pEyeRight);
                        nominalLen = (eyeDist > 0 ? eyeDist : 80.0) * 0.15;
                    }
                } else {
                    if (trackedHands.has(lock.anchorId)) {
                        targetCoords = trackedHands.get(lock.anchorId).landmarks;
                        anchorPos = targetCoords[lock.landmarkIdx];
                        targetExists = true;
                        
                        const wrist = targetCoords[0];
                        const middleMcp = targetCoords[9];
                        nominalLen = calculateDistance(wrist, middleMcp) * 0.25;
                    }
                }
                
                if (targetExists && anchorPos) {
                    const initP = lock.initialPinchPos;
                    const initA = lock.initialAnchorPos || anchorPos;
                    
                    const initOffsetX = initP.x - initA.x;
                    const initOffsetY = initP.y - initA.y;
                    
                    const currOffsetX = currentPinchPos.x - anchorPos.x;
                    const currOffsetY = currentPinchPos.y - anchorPos.y;
                    
                    const dragDx = currOffsetX - initOffsetX;
                    const dragDy = currOffsetY - initOffsetY;
                    const dragDist = Math.hypot(dragDx, dragDy);
                    
                    const normLen = nominalLen <= 0 ? 10.0 : nominalLen;
                    const scale = 1.0 + dragDist / normLen;
                    
                    activeStretches.push({
                        handId,
                        anchorType: lock.anchorType,
                        anchorId: lock.anchorId,
                        landmarkIdx: lock.landmarkIdx,
                        anchorPos,
                        pinchPos: currentPinchPos,
                        scale,
                        isPinching: true,
                        targetCoords,
                        nominalLen: normLen,
                        dragVector: { x: dragDx, y: dragDy },
                        dragDist,
                        initialPinchPos: initP
                    });
                } else {
                    activeLocks.delete(handId);
                }
            }
        } else {
            activeLocks.delete(handId);
        }
    }
    
    // ─── Physics Updates (Mass-Damping-Stiffness) ───
    const currentTime = performance.now() / 1000.0;
    const dt = currentTime - lastSpringTime;
    lastSpringTime = currentTime;
    const timeStep = Math.max(0.1, Math.min(dt * 60.0, 3.0));
    
    for (let s of luffyStretches.values()) {
        s.isActive = false;
    }
    
    activeStretches.forEach(active => {
        const hid = active.handId;
        if (!luffyStretches.has(hid)) {
            luffyStretches.set(hid, {
                springDist: new Spring1D(0.0, 0.18, 0.22)
            });
        }
        const s = luffyStretches.get(hid);
        s.isActive = true;
        s.anchorType = active.anchorType;
        s.anchorId = active.anchorId;
        s.landmarkIdx = active.landmarkIdx;
        s.springDist.restPos = active.dragDist;
        
        if (active.dragDist > 0.1) {
            s.lastDragVector = { x: active.dragVector.x / active.dragDist, y: active.dragVector.y / active.dragDist };
        } else {
            s.lastDragVector = { x: 1.0, y: 0.0 };
        }
        s.lastNominalLen = active.nominalLen;
    });
    
    const stretchesToDelete = [];
    let maxScale = 1.00;
    let hasActivePinch = false;
    
    for (let [hid, s] of luffyStretches) {
        if (!s.isActive) {
            s.springDist.restPos = 0.0;
        }
        s.springDist.update(timeStep);
        
        if (s.isActive) {
            hasActivePinch = true;
        }
        
        const warpDist = s.springDist.pos;
        const scaleVal = 1.0 + warpDist / s.lastNominalLen;
        maxScale = Math.max(maxScale, scaleVal);
        
        if (!s.isActive && warpDist < 1.0) {
            stretchesToDelete.push(hid);
            continue;
        }
        
        let coords = null;
        let isFace = true;
        if (s.anchorType === 'face') {
            if (trackedFaces.has(s.anchorId)) {
                coords = trackedFaces.get(s.anchorId);
            }
        } else {
            if (trackedHands.has(s.anchorId)) {
                coords = trackedHands.get(s.anchorId).landmarks;
                isFace = false;
            }
        }
        
        const anchorIdx = s.landmarkIdx;
        if (coords && anchorIdx < coords.length) {
            const anchorPos = coords[anchorIdx];
            
            const warpPinchPos = {
                x: Math.round(anchorPos.x + s.lastDragVector.x * warpDist),
                y: Math.round(anchorPos.y + s.lastDragVector.y * warpDist)
            };
            
            warpImageDirectional(coords, anchorIdx, warpPinchPos, isFace);
        }
    }
    
    stretchesToDelete.forEach(hid => luffyStretches.delete(hid));
    
    // Clean mouth trackers for deleted faces
    for (let fid of faceMouthTrackers.keys()) {
        if (!trackedFaces.has(fid)) {
            faceMouthTrackers.delete(fid);
        }
    }
    
    // Fallback mouth stretch
    let hasMouthStretch = false;
    if (!hasActivePinch && trackedFaces.size > 0) {
        let primaryFid = null;
        let maxEyeDist = -1;
        
        for (let [fid, coords] of trackedFaces) {
            if (coords.length >= 264) {
                const eyeDist = calculateDistance(coords[33], coords[263]);
                if (eyeDist > maxEyeDist) {
                    maxEyeDist = eyeDist;
                    primaryFid = fid;
                }
            }
        }
        
        if (primaryFid !== null) {
            const coords = trackedFaces.get(primaryFid);
            if (!faceMouthTrackers.has(primaryFid)) {
                faceMouthTrackers.set(primaryFid, new MouthTracker());
            }
            const mTracker = faceMouthTrackers.get(primaryFid);
            const mMetrics = mTracker.getMouthMetrics(coords);
            
            if (mMetrics && (mMetrics.isStretching || mMetrics.isOpen)) {
                const margin = 25;
                const leftCorner = mMetrics.leftCorner;
                const rightCorner = mMetrics.rightCorner;
                
                if (leftCorner.x > margin && leftCorner.x < w - margin &&
                    leftCorner.y > margin && leftCorner.y < h - margin &&
                    rightCorner.x > margin && rightCorner.x < w - margin &&
                    rightCorner.y > margin && rightCorner.y < h - margin) {
                    
                    const stretchScale = mMetrics.stretchRatio;
                    const heightFactor = mMetrics.height / 20.0;
                    const targetX = 1.0 + (stretchScale - 1.0) * 1.8;
                    const targetY = 1.0 + (heightFactor - 1.0) * 1.8;
                    
                    warpImageMouth(mMetrics, targetX, targetY);
                    maxScale = Math.max(maxScale, stretchScale);
                    hasMouthStretch = true;
                }
            }
        }
    }
    
    // Update UI Telemetry
    const stretchVal = document.getElementById('stretch-val');
    if (stretchVal) {
        stretchVal.innerText = maxScale.toFixed(2) + 'x';
    }
    const triggerVal = document.getElementById('trigger-val');
    if (triggerVal) {
        if (hasActivePinch) {
            triggerVal.innerText = 'PINCH DRAG';
            triggerVal.style.color = 'var(--primary)';
        } else if (hasMouthStretch) {
            triggerVal.innerText = 'MOUTH STRETCH';
            triggerVal.style.color = 'var(--accent)';
        } else {
            triggerVal.innerText = 'None';
            triggerVal.style.color = '#fff';
        }
    }
    
    // Debug Overlays
    if (showDebug) {
        for (let [fid, coords] of trackedFaces) {
            ctx.fillStyle = '#00ff00';
            for (let i = 0; i < coords.length; i++) {
                const pt = coords[i];
                ctx.fillRect(pt.x - 1, pt.y - 1, 2, 2);
            }
        }
        
        for (let [hid, hand] of trackedHands) {
            const landmarks = hand.landmarks;
            const color = hand.label === 'Left' ? '#00d2ff' : '#ffd700';
            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            
            for (let i = 0; i < landmarks.length; i++) {
                const pt = landmarks[i];
                ctx.fillRect(pt.x - 3, pt.y - 3, 6, 6);
            }
            
            const chains = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [5, 9, 13, 17],
                [0, 17, 18, 19, 20]
            ];
            
            chains.forEach(chain => {
                ctx.beginPath();
                ctx.moveTo(landmarks[chain[0]].x, landmarks[chain[0]].y);
                for (let i = 1; i < chain.length; i++) {
                    ctx.lineTo(landmarks[chain[i]].x, landmarks[chain[i]].y);
                }
                ctx.stroke();
            });
        }
    }
}

// ─── Bilinear pixel interpolation mapping ───
function bilinearRemap(src, w, h, x, y, dst, dstIdx) {
    if (x < 0) x = 0;
    if (x > w - 1) x = w - 1;
    if (y < 0) y = 0;
    if (y > h - 1) y = h - 1;
    
    const x0 = x | 0; // fast floor
    const x1 = x0 < w - 1 ? x0 + 1 : x0;
    const y0 = y | 0;
    const y1 = y0 < h - 1 ? y0 + 1 : y0;
    
    const dx = x - x0;
    const dy = y - y0;
    const dxI = 1 - dx;
    const dyI = 1 - dy;
    
    const idx00 = (y0 * w + x0) << 2;
    const idx01 = (y0 * w + x1) << 2;
    const idx10 = (y1 * w + x0) << 2;
    const idx11 = (y1 * w + x1) << 2;
    
    const w00 = dxI * dyI;
    const w01 = dx * dyI;
    const w10 = dxI * dy;
    const w11 = dx * dy;
    
    dst[dstIdx]     = (w00 * src[idx00]     + w01 * src[idx01]     + w10 * src[idx10]     + w11 * src[idx11]    ) + 0.5 | 0;
    dst[dstIdx + 1] = (w00 * src[idx00 + 1] + w01 * src[idx01 + 1] + w10 * src[idx10 + 1] + w11 * src[idx11 + 1]) + 0.5 | 0;
    dst[dstIdx + 2] = (w00 * src[idx00 + 2] + w01 * src[idx01 + 2] + w10 * src[idx10 + 2] + w11 * src[idx11 + 2]) + 0.5 | 0;
    dst[dstIdx + 3] = 255; // Alpha always fully opaque
}

// ─── Directional Warp Algorithm (IMPROVED — no tearing) ───
function warpImageDirectional(coords, anchorIdx, pinchPos, isFace) {
    const anchorPos = coords[anchorIdx];
    const ax = anchorPos.x;
    const ay = anchorPos.y;
    const px = pinchPos.x;
    const py = pinchPos.y;
    
    const dx = px - ax;
    const dy = py - ay;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    if (dist < 2) return;
    
    const ux = dx / dist;
    const uy = dy / dist;
    
    let R, R_in, R_out;
    if (isFace && coords.length >= 264) {
        const eyeDistance = calculateDistance(coords[33], coords[263]);
        const base = eyeDistance === 0 ? 1.0 : eyeDistance;
        R = Math.round(base * 1.4);
        R_in = Math.round(base * 1.1);
        R_out = Math.round(base * 0.4);
    } else {
        if (coords.length >= 10) {
            const handSize = calculateDistance(coords[0], coords[9]);
            const base = handSize === 0 ? 1.0 : handSize;
            R = Math.round(base * 1.1);
            R_in = Math.round(base * 0.9);
            R_out = Math.round(base * 0.35);
        } else {
            R = 100; R_in = 80; R_out = 30;
        }
    }
    
    const canvasW = canvasElement.width;
    const canvasH = canvasElement.height;
    
    const margin = Math.max(R, R_in) + 15;
    const xMin = Math.max(0, Math.floor(Math.min(ax, px) - margin));
    const yMin = Math.max(0, Math.floor(Math.min(ay, py) - margin));
    const xMax = Math.min(canvasW, Math.ceil(Math.max(ax, px) + margin));
    const yMax = Math.min(canvasH, Math.ceil(Math.max(ay, py) + margin));
    
    const w = xMax - xMin;
    const h = yMax - yMin;
    
    if (w <= 10 || h <= 10) return;
    
    // Helper to draw the skin paths
    function drawSkinPath(maskContext, xOffset, yOffset) {
        if (isFace && coords.length >= 264) {
            const faceOutline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
            maskContext.beginPath();
            const startPt = coords[faceOutline[0]];
            maskContext.moveTo(startPt.x - xOffset, startPt.y - yOffset);
            for (let i = 1; i < faceOutline.length; i++) {
                const pt = coords[faceOutline[i]];
                if (pt) {
                    maskContext.lineTo(pt.x - xOffset, pt.y - yOffset);
                }
            }
            maskContext.closePath();
            maskContext.fill();
        } else {
            maskContext.lineWidth = Math.round(R * 0.8);
            maskContext.lineCap = 'round';
            maskContext.lineJoin = 'round';
            const chains = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [5, 9, 13, 17],
                [0, 17, 18, 19, 20]
            ];
            chains.forEach(chain => {
                maskContext.beginPath();
                const startPt = coords[chain[0]];
                maskContext.moveTo(startPt.x - xOffset, startPt.y - yOffset);
                for (let i = 1; i < chain.length; i++) {
                    const pt = coords[chain[i]];
                    maskContext.lineTo(pt.x - xOffset, pt.y - yOffset);
                }
                maskContext.stroke();
            });
            coords.forEach(pt => {
                maskContext.beginPath();
                maskContext.arc(pt.x - xOffset, pt.y - yOffset, R * 0.4, 0, 2 * Math.PI);
                maskContext.fill();
            });
        }
    }

    // Draw the soft feathered mask for blending (prevents hard-edge tearing)
    maskCanvas.width = w;
    maskCanvas.height = h;
    maskCtx.clearRect(0, 0, w, h);
    
    // First pass: binary mask for skin detection
    maskCtx.filter = 'none';
    maskCtx.fillStyle = '#000';
    maskCtx.fillRect(0, 0, w, h);
    maskCtx.fillStyle = '#fff';
    maskCtx.strokeStyle = '#fff';
    drawSkinPath(maskCtx, xMin, yMin);
    const binaryMaskData = maskCtx.getImageData(0, 0, w, h).data;
    
    // Second pass: feathered mask for smooth blending at boundaries
    maskCtx.fillStyle = '#000';
    maskCtx.fillRect(0, 0, w, h);
    const featherPx = Math.max(4, Math.round(R * 0.08));
    maskCtx.filter = `blur(${featherPx}px)`;
    maskCtx.fillStyle = '#fff';
    maskCtx.strokeStyle = '#fff';
    drawSkinPath(maskCtx, xMin, yMin);
    maskCtx.filter = 'none';
    const featheredMaskData = maskCtx.getImageData(0, 0, w, h).data;
    
    const imgData = ctx.getImageData(xMin, yMin, w, h);
    const src = imgData.data;
    const dst = new Uint8ClampedArray(src.length);
    dst.set(src);
    
    const RSq = R * R;
    const invRSq = 1.0 / RSq;
    const wx0 = xMin - ax;
    
    // Reduced strength to prevent tearing at high stretch
    const strength = 0.85;
    
    for (let y = 0; y < h; y++) {
        const globalY = y + yMin;
        const wy = globalY - ay;
        const wy2 = wy * wy;
        const t_base = wx0 * ux + wy * uy;
        
        let wx = wx0;
        let t = t_base;
        
        for (let x = 0; x < w; x++) {
            const d2 = wx * wx + wy2;
            const dPerpSq = d2 - t * t;
            
            if (dPerpSq < RSq) {
                const diff = 1.0 - dPerpSq * invRSq;
                // Smoother cubic falloff instead of quadratic (prevents abrupt edges)
                const g = diff * diff * diff;
                const k = g * strength;
                
                let disp = 0;
                if (t >= -R_in && t <= dist) {
                    disp = k * (t + R_in) * (dist / (R_in + dist));
                } else if (t > dist && t < dist + R_out) {
                    const diff2 = 1.0 - (t - dist) / R_out;
                    disp = dist * k * diff2 * diff2;
                }
                
                if (disp > 0.5) {
                    const srcLocalX = x - disp * ux;
                    const srcLocalY = y - disp * uy;
                    
                    const srcXInt = (srcLocalX + 0.5) | 0;
                    const srcYInt = (srcLocalY + 0.5) | 0;
                    
                    // Check feathered mask alpha at destination
                    const dstMaskIdx = (y * w + x) * 4;
                    const dstAlpha = featheredMaskData[dstMaskIdx];
                    
                    if (dstAlpha < 5) {
                        wx++;
                        t += ux;
                        continue;
                    }
                    
                    // Check if source pixel is within skin (binary mask)
                    let isSrcSkin = false;
                    if (srcXInt >= 0 && srcXInt < w && srcYInt >= 0 && srcYInt < h) {
                        isSrcSkin = binaryMaskData[(srcYInt * w + srcXInt) * 4] > 127;
                    }
                    
                    const isDstSkin = binaryMaskData[dstMaskIdx] > 127;
                    
                    if (isDstSkin || isSrcSkin) {
                        let finalSrcX = srcLocalX;
                        let finalSrcY = srcLocalY;
                        
                        if (isDstSkin && !isSrcSkin) {
                            // Improved binary search with 8 iterations for smoother clamping
                            let low = 0;
                            let high = disp;
                            for (let i = 0; i < 8; i++) {
                                const mid = (low + high) * 0.5;
                                const tx = x - mid * ux;
                                const ty = y - mid * uy;
                                const txInt = (tx + 0.5) | 0;
                                const tyInt = (ty + 0.5) | 0;
                                if (txInt >= 0 && txInt < w && tyInt >= 0 && tyInt < h && binaryMaskData[(tyInt * w + txInt) * 4] > 127) {
                                    low = mid;
                                } else {
                                    high = mid;
                                }
                            }
                            finalSrcX = x - low * ux;
                            finalSrcY = y - low * uy;
                        }
                        
                        // Bilinear remap into a temporary pixel
                        const tmpIdx = dstMaskIdx; // reuse same index for dst
                        bilinearRemap(src, w, h, finalSrcX, finalSrcY, dst, tmpIdx);
                        
                        // Alpha-blend using feathered mask to eliminate hard seam tearing
                        if (dstAlpha < 250) {
                            const blendA = dstAlpha / 255.0;
                            const blendB = 1.0 - blendA;
                            dst[tmpIdx]     = (dst[tmpIdx]     * blendA + src[tmpIdx]     * blendB + 0.5) | 0;
                            dst[tmpIdx + 1] = (dst[tmpIdx + 1] * blendA + src[tmpIdx + 1] * blendB + 0.5) | 0;
                            dst[tmpIdx + 2] = (dst[tmpIdx + 2] * blendA + src[tmpIdx + 2] * blendB + 0.5) | 0;
                        }
                    }
                }
            }
            wx++;
            t += ux;
        }
    }
    
    imgData.data.set(dst);
    ctx.putImageData(imgData, xMin, yMin);
}

// ─── Composition-Based Feathered Mouth Warper (reuses cached canvases) ───
function warpImageMouth(mouthMetrics, scaleX, scaleY) {
    const center = mouthMetrics.center;
    const width = mouthMetrics.width;
    const cH = canvasElement.height;
    const cW = canvasElement.width;
    
    const bw = Math.round(width * 2.2);
    const bh = Math.round(width * 1.1);
    
    let bx = Math.max(0, center.x - Math.round(bw / 2));
    let by = Math.max(0, center.y - Math.round(bh / 2));
    const finalBw = Math.min(cW - bx, bw);
    const finalBh = Math.min(cH - by, bh);
    
    if (finalBw <= 15 || finalBh <= 15) return;
    
    // Crop the mouth from output canvas (reuse cached canvas)
    if (_mouthCropCanvas.width !== finalBw || _mouthCropCanvas.height !== finalBh) {
        _mouthCropCanvas.width = finalBw;
        _mouthCropCanvas.height = finalBh;
    } else {
        _mouthCropCtx.clearRect(0, 0, finalBw, finalBh);
    }
    _mouthCropCtx.drawImage(canvasElement, bx, by, finalBw, finalBh, 0, 0, finalBw, finalBh);
    
    // Stretched canvas
    const stretchedW = Math.round(finalBw * scaleX);
    const stretchedH = Math.round(finalBh * scaleY);
    if (stretchedW <= 10 || stretchedH <= 10) return;
    
    if (_mouthStretchCanvas.width !== stretchedW || _mouthStretchCanvas.height !== stretchedH) {
        _mouthStretchCanvas.width = stretchedW;
        _mouthStretchCanvas.height = stretchedH;
    } else {
        _mouthStretchCtx.clearRect(0, 0, stretchedW, stretchedH);
    }
    _mouthStretchCtx.globalCompositeOperation = 'source-over';
    _mouthStretchCtx.drawImage(_mouthCropCanvas, 0, 0, finalBw, finalBh, 0, 0, stretchedW, stretchedH);
    
    // Create soft elliptical mask (reuse cached canvas)
    if (_mouthMaskCanvas.width !== stretchedW || _mouthMaskCanvas.height !== stretchedH) {
        _mouthMaskCanvas.width = stretchedW;
        _mouthMaskCanvas.height = stretchedH;
    } else {
        _mouthMaskCtx.clearRect(0, 0, stretchedW, stretchedH);
    }
    
    const rx = Math.max(1, Math.round(stretchedW / 2) - 2);
    const ry = Math.max(1, Math.round(stretchedH / 2) - 2);
    
    _mouthMaskCtx.fillStyle = '#000';
    _mouthMaskCtx.fillRect(0, 0, stretchedW, stretchedH);
    
    const blurSize = Math.max(5, Math.round(width * 0.1));
    _mouthMaskCtx.filter = `blur(${blurSize}px)`;
    _mouthMaskCtx.fillStyle = '#fff';
    _mouthMaskCtx.beginPath();
    _mouthMaskCtx.ellipse(Math.round(stretchedW / 2), Math.round(stretchedH / 2), Math.max(1, rx - blurSize), Math.max(1, ry - blurSize), 0, 0, 2 * Math.PI);
    _mouthMaskCtx.fill();
    _mouthMaskCtx.filter = 'none';
    
    // Blend stretched mouth with mask using destination-in
    _mouthStretchCtx.globalCompositeOperation = 'destination-in';
    _mouthStretchCtx.drawImage(_mouthMaskCanvas, 0, 0);
    _mouthStretchCtx.globalCompositeOperation = 'source-over';
    
    // Paint blended mouth back onto main canvas
    let tx = center.x - Math.round(stretchedW / 2);
    let ty = center.y - Math.round(stretchedH / 2);
    
    tx = Math.max(5, Math.min(cW - stretchedW - 5, tx));
    ty = Math.max(5, Math.min(cH - stretchedH - 5, ty));
    
    ctx.drawImage(_mouthStretchCanvas, tx, ty);
}

// ─── Toggle Debug View ───
function toggleDebug() {
    showDebug = !showDebug;
    const btn = document.getElementById('dbg-button');
    if (btn) {
        if (showDebug) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }
}

// ─── Decoupled Render Loop ───
let isLoopRunning = false;
let mpInterval = null;

function renderLoop() {
    if (videoElement.readyState >= 2 && !videoElement.paused && !videoElement.ended) {
        updateAndDraw();
    }
    requestAnimationFrame(renderLoop);
}

// ─── Start camera capture and media loops ───
async function init() {
    console.log("init() called. Requesting webcam access...");
    try {
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
                audio: false
            });
        } catch (hdError) {
            console.warn("HD camera request failed, trying generic:", hdError);
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }
        console.log("Webcam stream access granted.");
        videoElement.srcObject = stream;
        
        function startLoops() {
            videoElement.play();
            resizeCanvas();
            if (!isLoopRunning) {
                isLoopRunning = true;
                // Render loop runs at display refresh rate (60Hz)
                requestAnimationFrame(renderLoop);
                // MediaPipe inference runs independently at ~15-20 fps to avoid blocking rendering
                mpInterval = setInterval(sendFrameToMediaPipe, 55);
            }
        }
        
        if (videoElement.readyState >= 2) {
            startLoops();
        } else {
            videoElement.addEventListener('loadedmetadata', startLoops, { once: true });
        }
    } catch (err) {
        console.error("Webcam access failed:", err);
        if (loadingBanner) {
            loadingBanner.innerHTML = '⚠️ WEBCAM ACCESS DENIED — Please allow camera permissions and reload.';
            loadingBanner.style.borderColor = '#ff5252';
            loadingBanner.style.color = '#ff5252';
        }
    }
}

// Launch
init();
