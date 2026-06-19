// public/app.js

// Landmark Indices
const EYE_LEFT_OUTER = 33;
const EYE_RIGHT_OUTER = 263;

// Global State
let showDebug = false;
let latestHands = [];
let latestFaces = [];
let activeLocks = new Map(); // handId -> Lock object
let luffyStretches = new Map(); // handId -> Spring object
let faceMouthTrackers = new Map(); // faceId -> MouthTracker object
let lastSpringTime = performance.now() / 1000.0;
let frameCounter = 0;
let lastFpsTime = performance.now();
let browserFps = 0.0;

// Persistent Entity ID Trackers
let prevHandCenters = new Map(); // handId -> {x, y}
let nextHandId = 0;
let prevFaceCenters = new Map(); // faceId -> {x, y}
let nextFaceId = 0;
let pinchInactiveFrames = new Map(); // handId -> count

// Smoothers cache
let faceSmoothers = new Map(); // faceId -> PointListSmoother
let handSmoothers = new Map(); // handId -> PointListSmoother

// Grab elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const ctx = canvasElement.getContext('2d', { willReadFrequently: true });
const loadingBanner = document.getElementById('loading-banner');

// Offscreen canvas for downscaled MediaPipe inference input
const mpCanvas = document.createElement('canvas');
mpCanvas.width = 384;
mpCanvas.height = 216;
const mpCtx = mpCanvas.getContext('2d');

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
});

// Helper Functions
function calculateDistance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

// EMA Smoothing Filter
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
                this.value = nextValue.map((n, idx) => this.alpha * n + (1.0 - this.alpha) * this.value[idx]);
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

// 1D Damped Spring-Mass Physics System
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

// Mouth Metrics Engine
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
        const isStretching = stretchRatio >= 1.35; // MOUTH_STRETCH_THRESHOLD
        const isOpen = height >= 25; // MOUTH_OPEN_THRESHOLD
        
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

// Track entities frame-to-frame to assign persistent IDs
function trackEntities(handsList, facesList, w, h) {
    const trackedHands = new Map();
    const newHandCenters = new Map();
    
    // 1. Track Hands
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
    
    // 2. Track Faces
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

// Hysteresis pinch check
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

// Main Frame Processing Loop
async function updateAndDraw() {
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
    
    // Draw original webcam frame unmirrored (CSS mirrors the canvas)
    ctx.drawImage(videoElement, 0, 0, w, h);
    
    // Track faces and hands with persistent ID classification mapping
    let { trackedHands, trackedFaces } = trackEntities(latestHands, latestFaces, w, h);
    
    // Hide loading banner on first successful frame
    if (trackedHands.size > 0 || trackedFaces.size > 0) {
        loadingBanner.style.opacity = 0;
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
    
    // Clean up unused face smoothers
    for (let fid of faceSmoothers.keys()) {
        if (!trackedFaces.has(fid)) {
            faceSmoothers.delete(fid);
        }
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
    
    // Clean up unused hand smoothers
    for (let hid of handSmoothers.keys()) {
        if (!trackedHands.has(hid)) {
            handSmoothers.delete(hid);
        }
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
                // Search for closest landmark among all faces and other hands
                let bestType = null;
                let bestId = null;
                let bestLmIdx = -1;
                let minDist = Infinity;
                
                // Search faces
                for (let [fid, coords] of trackedFaces) {
                    coords.forEach((pt, idx) => {
                        if (idx >= 468) return;
                        const d = calculateDistance(currentPinchPos, pt);
                        if (d < minDist) {
                            minDist = d;
                            bestType = 'face';
                            bestId = fid;
                            bestLmIdx = idx;
                        }
                    });
                }
                
                // Search other hands (finger stretching) - skip self
                for (let [hid, handInfo] of trackedHands) {
                    if (hid === handId) continue;
                    handInfo.landmarks.forEach((pt, idx) => {
                        const d = calculateDistance(currentPinchPos, pt);
                        if (d < minDist) {
                            minDist = d;
                            bestType = 'hand';
                            bestId = hid;
                            bestLmIdx = idx;
                        }
                    });
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
    
    // Physics Updates (Mass-Damping-Stiffness)
    const currentTime = performance.now() / 1000.0;
    const dt = currentTime - lastSpringTime;
    lastSpringTime = currentTime;
    const timeStep = Math.max(0.1, Math.min(dt * 60.0, 3.0));
    
    // Reset activity check flags
    for (let s of luffyStretches.values()) {
        s.isActive = false;
    }
    
    // Apply updates for new/active stretches
    activeStretches.forEach(active => {
        const hid = active.handId;
        if (!luffyStretches.has(hid)) {
            luffyStretches.set(hid, {
                springDist: new Spring1D(0.0, 0.18, 0.22) // stiffness 0.18, damping 0.22 for more accurate tracking
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
    
    // Process spring physics iteration & warp calculations
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
            
            // Displace along the unit relative drag vector by the spring's warp distance
            const warpPinchPos = {
                x: Math.round(anchorPos.x + s.lastDragVector.x * warpDist),
                y: Math.round(anchorPos.y + s.lastDragVector.y * warpDist)
            };
            
            warpImageDirectional(coords, anchorIdx, warpPinchPos, isFace);
        }
    }
    
    // Delete finished springs
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
                
                // Edge protection
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
        // Draw face landmarks
        for (let [fid, coords] of trackedFaces) {
            ctx.fillStyle = '#00ff00';
            coords.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 1.5, 0, 2 * Math.PI);
                ctx.fill();
            });
        }
        
        // Draw hand landmarks and chains
        for (let [hid, hand] of trackedHands) {
            const landmarks = hand.landmarks;
            const color = hand.label === 'Left' ? '#00d2ff' : '#ffd700';
            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            
            landmarks.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
                ctx.fill();
            });
            
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

// Bilinear pixel interpolation mapping
function bilinearRemap(src, w, h, x, y, dst, dstIdx) {
    if (x < 0) x = 0;
    if (x > w - 1) x = w - 1;
    if (y < 0) y = 0;
    if (y > h - 1) y = h - 1;
    
    const x0 = Math.floor(x);
    const x1 = Math.min(w - 1, x0 + 1);
    const y0 = Math.floor(y);
    const y1 = Math.min(h - 1, y0 + 1);
    
    const dx = x - x0;
    const dy = y - y0;
    
    const idx00 = (y0 * w + x0) * 4;
    const idx01 = (y0 * w + x1) * 4;
    const idx10 = (y1 * w + x0) * 4;
    const idx11 = (y1 * w + x1) * 4;
    
    for (let i = 0; i < 4; i++) {
        const val00 = src[idx00 + i];
        const val01 = src[idx01 + i];
        const val10 = src[idx10 + i];
        const val11 = src[idx11 + i];
        
        const val = (1 - dx) * (1 - dy) * val00 +
                    dx * (1 - dy) * val01 +
                    (1 - dx) * dy * val10 +
                    dx * dy * val11;
                    
        dst[dstIdx + i] = Math.round(val);
    }
}

// Directional Warp Algorithm in JS Canvas Image Data
function warpImageDirectional(coords, anchorIdx, pinchPos, isFace) {
    const anchorPos = coords[anchorIdx];
    const ax = anchorPos.x;
    const ay = anchorPos.y;
    const px = pinchPos.x;
    const py = pinchPos.y;
    
    const dx = px - ax;
    const dy = py - ay;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    if (dist < 3) return;
    
    const ux = dx / dist;
    const uy = dy / dist;
    
    let R, R_in, R_out;
    if (isFace && coords.length >= 264) {
        const eyeDistance = calculateDistance(coords[33], coords[263]);
        const base = eyeDistance === 0 ? 1.0 : eyeDistance;
        R = Math.round(base * 0.85);
        R_in = Math.round(base * 0.75);
        R_out = Math.round(base * 0.2);
    } else {
        if (coords.length >= 10) {
            const handSize = calculateDistance(coords[0], coords[9]);
            const base = handSize === 0 ? 1.0 : handSize;
            R = Math.round(base * 0.65);
            R_in = Math.round(base * 0.55);
            R_out = Math.round(base * 0.2);
        } else {
            R = 60; R_in = 50; R_out = 15;
        }
    }
    
    const canvasW = canvasElement.width;
    const canvasH = canvasElement.height;
    
    const margin = Math.max(R, R_in) + 10;
    const xMin = Math.max(0, Math.floor(Math.min(ax, px) - margin));
    const yMin = Math.max(0, Math.floor(Math.min(ay, py) - margin));
    const xMax = Math.min(canvasW, Math.ceil(Math.max(ax, px) + margin));
    const yMax = Math.min(canvasH, Math.ceil(Math.max(ay, py) + margin));
    
    const w = xMax - xMin;
    const h = yMax - yMin;
    
    if (w <= 10 || h <= 10) return;
    
    const imgData = ctx.getImageData(xMin, yMin, w, h);
    const src = imgData.data;
    const dstData = ctx.createImageData(w, h);
    dstData.data.set(src);
    const dst = dstData.data;
    
    const RSq = R * R;
    const wx0 = xMin - ax;
    const ux_step = ux;
    
    for (let y = 0; y < h; y++) {
        const globalY = y + yMin;
        const wy = globalY - ay;
        const wy2 = wy * wy;
        const t_const = wx0 * ux + wy * uy;
        
        let wx = wx0;
        let t = t_const;
        
        for (let x = 0; x < w; x++) {
            const d2 = wx * wx + wy2;
            const dPerpSq = d2 - t * t;
            
            if (dPerpSq < RSq) {
                const diff = 1.0 - dPerpSq / RSq;
                const g = diff * diff;
                const strength = 0.98;
                const k = g * strength;
                
                let disp = 0;
                if (t >= -R_in && t <= dist) {
                    disp = k * (t + R_in) * (dist / (R_in + dist));
                } else if (t > dist && t < dist + R_out) {
                    const diff2 = 1.0 - (t - dist) / R_out;
                    disp = dist * k * diff2 * diff2;
                }
                
                if (disp !== 0) {
                    const srcLocalX = x - disp * ux;
                    const srcLocalY = y - disp * uy;
                    
                    bilinearRemap(src, w, h, srcLocalX, srcLocalY, dst, (y * w + x) * 4);
                }
            }
            wx++;
            t += ux_step;
        }
    }
    
    ctx.putImageData(dstData, xMin, yMin);
}

// Composition-Based Feathered Mouth Warper (GPU Accelerated)
function warpImageMouth(mouthMetrics, scaleX, scaleY) {
    const center = mouthMetrics.center;
    const width = mouthMetrics.width;
    const h = canvasElement.height;
    const w = canvasElement.width;
    
    const bw = Math.round(width * 2.2);
    const bh = Math.round(width * 1.1);
    
    let bx = Math.max(0, center.x - Math.round(bw / 2));
    let by = Math.max(0, center.y - Math.round(bh / 2));
    const finalBw = Math.min(w - bx, bw);
    const finalBh = Math.min(h - by, bh);
    
    if (finalBw <= 15 || finalBh <= 15) return;
    
    // Crop the mouth from output canvas
    const offscreenMouth = document.createElement('canvas');
    offscreenMouth.width = finalBw;
    offscreenMouth.height = finalBh;
    const offCtx = offscreenMouth.getContext('2d');
    offCtx.drawImage(canvasElement, bx, by, finalBw, finalBh, 0, 0, finalBw, finalBh);
    
    // Stretched canvas
    const stretchedW = Math.round(finalBw * scaleX);
    const stretchedH = Math.round(finalBh * scaleY);
    if (stretchedW <= 10 || stretchedH <= 10) return;
    
    const stretchedCanvas = document.createElement('canvas');
    stretchedCanvas.width = stretchedW;
    stretchedCanvas.height = stretchedH;
    const strCtx = stretchedCanvas.getContext('2d');
    strCtx.drawImage(offscreenMouth, 0, 0, finalBw, finalBh, 0, 0, stretchedW, stretchedH);
    
    // Create soft elliptical mask
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = stretchedW;
    maskCanvas.height = stretchedH;
    const mCtx = maskCanvas.getContext('2d');
    
    const rx = Math.max(1, Math.round(stretchedW / 2) - 2);
    const ry = Math.max(1, Math.round(stretchedH / 2) - 2);
    
    mCtx.fillStyle = '#000';
    mCtx.fillRect(0, 0, stretchedW, stretchedH);
    
    // Draw blurred white ellipse
    const blurSize = Math.max(5, Math.round(width * 0.1));
    mCtx.filter = `blur(${blurSize}px)`;
    mCtx.fillStyle = '#fff';
    mCtx.beginPath();
    mCtx.ellipse(Math.round(stretchedW / 2), Math.round(stretchedH / 2), rx - blurSize, ry - blurSize, 0, 0, 2 * Math.PI);
    mCtx.fill();
    
    // Blend stretched mouth with mask using destination-in
    strCtx.globalCompositeOperation = 'destination-in';
    strCtx.drawImage(maskCanvas, 0, 0);
    
    // Paint blended mouth back onto main canvas
    let tx = center.x - Math.round(stretchedW / 2);
    let ty = center.y - Math.round(stretchedH / 2);
    
    // Clamp
    tx = Math.max(5, Math.min(w - stretchedW - 5, tx));
    ty = Math.max(5, Math.min(h - stretchedH - 5, ty));
    
    ctx.drawImage(stretchedCanvas, tx, ty);
}

// Toggle Debug View
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

// Start camera capture and media loops
async function init() {
    console.log("init() called. Requesting webcam access...");
    try {
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });
        } catch (constraintError) {
            console.warn("High-res front camera constraints failed, trying generic video stream:", constraintError);
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        }
        console.log("Webcam stream access granted.");
        videoElement.srcObject = stream;
        
        console.log("Setting up MediaPipe Camera utility...");
        // Start MediaPipe camera utility
        const camera = new Camera(videoElement, {
            onFrame: async () => {
                mpCtx.drawImage(videoElement, 0, 0, 384, 216);
                await holistic.send({ image: mpCanvas });
                await updateAndDraw();
            },
            width: 1280,
            height: 720
        });
        console.log("Starting Camera stream utility...");
        camera.start();
        console.log("Camera utility start() called.");
    } catch (err) {
        console.error("Webcam access failed:", err);
        alert("Could not access your webcam. Please check browser camera permissions.");
    }
}

// Launch
init();
