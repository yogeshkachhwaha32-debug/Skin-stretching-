# src/web_server.py
"""
Face Stretch Simulator - Web Server Portal.
Runs a local Flask application at http://localhost:5000/ to stream webcam frames,
apply computer vision elastic deformation overlays, and host a premium dashboard.
"""
import sys
import os
import time
import math
import random
import numpy as np
import cv2
import threading
from flask import Flask, Response, render_template_string, jsonify, request

# Adjust path to import local modules if launched directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.trackers.camera import CameraEngine
from src.trackers.face_tracker import FaceTracker
from src.trackers.hand_tracker import HandTracker
from src.trackers.mouth_tracker import MouthTracker
from src.gestures.gesture_detector import GestureDetector
from src.physics.physics_engine import Spring1D
from src.physics.smoothing import PointListSmoother
from src.recording.recorder import VideoRecorder
from src.utils.helpers import calculate_distance
from src.rendering.warp_engine import WarpEngine

app = Flask(__name__)

# Global instances of engines
camera = None
face_tracker = None
hand_tracker = None
mouth_tracker = None
gesture_detector = None
recorder = None
warp_engine = None

# Global states
show_debug = False
is_recording = False
fps_val = 0.0
stretch_val = 1.00
gesture_status = "None"
no_cam_flag = False

latest_processed_frame = None
latest_frame_lock = threading.Lock()
frame_event = threading.Event()

active_clients = 0
active_clients_lock = threading.Lock()



# # Character effect specific states (recreated for OpenCV equivalents)
luffy_stretches = {}
last_spring_time = time.time()
face_mouth_trackers = {}

def bg_frame_grabber():
    global camera, face_tracker, hand_tracker, mouth_tracker, gesture_detector, warp_engine
    global latest_processed_frame, fps_val, stretch_val, gesture_status, no_cam_flag, show_debug, is_recording, recorder
    global luffy_stretches, last_spring_time, face_mouth_trackers
    
    face_smoothers = {}
    hand_smoothers = {}
    
    while True:
        start_time = time.time()
        
        if not camera.is_running:
            time.sleep(0.2)
            continue
            
        ret, frame = camera.get_frame()
        no_cam_flag = False
        if not ret or frame is None:
            frame = np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)
            no_cam_flag = True
            time.sleep(0.03) # Prevent CPU spinning if camera fails
            
        if not no_cam_flag:
            frame = cv2.resize(frame, (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            # Create a downscaled version (width = 256) for MediaPipe processing to optimize CPU usage and eliminate lag
            h_orig, w_orig, _ = frame.shape
            scale_ratio = 256.0 / w_orig if w_orig > 256 else 1.0
            if scale_ratio < 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
            else:
                small_frame = frame
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        face_coords_list = face_tracker.process_frame(rgb_small, target_size=(config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        hand_data = hand_tracker.process_frame(rgb_small, target_size=(config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        
        # GestureDetector tracks multiple entities
        active_stretches, tracked_faces, tracked_hands = gesture_detector.update_gestures(hand_data, face_coords_list)
        
        # 1. Smooth Face Landmarks
        smoothed_faces = {}
        for fid, coords in tracked_faces.items():
            if fid not in face_smoothers:
                face_smoothers[fid] = PointListSmoother(len(coords), alpha=0.15)
            smoothed_landmarks = face_smoothers[fid].filter(coords)
            smoothed_faces[fid] = [(int(x), int(y)) for x, y in smoothed_landmarks]
        tracked_faces = smoothed_faces
        
        # Clean up unused face smoothers
        for fid in list(face_smoothers.keys()):
            if fid not in tracked_faces:
                del face_smoothers[fid]
                
        # 2. Smooth Hand Landmarks
        smoothed_hands = {}
        for hid, hand in tracked_hands.items():
            if hid not in hand_smoothers:
                hand_smoothers[hid] = PointListSmoother(len(hand["landmarks"]), alpha=0.12)
            smoothed_landmarks = hand_smoothers[hid].filter(hand["landmarks"])
            # Update landmarks in hand dictionary (cast coordinates to int for rendering/indices)
            smoothed_hand = hand.copy()
            smoothed_hand["landmarks"] = [(int(x), int(y)) for x, y in smoothed_landmarks]
            smoothed_hands[hid] = smoothed_hand
        tracked_hands = smoothed_hands
        
        # Clean up unused hand smoothers
        for hid in list(hand_smoothers.keys()):
            if hid not in tracked_hands:
                del hand_smoothers[hid]
        
        # Calculate time step for physics
        current_time = time.time()
        dt = current_time - last_spring_time
        last_spring_time = current_time
        time_step = max(0.1, min(dt * 60.0, 3.0)) # bounding time step for stability
        
        # Reset activity flag
        for s in luffy_stretches.values():
            s["is_active"] = False
            
        # Update or register stretches
        for active in active_stretches:
            hid = active["hand_id"]
            if hid not in luffy_stretches:
                # Bouncy spring with 0.18 stiffness and 0.22 damping tracking drag distance in pixels
                luffy_stretches[hid] = {
                    "spring_dist": Spring1D(rest_pos=0.0, stiffness=0.18, damping=0.22)
                }
            s = luffy_stretches[hid]
            s["is_active"] = True
            s["anchor_type"] = active["anchor_type"]
            s["anchor_id"] = active["anchor_id"]
            s["landmark_idx"] = active["landmark_idx"]
            
            # Set target spring values to actual drag distance
            drag_dist = active["drag_dist"]
            s["spring_dist"].rest_pos = drag_dist
            
            # Save parameters for elastic release wiggle
            drag_dx, drag_dy = active["drag_vector"]
            if drag_dist > 0.1:
                s["last_drag_vector"] = (drag_dx / drag_dist, drag_dy / drag_dist)
            else:
                s["last_drag_vector"] = (1.0, 0.0)
            s["last_nominal_len"] = active["nominal_len"]

        # Run spring updates and apply all active/released warps sequentially
        stretches_to_delete = []
        max_scale = 1.00
        has_active_pinch = False
        
        for hid, s in luffy_stretches.items():
            if not s["is_active"]:
                s["spring_dist"].rest_pos = 0.0
            s["spring_dist"].update(time_step)
            
            if s["is_active"]:
                has_active_pinch = True
                
            warp_dist = s["spring_dist"].pos
            scale_val = 1.0 + warp_dist / s["last_nominal_len"]
            max_scale = max(max_scale, scale_val)
            
            # Delete stretch when snapback is complete (less than 1.0 pixel of warp remains)
            if not s["is_active"] and warp_dist < 1.0:
                stretches_to_delete.append(hid)
                continue
                
            # Retrieve coordinates
            coords = None
            is_face = True
            if s["anchor_type"] == "face":
                if s["anchor_id"] in tracked_faces:
                    coords = tracked_faces[s["anchor_id"]]
            else:
                if s["anchor_id"] in tracked_hands:
                    coords = tracked_hands[s["anchor_id"]]["landmarks"]
                    is_face = False
                    
            # Check anchors and apply OpenCV directional warp
            anchor_idx = s["landmark_idx"]
            pinch_pos = None
            
            if coords is not None and anchor_idx < len(coords):
                anchor_pos = coords[anchor_idx]
                # The warp target is always displaced from the anchor landmark
                # along the relative drag unit vector by the spring distance (warp_dist).
                pinch_pos = (
                    int(anchor_pos[0] + s["last_drag_vector"][0] * warp_dist),
                    int(anchor_pos[1] + s["last_drag_vector"][1] * warp_dist)
                )
                    
                if pinch_pos is not None:
                    frame = warp_engine.warp_np_directional(
                        frame,
                        coords=coords,
                        anchor_idx=anchor_idx,
                        pinch_pos=pinch_pos,
                        scale_x=scale_val,
                        scale_y=scale_val,
                        is_face=is_face
                    )
                
        for hid in stretches_to_delete:
            del luffy_stretches[hid]
            
        # Clean mouth trackers for deleted faces
        for fid in list(face_mouth_trackers.keys()):
            if fid not in tracked_faces:
                del face_mouth_trackers[fid]
                
        # Fallback to mouth stretching if no hand pinch is active
        has_mouth_stretch = False
        if not has_active_pinch and tracked_faces:
            # Find the primary face (the face with the largest eye distance)
            primary_fid = None
            max_eye_dist = -1
            
            for fid, coords in tracked_faces.items():
                if len(coords) >= 264:
                    p_left = coords[33]
                    p_right = coords[263]
                    eye_dist = calculate_distance(p_left, p_right)
                    if eye_dist > max_eye_dist:
                        max_eye_dist = eye_dist
                        primary_fid = fid
            
            # Apply mouth stretch fallback ONLY to the primary face
            if primary_fid is not None:
                coords = tracked_faces[primary_fid]
                if primary_fid not in face_mouth_trackers:
                    face_mouth_trackers[primary_fid] = MouthTracker()
                m_tracker = face_mouth_trackers[primary_fid]
                m_metrics = m_tracker.get_mouth_metrics(coords)
                if m_metrics and (m_metrics["is_stretching"] or m_metrics["is_open"]):
                    h, w, _ = frame.shape
                    left_corner = m_metrics["left_corner"]
                    right_corner = m_metrics["right_corner"]
                    
                    # Ignore mouth stretches when face is cut off by the edge of the frame
                    margin = 25
                    if (margin < left_corner[0] < w - margin and margin < left_corner[1] < h - margin and
                        margin < right_corner[0] < w - margin and margin < right_corner[1] < h - margin):
                        
                        stretch_scale = m_metrics["stretch_ratio"]
                        height_factor = m_metrics["height"] / 20.0
                        target_x = 1.0 + (stretch_scale - 1.0) * 1.8
                        target_y = 1.0 + (height_factor - 1.0) * 1.8
                        
                        frame = warp_engine.warp_np_mouth_fallback(frame, m_metrics, target_x, target_y)
                        max_scale = max(max_scale, stretch_scale)
                        has_mouth_stretch = True

        # Telemetry updates
        fps_val = camera.get_fps()
        stretch_val = max_scale
        if has_active_pinch:
            gesture_status = "PINCH DRAG"
        elif has_mouth_stretch:
            gesture_status = "MOUTH STRETCH"
        else:
            gesture_status = "None"
            
        # Draw telemetry debug wireframes
        if show_debug:
            # Faces
            for fid, coords in tracked_faces.items():
                for pt in coords:
                    cv2.circle(frame, pt, 1, (0, 255, 0), -1)
            # Hands
            for hid, hand in tracked_hands.items():
                landmarks = hand["landmarks"]
                color = (255, 255, 0) if hand["label"] == "Left" else (0, 215, 255)
                for pt in landmarks:
                    cv2.circle(frame, pt, 3, color, -1)
                for chain in [[0,1,2,3,4], [0,5,6,7,8], [5,9,13,17], [0,17,18,19,20]]:
                    for idx in range(len(chain) - 1):
                        cv2.line(frame, landmarks[chain[idx]], landmarks[chain[idx+1]], color, 1)
                        
        # Overlay warning banner if camera missing
        if no_cam_flag:
            cv2.rectangle(frame, (config.SCREEN_WIDTH//2 - 300, config.SCREEN_HEIGHT//2 - 30),
                          (config.SCREEN_WIDTH//2 + 300, config.SCREEN_HEIGHT//2 + 30), (50, 50, 200), -1)
            cv2.putText(frame, "NO WEBCAM FEED DETECTED - CHECK PERMISSIONS", 
                        (config.SCREEN_WIDTH//2 - 260, config.SCREEN_HEIGHT//2 + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Record video
        if is_recording:
            recorder.record_frame(frame)
            
        with latest_frame_lock:
            latest_processed_frame = frame.copy()
        frame_event.set()
        frame_event.clear()
            
        elapsed = time.time() - start_time
        sleep_time = max(0.001, (1.0 / config.TARGET_FPS) - elapsed)
        time.sleep(sleep_time)

def generate_video_stream():
    global latest_processed_frame, latest_frame_lock, active_clients, active_clients_lock, camera
    
    with active_clients_lock:
        active_clients += 1
        if not camera.is_running:
            print("Active client connected! Starting Camera Engine...")
            camera.start()
            
    try:
        while True:
            # Wait for next frame event with a 100ms timeout
            frame_event.wait(timeout=0.1)
            
            frame_to_send = None
            with latest_frame_lock:
                if latest_processed_frame is not None:
                    frame_to_send = latest_processed_frame.copy()
                    
            if frame_to_send is None:
                frame_to_send = np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "INITIALIZING CAMERA FEED...", 
                            (config.SCREEN_WIDTH//2 - 200, config.SCREEN_HEIGHT//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            ret, jpeg = cv2.imencode('.jpg', frame_to_send)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    finally:
        with active_clients_lock:
            active_clients = max(0, active_clients - 1)
            if active_clients == 0:
                print("All clients disconnected. Releasing Camera Engine...")
                camera.stop()

# Glassmorphic HTML web page template
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Experience real-time interactive face stretching and elastic warping animations powered by computer vision. Pinch, drag, and stretch landmarks with high-fidelity spring physics.">
    <title>Skin Stretching Simulation - Web Portal</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧬</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0b12;
            --panel-bg: rgba(18, 18, 30, 0.65);
            --primary: #00d2ff;
            --accent: #ff7800;
            --glow: 0 0 20px rgba(0, 210, 255, 0.35);
        }
        body {
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: #fff;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            display: flex;
        }
        .video-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1;
            overflow: hidden;
            background: #000;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        /* Center Floating Header Panel */
        .header-panel {
            position: fixed;
            top: 25px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            padding: 15px 35px;
            text-align: center;
            pointer-events: none; /* Let clicks pass through */
        }
        h1 {
            font-size: 1.8rem;
            line-height: 1.2;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(45deg, var(--primary), #b000ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: transform 0.3s ease;
        }
        h1:hover {
            transform: scale(1.02);
        }
        p.subtitle {
            color: #888a9e;
            margin: 5px 0 0 0;
            font-size: 0.95rem;
            font-weight: 300;
        }
        /* Footer Taskbar */
        .footer-taskbar {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            height: 60px;
            z-index: 10;
            background: rgba(10, 10, 18, 0.65);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            display: flex;
            align-items: center;
            padding: 0 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            box-sizing: border-box;
        }
        .taskbar-content {
            display: flex;
            align-items: center;
            gap: 15px;
            width: 100%;
            white-space: nowrap;
            overflow-x: auto;
        }
        .taskbar-content::-webkit-scrollbar {
            display: none;
        }
        .taskbar-label {
            font-size: 0.85rem;
            font-weight: 800;
            text-transform: uppercase;
            color: var(--primary);
            letter-spacing: 1.5px;
        }
        .taskbar-item {
            font-size: 0.85rem;
            color: #b0b1c5;
        }
        .taskbar-divider {
            color: rgba(255, 255, 255, 0.15);
            font-weight: 300;
        }
        @media (max-width: 768px) {
            .header-panel {
                width: 90%;
                padding: 10px 20px;
            }
            h1 {
                font-size: 1.4rem;
            }
            p.subtitle {
                font-size: 0.8rem;
            }
            .footer-taskbar {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="video-container">
        <img src="/video_feed" alt="Video Feed">
    </div>
    
    <div class="header-panel">
        <h1>SKIN STRETCHING SIMULATION</h1>
        <p class="subtitle">OBLIQ AI Educational Foundation</p>
    </div>

    <div class="footer-taskbar">
        <div class="taskbar-content">
            <span class="taskbar-label">Instructions:</span>
            <span class="taskbar-item"><strong>1. Mouth Stretch:</strong> Smile wide or open your mouth to stretch your cheeks.</span>
            <span class="taskbar-divider">|</span>
            <span class="taskbar-item"><strong>2. Directional Pinch:</strong> Pinch near forehead, nose, ears, chin, or cheeks and drag to stretch.</span>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index_page():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global is_recording
    if not is_recording:
        filename = time.strftime("recording_%Y%m%d_%H%M%S.mp4")
        success = recorder.start_recording(filename)
        if success:
            is_recording = True
    else:
        recorder.stop_recording()
        is_recording = False
    return jsonify({"status": "success", "recording": is_recording})

@app.route('/toggle_debug', methods=['POST'])
def toggle_debug():
    global show_debug
    show_debug = not show_debug
    return jsonify({"status": "success", "debug": show_debug})

@app.route('/get_telemetry')
def get_telemetry():
    global fps_val, stretch_val, gesture_status, show_debug, is_recording
    return jsonify({
        "fps": fps_val,
        "stretch": stretch_val,
        "gesture": gesture_status,
        "debug": show_debug,
        "recording": is_recording
    })

def run_server():
    global camera, face_tracker, hand_tracker, mouth_tracker, gesture_detector, recorder, warp_engine
    
    print("Initializing Web Server backend services...")
    camera = CameraEngine()
    # Verify camera hardware works, then immediately release it until a client connects
    if camera.start():
        camera.stop()
        print("Camera hardware verified and released successfully. Ready for dynamic activation.")
    else:
        print("Warning: Camera hardware verification failed. Will attempt reconnection on client connection.")
        
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    mouth_tracker = MouthTracker()
    gesture_detector = GestureDetector()
    warp_engine = WarpEngine()
    

    
    recorder = VideoRecorder()
    
    # Start the background grabber thread
    grabber_thread = threading.Thread(target=bg_frame_grabber, daemon=True)
    grabber_thread.start()
    
    print("\n------------------------------------------------------------")
    print("Face Stretch Web Simulator is ready to launch!")
    print("Open this link in your browser to test: http://127.0.0.1:5000/")
    print("------------------------------------------------------------\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    run_server()
