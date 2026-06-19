# src/trackers/camera.py
"""
Webcam frame acquisition engine using OpenCV.
Features background thread capture to eliminate blocking IO lag.
"""
import cv2
import time
import threading
from src import config

class CameraEngine:
    def __init__(self, camera_index=None, width=None, height=None):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width if width is not None else config.SCREEN_WIDTH
        self.height = height if height is not None else config.SCREEN_HEIGHT
        
        self.cap = None
        self.fps = 0.0
        self.prev_time = time.time()
        self.is_running = False
        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.thread = None
        
    def start(self):
        """Initialize and open the webcam."""
        if self.cap is not None:
            self.stop()
            
        # Try DirectShow first on Windows as it is much more stable and fast to initialize.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        
        # We also want to support fallback to other indices (0, 1, 2) in case the default index config is wrong
        indices_to_try = [self.camera_index]
        for fallback_idx in [0, 1, 2]:
            if fallback_idx not in indices_to_try:
                indices_to_try.append(fallback_idx)
                
        for idx in indices_to_try:
            for backend in backends:
                try:
                    if backend is not None:
                        self.cap = cv2.VideoCapture(idx, backend)
                    else:
                        self.cap = cv2.VideoCapture(idx)
                        
                    if self.cap is not None and self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set to 1 as fallback/hint
                        
                        # Verify we can actually read a frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.camera_index = idx
                            # Flip frame horizontally to act like a mirror initially
                            self.latest_frame = cv2.flip(frame, 1)
                            self.is_running = True
                            self.prev_time = time.time()
                            
                            # Start thread to continuously grab frames
                            self.thread = threading.Thread(target=self._update_frame, daemon=True)
                            self.thread.start()
                            
                            print(f"Successfully opened camera at index {self.camera_index} with backend {backend if backend is not None else 'default'}")
                            return True
                        else:
                            self.cap.release()
                            self.cap = None
                except Exception:
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                        
        print("Error: Could not open webcam on any index or backend.")
        return False
        
    def _update_frame(self):
        """Continuously grab frames from cap in background."""
        last_read_time = time.time()
        while self.is_running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Flip frame horizontally to act like a mirror
                    flipped = cv2.flip(frame, 1)
                    with self.lock:
                        self.latest_frame = flipped
                        
                    # Calculate camera capture FPS
                    current_time = time.time()
                    time_diff = current_time - last_read_time
                    if time_diff > 0:
                        current_fps = 1.0 / time_diff
                        # Smooth FPS using Exponential Moving Average
                        with self.lock:
                            self.fps = 0.9 * self.fps + 0.1 * current_fps
                    last_read_time = current_time
                else:
                    time.sleep(0.005)  # Rest slightly on failure
            else:
                time.sleep(0.01)
                
    def get_frame(self):
        """Get the latest cached frame. Non-blocking."""
        if not self.is_running:
            return False, None
            
        with self.lock:
            frame = self.latest_frame
            
        if frame is None:
            return False, None
            
        return True, frame
        
    def stop(self):
        """Release the camera resources."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def get_fps(self):
        """Get the current running FPS."""
        with self.lock:
            return self.fps
