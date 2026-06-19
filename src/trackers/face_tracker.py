# src/trackers/face_tracker.py
"""
Face landmarks tracking engine using MediaPipe Face Mesh.
"""
import cv2
import mediapipe as mp
from src import config

class FaceTracker:
    def __init__(self, max_num_faces=config.FACE_MESH_MAX_FACES):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,  # Disable iris refinement for faster performance
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.latest_landmarks = None
        
    def process_frame(self, rgb_frame, target_size=None):
        """
        Process the RGB frame to find face landmarks.
        Returns a list of face landmarks list (each landmark converted to pixel coordinates).
        """
        results = self.face_mesh.process(rgb_frame)
        self.latest_landmarks = []
        
        if results.multi_face_landmarks:
            w, h = target_size if target_size else (rgb_frame.shape[1], rgb_frame.shape[0])
            for face_landmarks in results.multi_face_landmarks:
                face_coords = []
                for lm in face_landmarks.landmark:
                    # Convert normalized coords to target pixel coords
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face_coords.append((cx, cy))
                self.latest_landmarks.append(face_coords)
                
        return self.latest_landmarks
        
    def draw_landmarks(self, frame, face_coords):
        """Draw face mesh landmarks on a frame (for debugging)."""
        # We can implement a custom drawing of landmarks since we have raw pixel coords
        for point in face_coords:
            cv2.circle(frame, point, 1, config.COLOR_GREEN, -1)
            
    def close(self):
        self.face_mesh.close()
