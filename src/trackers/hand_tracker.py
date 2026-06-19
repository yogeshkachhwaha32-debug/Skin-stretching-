# src/trackers/hand_tracker.py
"""
Hand tracking engine using MediaPipe Hands.
Detects up to 2 hands and converts coordinates to screen-space pixel locations.
"""
import cv2
import mediapipe as mp
from src import config

class HandTracker:
    def __init__(self, max_num_hands=config.HAND_MAX_NUM_HANDS):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=0,  # 0 for lightweight model (much faster on CPU)
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.latest_hands = []
        
    def process_frame(self, rgb_frame, target_size=None):
        """
        Process the RGB frame to find hand landmarks.
        Returns a list of dictionaries, each containing:
        - "label": "Left" or "Right" (from perspective of user)
        - "landmarks": list of 21 pixel coordinate tuples (x, y)
        - "raw_landmarks": the raw MediaPipe normalized landmarks
        """
        results = self.hands.process(rgb_frame)
        self.latest_hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            w, h = target_size if target_size else (rgb_frame.shape[1], rgb_frame.shape[0])
            # Zip hands and handedness
            # Note: MediaPipe hand labeling is inverted due to mirroring
            # "Left" in MediaPipe handedness represents user's actual right hand in mirrored view.
            # Let's extract the classification label.
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_coords.append((cx, cy))
                    
                self.latest_hands.append({
                    "label": label,
                    "landmarks": hand_coords,
                    "raw_landmarks": hand_landmarks.landmark
                })
                
        return self.latest_hands
        
    def draw_landmarks(self, frame, hand_coords):
        """Draw hand landmarks and connections (for debugging)."""
        # We can implement a custom drawing of connections for lightweight rendering
        mp_drawing = mp.solutions.drawing_utils
        # Reconstruct hand landmark object for drawing if needed,
        # or simply draw circles on raw coordinates.
        for point in hand_coords:
            cv2.circle(frame, point, 4, config.COLOR_CYAN, -1)
            
    def close(self):
        self.hands.close()
