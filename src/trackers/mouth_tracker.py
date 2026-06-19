# src/trackers/mouth_tracker.py
"""
Mouth tracking engine. Extracts mouth metrics from face mesh coordinates.
Uses eye corner distance to normalize for distance from webcam.
"""
from src.utils.helpers import calculate_distance
from src import config

# MediaPipe Face Mesh Landmark indices
LIP_LEFT_CORNER = 61
LIP_RIGHT_CORNER = 291
LIP_UPPER_OUTER = 0
LIP_LOWER_OUTER = 17

LIP_UPPER_INNER = 13
LIP_LOWER_INNER = 14

EYE_LEFT_OUTER = 33
EYE_RIGHT_OUTER = 263

class MouthTracker:
    def __init__(self):
        self.calibration_frames = 0
        self.calibration_limit = 60  # Calibrate over first 60 frames of rest state
        self.rest_normalized_width = 0.5  # Default fallback
        self.rest_widths = []
        
    def get_mouth_metrics(self, face_coords):
        """
        Extract mouth dimensions from face mesh coordinates.
        Returns a dictionary of metrics:
        - width: pixel width of mouth
        - height: pixel height of mouth (inner lip distance)
        - center: (x, y) coordinates of mouth center
        - stretch_ratio: current normalized width / rest normalized width
        - is_stretching: boolean
        - is_open: boolean
        """
        if not face_coords or len(face_coords) < 300:
            return None
            
        p_left = face_coords[LIP_LEFT_CORNER]
        p_right = face_coords[LIP_RIGHT_CORNER]
        p_upper = face_coords[LIP_UPPER_OUTER]
        p_lower = face_coords[LIP_LOWER_OUTER]
        
        p_inner_upper = face_coords[LIP_UPPER_INNER]
        p_inner_lower = face_coords[LIP_LOWER_INNER]
        
        p_eye_left = face_coords[EYE_LEFT_OUTER]
        p_eye_right = face_coords[EYE_RIGHT_OUTER]
        
        # Calculate raw pixel distances
        width = calculate_distance(p_left, p_right)
        height = calculate_distance(p_inner_upper, p_inner_lower)
        
        # Mouth center point
        center_x = int((p_left[0] + p_right[0]) // 2)
        center_y = int((p_upper[1] + p_lower[1]) // 2)
        center = (center_x, center_y)
        
        # Distance between eye outer corners (used to normalize for depth/distance from camera)
        eye_distance = calculate_distance(p_eye_left, p_eye_right)
        if eye_distance == 0:
            eye_distance = 1.0
            
        normalized_width = width / eye_distance
        
        # Calibration phase: calibrate the user's rest mouth width
        if self.calibration_frames < self.calibration_limit:
            # Assume user is relaxed during the start
            self.rest_widths.append(normalized_width)
            self.calibration_frames += 1
            if self.calibration_frames == self.calibration_limit:
                self.rest_normalized_width = sum(self.rest_widths) / len(self.rest_widths)
                print(f"Mouth Calibration complete. Base rest normalized width: {self.rest_normalized_width:.4f}")
                
        # Calculate stretch ratio
        stretch_ratio = normalized_width / self.rest_normalized_width
        
        # Determine states
        is_stretching = stretch_ratio >= config.MOUTH_STRETCH_THRESHOLD
        is_open = height >= config.MOUTH_OPEN_THRESHOLD
        
        return {
            "width": width,
            "height": height,
            "center": center,
            "stretch_ratio": stretch_ratio,
            "is_stretching": is_stretching,
            "is_open": is_open,
            "left_corner": p_left,
            "right_corner": p_right,
            "upper_lip": p_upper,
            "lower_lip": p_lower,
            "face_coords": face_coords
        }
        
    def reset_calibration(self):
        """Reset mouth rest calibration data."""
        self.calibration_frames = 0
        self.rest_widths = []
