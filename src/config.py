# config.py
"""
Configuration settings for the Anime Power Simulator.
"""

# Screen & Camera Configuration
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
TARGET_FPS = 60
CAMERA_INDEX = 0

# Trackers Settings
FACE_MESH_MAX_FACES = 1
HAND_MAX_NUM_HANDS = 2

# Thresholds for Gesture Recognition
# Mouth stretch: ratio of current mouth width vs. nominal mouth width
MOUTH_STRETCH_THRESHOLD = 1.35
# Mouth open: height of mouth opening
MOUTH_OPEN_THRESHOLD = 25  # in pixels (scaled)

# Distance threshold (normalized) between index tip and thumb tip for Pinch gesture
PINCH_THRESHOLD = 0.05
# Palm open threshold based on finger extension
FINGER_EXTENDED_THRESHOLD = 0.5



# Colors (RGB)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 50, 50)
COLOR_GREEN = (50, 255, 50)
COLOR_BLUE = (50, 150, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_GOLD = (255, 215, 0)
COLOR_ORANGE = (255, 120, 0)
COLOR_PURPLE = (160, 32, 240)
COLOR_AURORA = (100, 255, 200)
