# src/recording/recorder.py
"""
Recording engine using OpenCV cv2.VideoWriter to capture and export gameplay clips.
"""
import os
import cv2
import numpy as np
from src import config

class VideoRecorder:
    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.output_path = ""
        
    def start_recording(self, filename="anime_power_output.mp4"):
        """Initialize VideoWriter."""
        if self.is_recording:
            self.stop_recording()
            
        # Ensure output directory exists (assets/videos or root)
        os.makedirs("assets/recordings", exist_ok=True)
        self.output_path = os.path.join("assets/recordings", filename)
        
        # Define codec and create VideoWriter object
        # MP4V is a widely compatible codec on Windows
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            30.0,  # Save at 30 FPS
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        
        if not self.writer.isOpened():
            print("Warning: Could not initialize VideoWriter codec.")
            self.writer = None
            return False
            
        self.is_recording = True
        print(f"Recording started. Output will be saved to: {self.output_path}")
        return True
        
    def record_frame(self, bgr_frame):
        """Write a BGR OpenCV frame to the video file."""
        if not self.is_recording or self.writer is None:
            return
        self.writer.write(bgr_frame)
        
    def stop_recording(self):
        """Release VideoWriter and finalize file."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Recording stopped and saved to: {self.output_path}")
            
    def get_status(self):
        return self.is_recording

