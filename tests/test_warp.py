# tests/test_warp.py
"""
Unit tests for the WarpEngine (mesh transformations and fallback mouth warping).
"""
import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rendering.warp_engine import WarpEngine

def test_warp_np_directional_noop():
    """Verify that warp_np_directional handles empty or invalid values gracefully."""
    engine = WarpEngine()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # None parameters should act as no-op and return original frame
    out = engine.warp_np_directional(frame, None, None, None, 1.0, 1.0)
    assert np.array_equal(out, frame)

def test_warp_np_directional_valid():
    """Verify warp_np_directional works correctly with standard mock landmarks."""
    engine = WarpEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Create face landmarks: nose (4) at (100, 100)
    coords = [(i, i) for i in range(300)]
    coords[4] = (100, 100)
    coords[33] = (80, 80)
    coords[263] = (120, 80)
    
    pinch_pos = (110, 110)
    
    # Test directional face warping
    out = engine.warp_np_directional(frame, coords, 4, pinch_pos, 1.2, 1.2, is_face=True)
    assert out.shape == frame.shape

def test_warp_np_mouth_fallback():
    """Verify that warp_np_mouth_fallback executes correctly and clamps coordinates."""
    engine = WarpEngine()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 80), (120, 120), (255, 255, 255), -1)
    
    mouth_metrics = {
        "center": (100, 100),
        "width": 30.0,
        "height": 10.0
    }
    
    # Standard mouth fallback warp
    out = engine.warp_np_mouth_fallback(frame, mouth_metrics, 1.5, 1.2)
    assert out.shape == frame.shape
    
    # Test boundary condition where mouth metrics width is very small
    small_metrics = {
        "center": (10, 10),
        "width": 2.0,
        "height": 1.0
    }
    out_small = engine.warp_np_mouth_fallback(frame, small_metrics, 2.0, 2.0)
    assert out_small.shape == frame.shape
