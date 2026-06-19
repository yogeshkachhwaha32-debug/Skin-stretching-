# tests/test_gestures.py
"""
Unit tests for gesture recognition checks.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gestures.gesture_detector import GestureDetector
from src import config

def test_detect_pinch():
    """Verify pinch gesture detection using mock landmark coordinates."""
    detector = GestureDetector()
    
    # Mock hand:
    # wrist (0) at (0, 100)
    # middle MCP (9) at (0, 50) -> hand_size = 50
    # thumb tip (4) at (0, 0)
    # index tip (8) at (0, 1) -> pinch distance = 1.
    # normalized_pinch = 1 / 50 = 0.02
    # config.PINCH_THRESHOLD is 0.05
    # 0.02 < 0.05 -> Pinch detected!
    hand_pinch = {
        "landmarks": [
            (0, 100),  # wrist (0)
            None, None, None,
            (0, 0),    # thumb tip (4)
            None, None, None,
            (0, 1),    # index tip (8)
            (0, 50),   # middle MCP (9)
        ]
    }
    assert detector.detect_pinch(hand_pinch) is True

    # Mock hand where fingers are spread out:
    # thumb tip (4) at (0, 0)
    # index tip (8) at (0, 20) -> pinch distance = 20.
    # normalized_pinch = 20 / 50 = 0.4 > 0.05 -> No pinch!
    hand_no_pinch = {
        "landmarks": [
            (0, 100),  # wrist (0)
            None, None, None,
            (0, 0),    # thumb tip (4)
            None, None, None,
            (0, 20),   # index tip (8)
            (0, 50),   # middle MCP (9)
        ]
    }
    assert detector.detect_pinch(hand_no_pinch) is False

def test_update_gestures_and_relative_drag():
    """Verify that update_gestures handles locks, sets initial landmarks, and computes drag vector correctly."""
    detector = GestureDetector()
    
    # 1. First frame: Face at (100, 100). Hand is pinching at (100, 100)
    mock_face = [(i, i) for i in range(300)]
    mock_face[4] = (100, 100) # anchor point (nose tip)
    mock_face[33] = (80, 80)
    mock_face[263] = (120, 80)
    
    mock_hand = {
        "landmarks": [
            (0, 100),  # wrist (0)
            (0, 0), (0, 0), (0, 0),
            (100, 100), # thumb (4)
            (0, 0), (0, 0), (0, 0),
            (100, 101), # index (8) -> pinch at (100, 100)
            (0, 50) # middle MCP (9)
        ]
    }
    
    # Run first frame
    stretches, tracked_faces, tracked_hands = detector.update_gestures([mock_hand], [mock_face])
    
    # Hand ID 0 should be locked on face ID 0, landmark index 4
    assert len(detector.active_locks) == 1
    lock = detector.active_locks[0]
    assert lock["anchor_type"] == "face"
    assert lock["landmark_idx"] == 4
    assert lock["initial_anchor_pos"] == (100, 100)
    
    # 2. Second frame: Face moves to (110, 110). Hand moves to (120, 120).
    # Drag vector relative to screen = (120 - 100, 120 - 100) = (20, 20)
    # Drag vector relative to face offset change:
    # initial offset = (100 - 100, 100 - 100) = (0, 0)
    # current offset = (120 - 110, 120 - 110) = (10, 10)
    # drag vector relative to face = current offset - initial offset = (10, 10)
    mock_face_2 = [(i, i) for i in range(300)]
    mock_face_2[4] = (110, 110) # anchor point moves to 110, 110
    mock_face_2[33] = (90, 90)
    mock_face_2[263] = (130, 90)
    
    mock_hand_2 = {
        "landmarks": [
            (0, 100),
            (0, 0), (0, 0), (0, 0),
            (120, 120), # thumb (4)
            (0, 0), (0, 0), (0, 0),
            (120, 121), # index (8) -> pinch at (120, 120)
            (0, 50)
        ]
    }
    
    stretches_2, _, _ = detector.update_gestures([mock_hand_2], [mock_face_2])
    
    assert len(stretches_2) == 1
    # Check that the drag_vector is (10, 10) rather than (20, 20)
    # which proves it's invariant to head motion!
    assert stretches_2[0]["drag_vector"] == (10.0, 10.0)

def test_hand_landmark_locking():
    """Verify that a pinching hand can lock onto and stretch another hand's landmarks."""
    detector = GestureDetector()
    
    # Hand 1 is pinching at (100, 100)
    hand_pinching = {
        "landmarks": [
            (0, 100),  # wrist (0)
            (0, 0), (0, 0), (0, 0),
            (100, 100), # thumb (4)
            (0, 0), (0, 0), (0, 0),
            (100, 101), # index (8) -> pinch at (100, 100)
            (0, 50) # middle MCP (9)
        ]
    }
    
    # Hand 2 has landmark 5 (index MCP) at (95, 95), which is close to the pinch position
    # Landmark 4 (thumb) and 8 (index tip) are placed far apart so it doesn't pinch itself
    hand_target = {
        "landmarks": [
            (0, 100),  # wrist (0)
            (0, 0), (0, 0), (0, 0),
            (0, 0),    # thumb (4)
            (95, 95),  # index MCP (5)
            (0, 0), (0, 0),
            (50, 50),  # index tip (8)
            (0, 50)    # middle MCP (9)
        ]
    }
    
    # We pass both hands to update_gestures, with NO faces.
    # Hand tracking will identify them as persistent IDs 0 and 1.
    stretches, _, tracked_hands = detector.update_gestures([hand_pinching, hand_target], [])
    
    # There should be an active lock established on hand 1 (which is hand_target's ID)
    assert len(detector.active_locks) == 1
    
    # Find which hand pinched. Let's find the persistent ID of hand_pinching.
    pinching_id = None
    target_id = None
    for hid, h in tracked_hands.items():
        if h["landmarks"][4] == (100, 100):
            pinching_id = hid
        else:
            target_id = hid
            
    assert pinching_id is not None
    assert target_id is not None
    
    lock = detector.active_locks[pinching_id]
    assert lock["anchor_type"] == "hand"
    assert lock["anchor_id"] == target_id
    assert lock["landmark_idx"] == 5
    assert lock["initial_anchor_pos"] == (95, 95)

