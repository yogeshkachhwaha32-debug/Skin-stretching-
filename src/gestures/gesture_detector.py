# src/gestures/gesture_detector.py
"""
Gesture Detection Engine. Specialized for multi-face and multi-hand Face & Hand Stretch.
Tracks coordinates, handles persistent identity tracking, and manages simultaneous pinch locks.
"""
import math
from src.utils.helpers import calculate_distance
from src import config

EYE_LEFT_OUTER = 33
EYE_RIGHT_OUTER = 263

class GestureDetector:
    def __init__(self):
        # Persistent ID mapping
        self.next_hand_id = 0
        self.prev_hand_centers = {}  # hand_id -> (cx, cy)
        
        self.next_face_id = 0
        self.prev_face_centers = {}  # face_id -> (cx, cy)
        
        # Debounce frame counter for temporary tracking or pinch loss
        self.pinch_inactive_frames = {}  # hand_id -> frame_count
        
        # Locks indexed by pinching hand's persistent ID:
        # hand_id -> {
        #    "anchor_type": "face" or "hand",
        #    "anchor_id": face_id or other_hand_id,
        #    "landmark_idx": int,
        #    "initial_dist": float,
        #    "last_pinch_pos": (x, y)
        # }
        self.active_locks = {}

    def detect_pinch(self, hand, hand_id=None):
        """
        Detect if index tip and thumb tip are pinched together.
        Uses a scale-invariant threshold with hysteresis.
        """
        landmarks = hand["landmarks"]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pixel_dist = calculate_distance(thumb_tip, index_tip)
        
        # Normalize pinch distance against hand size (wrist to middle knuckle)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        hand_size = calculate_distance(wrist, middle_mcp)
        if hand_size == 0:
            hand_size = 1.0
            
        normalized_pinch = pixel_dist / hand_size
        
        # Hysteresis: if hand is already pinching, use a much wider threshold (0.35)
        # to prevent accidental drops during dragging. Otherwise, require a threshold of 0.22 to start.
        threshold = 0.35 if (hand_id is not None and hand_id in self.active_locks) else 0.22
        return normalized_pinch < threshold

    def track_entities(self, hands, faces_coords_list):
        """
        Track hands and faces frame-to-frame to assign persistent IDs.
        Returns tracked_hands dict {hand_id: hand_data} and tracked_faces dict {face_id: coords}.
        """
        # 1. Track Hands
        tracked_hands = {}
        new_hand_centers = {}
        for hand in hands:
            wrist = hand["landmarks"][0]
            mcp = hand["landmarks"][9]
            cx = (wrist[0] + mcp[0]) // 2
            cy = (wrist[1] + mcp[1]) // 2
            
            # Find best match
            matched_id = None
            min_dist = 999999.0
            for hid, prev_c in self.prev_hand_centers.items():
                d = math.hypot(cx - prev_c[0], cy - prev_c[1])
                if d < min_dist and d < 180:
                    min_dist = d
                    matched_id = hid
                    
            if matched_id is not None and matched_id not in tracked_hands:
                hand_id = matched_id
            else:
                hand_id = self.next_hand_id
                self.next_hand_id += 1
                
            tracked_hands[hand_id] = hand
            new_hand_centers[hand_id] = (cx, cy)
        self.prev_hand_centers = new_hand_centers

        # 2. Track Faces
        tracked_faces = {}
        new_face_centers = {}
        for coords in faces_coords_list:
            if not coords or len(coords) < 10:
                continue
            # Use nose tip (landmark 4) as center estimation
            nose = coords[4]
            cx, cy = nose[0], nose[1]
            
            matched_id = None
            min_dist = 999999.0
            for fid, prev_c in self.prev_face_centers.items():
                d = math.hypot(cx - prev_c[0], cy - prev_c[1])
                if d < min_dist and d < 200:
                    min_dist = d
                    matched_id = fid
                    
            if matched_id is not None and matched_id not in tracked_faces:
                face_id = matched_id
            else:
                face_id = self.next_face_id
                self.next_face_id += 1
                
            tracked_faces[face_id] = coords
            new_face_centers[face_id] = (cx, cy)
        self.prev_face_centers = new_face_centers

        return tracked_hands, tracked_faces

    def update_gestures(self, hands, faces_coords_list):
        """
        Process trackers output to determine face and hand stretch gesture states.
        Returns:
        - active_stretches: list of active stretch dicts containing real-time warp metrics.
        - tracked_faces: persistent dictionary of face coordinates.
        - tracked_hands: persistent dictionary of hand coordinates.
        """
        tracked_hands, tracked_faces = self.track_entities(hands, faces_coords_list)
        active_stretches = []
        
        # Debounce hand tracking loss
        current_hand_ids = set(tracked_hands.keys())
        for hid in list(self.active_locks.keys()):
            if hid not in current_hand_ids:
                self.pinch_inactive_frames[hid] = self.pinch_inactive_frames.get(hid, 0) + 1
                if self.pinch_inactive_frames[hid] > 10:
                    del self.active_locks[hid]
                    if hid in self.pinch_inactive_frames:
                        del self.pinch_inactive_frames[hid]

        # Process all hand IDs currently tracked or holding active debounced locks
        all_processing_hids = set(tracked_hands.keys()).union(self.active_locks.keys())

        for hand_id in all_processing_hids:
            hand = tracked_hands.get(hand_id)
            
            # Detect pinch or fallback to debounced active lock
            is_pinching = False
            if hand is not None:
                is_pinching = self.detect_pinch(hand, hand_id)
                
            if is_pinching:
                self.pinch_inactive_frames[hand_id] = 0
            else:
                self.pinch_inactive_frames[hand_id] = self.pinch_inactive_frames.get(hand_id, 0) + 1
                
            # Allow active warp to stay locked for up to 10 frames of tracking or pinch loss
            is_pinch_active = is_pinching or (hand_id in self.active_locks and self.pinch_inactive_frames[hand_id] <= 10)
            
            if is_pinch_active:
                if hand is not None:
                    thumb_tip = hand["landmarks"][4]
                    index_tip = hand["landmarks"][8]
                    current_pinch_pos = (
                        (thumb_tip[0] + index_tip[0]) // 2,
                        (thumb_tip[1] + index_tip[1]) // 2
                    )
                else:
                    current_pinch_pos = self.active_locks[hand_id].get("last_pinch_pos")
                    
                if current_pinch_pos is None:
                    continue
                    
                # Check if we already have a lock
                lock = self.active_locks.get(hand_id)
                
                if lock is None and hand is not None:
                    # Search for closest landmark among all faces and other hands
                    best_type = None
                    best_id = None
                    best_lm_idx = -1
                    min_dist = 999999.0
                    
                    # 1. Search face landmarks
                    for fid, coords in tracked_faces.items():
                        eye_dist = 100.0
                        if len(coords) >= 264:
                            p_left = coords[33]
                            p_right = coords[263]
                            eye_dist = calculate_distance(p_left, p_right)
                        # Only lock if pinch is directly on the skin/face.
                        # Threshold is 35% of the eye corner distance, bounded between 25 and 55 pixels.
                        face_threshold = max(25.0, min(55.0, eye_dist * 0.35))
                        
                        for idx, pt in enumerate(coords):
                            if idx >= 468:  # limit to standard face mesh
                                break
                            d = calculate_distance(current_pinch_pos, pt)
                            if d < face_threshold and d < min_dist:
                                min_dist = d
                                best_type = "face"
                                best_id = fid
                                best_lm_idx = idx
                                
                    # 2. Search hand landmarks (finger/hand stretching)
                    for hid, hand_info in tracked_hands.items():
                        # A hand cannot pinch and stretch itself
                        if hid == hand_id:
                            continue
                        wrist = hand_info["landmarks"][0]
                        middle_mcp = hand_info["landmarks"][9]
                        hand_size = calculate_distance(wrist, middle_mcp)
                        # Threshold is 40% of the hand size, bounded between 25 and 50 pixels.
                        hand_threshold = max(25.0, min(50.0, hand_size * 0.40))
                        
                        for idx, pt in enumerate(hand_info["landmarks"]):
                            d = calculate_distance(current_pinch_pos, pt)
                            if d < hand_threshold and d < min_dist:
                                min_dist = d
                                best_type = "hand"
                                best_id = hid
                                best_lm_idx = idx
                                    
                    # Lock if within the dynamic distance threshold
                    if best_lm_idx != -1:
                        init_anchor = None
                        if best_type == "face":
                            init_anchor = tracked_faces[best_id][best_lm_idx]
                        elif best_type == "hand":
                            init_anchor = tracked_hands[best_id]["landmarks"][best_lm_idx]
                            
                        lock = {
                            "anchor_type": best_type,
                            "anchor_id": best_id,
                            "landmark_idx": best_lm_idx,
                            "initial_pinch_pos": current_pinch_pos,
                            "initial_anchor_pos": init_anchor,
                            "initial_dist": min_dist,
                            "last_pinch_pos": current_pinch_pos
                        }
                        self.active_locks[hand_id] = lock
                        
                # If lock exists, verify the target still exists and compute scale
                if lock is not None:
                    lock["last_pinch_pos"] = current_pinch_pos
                    target_exists = False
                    anchor_pos = None
                    nominal_len = 50.0
                    target_coords = None
                    
                    if lock["anchor_type"] == "face":
                        if lock["anchor_id"] in tracked_faces:
                            target_coords = tracked_faces[lock["anchor_id"]]
                            anchor_pos = target_coords[lock["landmark_idx"]]
                            target_exists = True
                            
                            # Standard eye distance for scale
                            p_eye_left = target_coords[EYE_LEFT_OUTER]
                            p_eye_right = target_coords[EYE_RIGHT_OUTER]
                            eye_dist = calculate_distance(p_eye_left, p_eye_right)
                            nominal_len = (eye_dist if eye_dist > 0 else 80.0) * 0.15
                    else: # Hand
                        if lock["anchor_id"] in tracked_hands:
                            target_coords = tracked_hands[lock["anchor_id"]]["landmarks"]
                            anchor_pos = target_coords[lock["landmark_idx"]]
                            target_exists = True
                            
                            # Estimated hand size
                            wrist = target_coords[0]
                            middle_mcp = target_coords[9]
                            nominal_len = calculate_distance(wrist, middle_mcp) * 0.25
                            
                    if target_exists and anchor_pos is not None:
                        # Calculate drag vector relative to moving face/anchor
                        init_p = lock["initial_pinch_pos"]
                        init_a = lock.get("initial_anchor_pos", anchor_pos)
                        
                        init_offset_x = init_p[0] - init_a[0]
                        init_offset_y = init_p[1] - init_a[1]
                        
                        curr_offset_x = current_pinch_pos[0] - anchor_pos[0]
                        curr_offset_y = current_pinch_pos[1] - anchor_pos[1]
                        
                        drag_dx = curr_offset_x - init_offset_x
                        drag_dy = curr_offset_y - init_offset_y
                        drag_dist = math.hypot(drag_dx, drag_dy)
                        
                        if nominal_len <= 0:
                            nominal_len = 10.0
                            
                        # Calculate stretch scale based on drag distance
                        scale = 1.0 + drag_dist / nominal_len
                        
                        active_stretches.append({
                            "hand_id": hand_id,
                            "anchor_type": lock["anchor_type"],
                            "anchor_id": lock["anchor_id"],
                            "landmark_idx": lock["landmark_idx"],
                            "anchor_pos": anchor_pos,
                            "pinch_pos": current_pinch_pos,
                            "scale": scale,
                            "is_pinching": True,
                            "target_coords": target_coords,
                            "nominal_len": nominal_len,
                            "drag_vector": (drag_dx, drag_dy),
                            "drag_dist": drag_dist,
                            "initial_pinch_pos": init_p
                        })
                    else:
                        # Target no longer present, release lock
                        del self.active_locks[hand_id]
            else:
                # Release lock when pinch ends
                if hand_id in self.active_locks:
                    del self.active_locks[hand_id]
                    
        return active_stretches, tracked_faces, tracked_hands

    def reset(self):
        self.active_locks.clear()
        self.prev_hand_centers.clear()
        self.prev_face_centers.clear()
        self.pinch_inactive_frames.clear()
        self.next_hand_id = 0
        self.next_face_id = 0

