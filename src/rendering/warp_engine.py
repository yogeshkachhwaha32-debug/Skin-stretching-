# src/rendering/warp_engine.py
"""
Warp Engine handling mesh transformations and directional elastic image stretching.
Optimized for smooth, high-fidelity real-time face and finger elastic deformation.
"""
import cv2
import math
import numpy as np
from src.utils.helpers import calculate_distance

class WarpEngine:
    def __init__(self):
        pass
        
    def warp_np_directional(self, frame, coords, anchor_idx, pinch_pos, scale_x, scale_y, is_face=True):
        """
        Warp a NumPy BGR image to stretch a face or hand region directionally from an anchor landmark to pinch_pos.
        Uses a mathematically guaranteed monotonic coordinate mapping for highly realistic elastic stretching.
        """
        if coords is None or anchor_idx is None or pinch_pos is None or anchor_idx >= len(coords):
            return frame
            
        anchor_pos = coords[anchor_idx]
        ax, ay = anchor_pos
        px, py = pinch_pos
        
        dx = px - ax
        dy = py - ay
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 3:  # avoid division by zero/extremely tiny warps
            return frame
            
        # Direction vector
        ux = dx / dist
        uy = dy / dist
        
        # Determine warp corridor width R dynamically
        if is_face and len(coords) >= 264:
            p_eye_left = coords[33]
            p_eye_right = coords[263]
            eye_distance = calculate_distance(p_eye_left, p_eye_right)
            if eye_distance == 0:
                eye_distance = 1.0
            # Localized corridor radius and soft falloff transitions
            R = int(eye_distance * 0.85)     # Corridor half-width
            R_in = int(eye_distance * 0.75)  # Falloff behind the anchor
            R_out = int(eye_distance * 0.2)   # Falloff beyond the pinch point
        else:
            # Hand
            if len(coords) >= 10:
                wrist = coords[0]
                middle_mcp = coords[9]
                hand_size = calculate_distance(wrist, middle_mcp)
                if hand_size == 0:
                    hand_size = 1.0
                R = int(hand_size * 0.65)
                R_in = int(hand_size * 0.55)
                R_out = int(hand_size * 0.2)
            else:
                R = 60
                R_in = 50
                R_out = 15
                
        h, w, _ = frame.shape
        margin_x = max(R, R_in) + 10
        margin_y = max(R, R_in) + 10
        x_min = max(0, int(min(ax, px) - margin_x))
        y_min = max(0, int(min(ay, py) - margin_y))
        x_max = min(w, int(max(ax, px) + margin_x))
        y_max = min(h, int(max(ay, py) + margin_y))
        
        if (x_max - x_min) <= 10 or (y_max - y_min) <= 10:
            return frame
            
        # Crop the region of interest (ROI)
        roi = frame[y_min:y_max, x_min:x_max].copy()
        roi_h, roi_w, _ = roi.shape
        
        # Generate grid of coordinates for the remapping
        grid_x, grid_y = np.meshgrid(np.arange(roi_w, dtype=np.float32), np.arange(roi_h, dtype=np.float32))
        
        # Convert grid to global coordinates
        global_x = grid_x + x_min
        global_y = grid_y + y_min
        
        # Project pixels onto the drag vector
        wx = global_x - ax
        wy = global_y - ay
        t = wx * ux + wy * uy
        
        # Perpendicular distance squared to the line of drag
        d2 = wx * wx + wy * wy
        d_perp_sq = d2 - t * t
        
        # Initialize map to original coordinates
        map_x = grid_x.copy()
        map_y = grid_y.copy()
        
        # Falloff calculation perpendicular to the drag line using squared distance
        R_sq = float(R * R)
        mask_influence = d_perp_sq < R_sq
        if np.any(mask_influence):
            g = (1.0 - d_perp_sq[mask_influence] / R_sq) ** 2
            
            # Displacement strength
            strength = 0.98
            k = g * strength
            
            t_sel = t[mask_influence]
            disp = np.zeros_like(t_sel)
            
            # Region 1: Stretching corridor [-R_in, dist]
            mask_stretch = (t_sel >= -R_in) & (t_sel <= dist)
            if np.any(mask_stretch):
                ts = t_sel[mask_stretch]
                disp[mask_stretch] = k[mask_stretch] * (ts + R_in) * (dist / (R_in + dist))
                
            # Region 2: Beyond the pinch point (fade-out)
            mask_beyond = (t_sel > dist) & (t_sel < dist + R_out)
            if np.any(mask_beyond):
                tb = t_sel[mask_beyond]
                disp[mask_beyond] = dist * k[mask_beyond] * ((1.0 - (tb - dist) / R_out) ** 2)
                
            # Shift pixels backwards along the drag direction
            map_x[mask_influence] -= (disp * ux).astype(np.float32)
            map_y[mask_influence] -= (disp * uy).astype(np.float32)
            
        # Perform remapping
        warped_roi = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Copy remapped ROI back to frame
        frame[y_min:y_max, x_min:x_max] = warped_roi
        return frame

    def warp_np_mouth_fallback(self, frame, mouth_metrics, scale_x, scale_y):
        """Perform mouth horizontal and vertical stretch centered on the lips."""
        center = mouth_metrics["center"]
        width = int(mouth_metrics["width"])
        
        bw = int(width * 2.2)
        bh = int(width * 1.1)
        
        bx = max(0, center[0] - bw // 2)
        by = max(0, center[1] - bh // 2)
        bw = min(frame.shape[1] - bx, bw)
        bh = min(frame.shape[0] - by, bh)
        
        if bw <= 15 or bh <= 15:
            return frame
            
        crop = frame[by:by+bh, bx:bx+bw]
        
        # Stretch independently horizontally (scale_x) and vertically (scale_y)
        new_w = int(bw * scale_x)
        new_h = int(bh * scale_y)
        new_w = max(10, new_w)
        new_h = max(10, new_h)
        
        try:
            stretched = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Clamp ellipse axes to be at least 1 to prevent errors on tiny selections
            rx = max(1, new_w // 2 - 2)
            ry = max(1, new_h // 2 - 2)
            
            # Create a single-channel binary mask of same size as stretched
            mask = np.zeros((new_h, new_w), dtype=np.uint8)
            cv2.ellipse(mask, (new_w // 2, new_h // 2), (rx, ry), 0, 0, 360, 255, -1)
            
            # Feather blend boundary using Gaussian blur (kernel size proportional to mouth width)
            blur_size = int(max(width * 0.1, 5))
            if blur_size % 2 == 0:
                blur_size += 1
            mask_blur = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
            # Convert mask to 3-channel normalized float array
            alpha = mask_blur.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            cx = center[0]
            cy = center[1]
            
            # Clamp destination coordinates to prevent going out of bounds
            cx = max(new_w//2 + 5, min(frame.shape[1] - new_w//2 - 5, cx))
            cy = max(new_h//2 + 5, min(frame.shape[0] - new_h//2 - 5, cy))
            
            # Bounding box of target area in frame
            tx_min = cx - new_w // 2
            ty_min = cy - new_h // 2
            tx_max = tx_min + new_w
            ty_max = ty_min + new_h
            
            target_roi = frame[ty_min:ty_max, tx_min:tx_max]
            
            # Perform high-performance linear alpha blend
            blended = alpha * stretched.astype(np.float32) + (1.0 - alpha) * target_roi.astype(np.float32)
            frame[ty_min:ty_max, tx_min:tx_max] = blended.astype(np.uint8)
            
        except Exception as e:
            print("Mouth-only warp error:", e)
            
        return frame
