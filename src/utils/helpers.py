# src/utils/helpers.py
"""
Helper utilities for math, distance calculations, and coordinate conversions.
"""
import math
import numpy as np

def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points (x, y) or (x, y, z).
    """
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)))

def get_angle(p1, p2, p3):
    """
    Calculate angle in degrees at p2, formed by lines p1-p2 and p2-p3.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def map_value(value, left_min, left_max, right_min, right_max):
    """
    Map value from one range to another, clamped to the target range bounds.
    """
    # Avoid division by zero
    if left_max - left_min == 0:
        return right_min
    
    # Calculate percentage
    span = left_max - left_min
    value_scaled = float(value - left_min) / float(span)
    
    # Interpolate
    result = right_min + (value_scaled * (right_max - right_min))
    
    # Clamp
    if right_min < right_max:
        return max(right_min, min(right_max, result))
    else:
        return max(right_max, min(right_min, result))
