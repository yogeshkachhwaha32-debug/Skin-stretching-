# tests/test_smoothing.py
"""
Unit tests for coordinate smoothing filters.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.smoothing import EMASmoother

def test_ema_smoother_scalar():
    """Verify that EMASmoother filters scalar value jitter."""
    smoother = EMASmoother(alpha=0.5)
    
    # First value should set the initial state
    assert smoother.filter(10.0) == 10.0
    
    # Next value is 20.0. Output should be alpha*20 + (1-alpha)*10 = 0.5*20 + 0.5*10 = 15.0
    assert smoother.filter(20.0) == 15.0
    
    # Third value is 30.0. Output: 0.5*30 + 0.5*15 = 22.5
    assert smoother.filter(30.0) == 22.5

def test_ema_smoother_vector():
    """Verify that EMASmoother works with coordinate tuples (x, y)."""
    smoother = EMASmoother(alpha=0.4)
    
    # First point
    assert smoother.filter((10.0, 50.0)) == (10.0, 50.0)
    
    # Second point: x=20, y=100.
    # Output x: 0.4*20 + 0.6*10 = 14.0
    # Output y: 0.4*100 + 0.6*50 = 70.0
    result = smoother.filter((20.0, 100.0))
    assert abs(result[0] - 14.0) < 1e-5
    assert abs(result[1] - 70.0) < 1e-5
