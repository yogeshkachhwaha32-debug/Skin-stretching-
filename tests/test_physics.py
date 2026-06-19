# tests/test_physics.py
"""
Unit tests for physics spring engine.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.physics_engine import Spring1D, Spring2D

def test_spring_1d_rest():
    """Verify that a spring at rest doesn't move."""
    spring = Spring1D(rest_pos=5.0, stiffness=0.1, damping=0.05)
    # At start, pos is equal to rest position
    assert spring.pos == 5.0
    # Update shouldn't change position if at rest
    for _ in range(10):
        spring.update()
    assert abs(spring.pos - 5.0) < 1e-5

def test_spring_1d_stretching():
    """Verify that a stretched spring oscillates back towards rest position."""
    spring = Spring1D(rest_pos=1.0, stiffness=0.2, damping=0.1)
    spring.set_target(3.0)  # stretch to 3.0
    assert spring.pos == 3.0
    
    # Update should pull the spring back towards 1.0
    spring.update()
    assert spring.pos < 3.0  # Moving back
    
    # Run a few updates, check convergence
    for _ in range(100):
        spring.update()
    # Should converge close to rest pos of 1.0
    assert abs(spring.pos - 1.0) < 0.05

def test_spring_2d():
    """Verify 2D spring coordinates update correctly."""
    spring = Spring2D(rest_x=0.0, rest_y=0.0, stiffness=0.1, damping=0.05)
    spring.set_target(10.0, -10.0)
    assert spring.get_pos() == (10.0, -10.0)
    
    spring.update()
    pos = spring.get_pos()
    assert pos[0] < 10.0
    assert pos[1] > -10.0
