# src/physics/physics_engine.py
"""
Physics Engine simulating spring-mass systems, gravity, velocity, and damping.
Used for elastic stretching animations.
"""

class Spring1D:
    def __init__(self, rest_pos=0.0, stiffness=0.15, damping=0.05, mass=1.0):
        self.rest_pos = rest_pos
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass
        
        self.pos = rest_pos
        self.vel = 0.0
        
    def update(self, time_step=1.0):
        """Update spring state using Hooke's Law."""
        displacement = self.pos - self.rest_pos
        spring_force = -self.stiffness * displacement
        damping_force = -self.damping * self.vel
        
        force = spring_force + damping_force
        accel = force / self.mass
        
        self.vel += accel * time_step
        self.pos += self.vel * time_step
        return self.pos

    def set_target(self, target):
        """Directly offset the spring position to stretch it."""
        self.pos = target

    def set_rest(self, rest_pos):
        self.rest_pos = rest_pos


class Spring2D:
    def __init__(self, rest_x=0.0, rest_y=0.0, stiffness=0.15, damping=0.05, mass=1.0):
        self.spring_x = Spring1D(rest_x, stiffness, damping, mass)
        self.spring_y = Spring1D(rest_y, stiffness, damping, mass)
        
    def update(self, time_step=1.0):
        x = self.spring_x.update(time_step)
        y = self.spring_y.update(time_step)
        return x, y
        
    def set_target(self, x, y):
        self.spring_x.set_target(x)
        self.spring_y.set_target(y)
        
    def set_rest(self, x, y):
        self.spring_x.set_rest(x)
        self.spring_y.set_rest(y)
        
    def get_pos(self):
        return self.spring_x.pos, self.spring_y.pos
