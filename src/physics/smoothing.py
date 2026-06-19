# src/physics/smoothing.py
"""
Signal smoothing filters (e.g. Exponential Moving Average) to eliminate MediaPipe landmark jitter.
"""

class EMASmoother:
    def __init__(self, alpha=0.3):
        """
        alpha: Smoothing factor (0.0 to 1.0). 
        Lower value means more smoothing but higher latency.
        """
        self.alpha = alpha
        self.value = None
        
    def filter(self, next_value):
        if self.value is None:
            self.value = next_value
        else:
            if isinstance(next_value, (list, tuple)):
                # Handle vectors (x, y) or list of coords
                self.value = tuple(
                    self.alpha * n + (1.0 - self.alpha) * v 
                    for n, v in zip(next_value, self.value)
                )
            else:
                # Handle scalar value
                self.value = self.alpha * next_value + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class PointListSmoother:
    def __init__(self, size, alpha=0.3):
        self.smoothers = [EMASmoother(alpha) for _ in range(size)]
        
    def filter(self, points):
        if not points:
            return points
        if len(points) > len(self.smoothers):
            # Dynamically grow if needed
            diff = len(points) - len(self.smoothers)
            self.smoothers.extend([EMASmoother(self.smoothers[0].alpha) for _ in range(diff)])
            
        return [self.smoothers[i].filter(p) for i, p in enumerate(points)]
        
    def reset(self):
        for s in self.smoothers:
            s.reset()
