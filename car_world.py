"""Reference path, obstacles, and bicycle-model vehicle."""

import math
import numpy as np

OBSTACLES = [
    (2, 5, 8, 12),
    (12, -14, 20, -6),
    (22, -14, 30, -6),
    (18, 2, 28, 10),
    (32, 8, 40, 18),
    (38, -12, 48, -4),
]


def reference_path(num=220):
    xs = np.linspace(0, 52, num)
    ys = 14 * np.sin(0.11 * xs) - 6 * np.cos(0.07 * xs)
    return xs, ys


def nearest_path_index(x, y, path_x, path_y):
    d2 = (path_x - x) ** 2 + (path_y - y) ** 2
    return int(np.argmin(d2))


def pure_pursuit(x, y, psi, path_x, path_y, lookahead=6.0, wheelbase=2.5):
    idx = nearest_path_index(x, y, path_x, path_y)
    goal = idx
    while goal < len(path_x) - 1:
        dx = path_x[goal] - x
        dy = path_y[goal] - y
        if math.hypot(dx, dy) >= lookahead:
            break
        goal += 1
    dx = path_x[goal] - x
    dy = path_y[goal] - y
    alpha = math.atan2(dy, dx) - psi
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    return math.atan2(2 * wheelbase * math.sin(alpha), lookahead)


class Car:
    def __init__(self, speed_kmh=20.0):
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.2
        self.v = speed_kmh / 3.6
        self.wheelbase = 2.5
        self.speed_kmh = speed_kmh

    def step(self, delta, dt):
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt
        self.psi += (self.v / self.wheelbase) * math.tan(delta) * dt

    def accel_body(self, delta, dt):
        """IMU acceleration in world frame (for LCKF predict step)."""
        yaw_rate = (self.v / self.wheelbase) * math.tan(delta)
        ax = -self.v * yaw_rate * math.sin(self.psi)
        ay = self.v * yaw_rate * math.cos(self.psi)
        return ax, ay, 0.0
