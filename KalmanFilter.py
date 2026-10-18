"""
Loose-coupled Kalman filter (GPS + IMU fusion).

Adapted from: https://github.com/Bee1997/LooseCoupled-KalmanFilter (MIT License)
"""

import numpy as np


class KalmanFilter:
    """6-state LCKF: [x, vx, y, vy, z, vz] with GPS position updates."""

    def __init__(self, dt, process_noise_cov, measurement_noise_cov, state_cov, initial_state_vector):
        self.dt = dt
        self.state_dim = 6
        self.state = np.array(initial_state_vector, dtype=float).reshape((6, 1))

        self.F = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        self.P = state_cov
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        self.use_imu_acceleration = False

    def predict(self, imu_sample):
        if self.use_imu_acceleration:
            ax, ay, az = imu_sample
            dt = self.dt
            self.state[1, 0] += ax * dt
            self.state[3, 0] += ay * dt
            self.state[5, 0] += az * dt
            self.state[0, 0] += self.state[1, 0] * dt
            self.state[2, 0] += self.state[3, 0] * dt
            self.state[4, 0] += self.state[5, 0] * dt
        else:
            self.state = self.F @ self.state

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, gps_position):
        gps_x, gps_y, gps_z = gps_position
        z = np.array([[gps_x], [gps_y], [gps_z]], dtype=float)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.state = self.state + k @ (z - self.H @ self.state)
        i = np.eye(self.state_dim)
        self.P = (i - k @ self.H) @ self.P @ (i - k @ self.H).T + k @ self.R @ k.T
        return self.state

    def position_xy(self):
        return float(self.state[0, 0]), float(self.state[2, 0])

    def covariance_xy(self):
        return self.P[[0, 2], :][:, [0, 2]]


class ImuDeadReckoning:
    """IMU-only integrator (no GPS) — shows drift between GPS fixes."""

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def predict(self, ax, ay, dt):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
