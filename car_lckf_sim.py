"""Loose-coupled Kalman navigation + car path following."""

import math
import numpy as np

from KalmanFilter import ImuDeadReckoning, KalmanFilter
from car_world import Car, pure_pursuit, reference_path
import lckf_config as cfg


def _gps_noise():
    s = cfg.GPS_MEAS_STD
    return np.random.randn() * s, np.random.randn() * s


def run_simulation(duration_s=6.0):
    dt = 1.0 / cfg.FS_IMU
    gps_interval = max(1, int(cfg.FS_IMU / cfg.FS_GPS))
    path_x, path_y = reference_path()

    car = Car(speed_kmh=20.0)
    v0 = car.v
    vx0 = v0 * math.cos(car.psi)
    vy0 = v0 * math.sin(car.psi)
    kf = KalmanFilter(
        dt=dt,
        process_noise_cov=np.diag([cfg.Q, cfg.Q * 0.2, cfg.Q, cfg.Q * 0.2, cfg.Q, cfg.Q * 0.2]),
        measurement_noise_cov=np.eye(3) * cfg.R,
        state_cov=np.eye(6) * 0.5,
        initial_state_vector=[car.x, vx0, car.y, vy0, 0.0, 0.0],
    )
    kf.use_imu_acceleration = cfg.IMU_USE_ACCELERATION

    imu_only = ImuDeadReckoning(car.x, car.y, vx0, vy0)

    steps = int(duration_s * cfg.FS_IMU)
    history = {
        "t": [],
        "true_x": [],
        "true_y": [],
        "true_psi": [],
        "imu_x": [],
        "imu_y": [],
        "pred_x": [],
        "pred_y": [],
        "fused_x": [],
        "fused_y": [],
        "gps_fixes": [],
        "P_xy": [],
    }

    for step in range(steps):
        t = step * dt

        delta = pure_pursuit(car.x, car.y, car.psi, path_x, path_y)
        car.step(delta, dt)

        ax, ay, az = car.accel_body(delta, dt)
        imu = (ax + np.random.randn() * 0.01, ay + np.random.randn() * 0.01, az)

        imu_only.predict(imu[0], imu[1], dt)
        kf.predict(imu)
        px, py = kf.position_xy()

        if step % gps_interval == 0:
            nx, ny = _gps_noise()
            gps = (car.x + nx, car.y + ny, 0.0)
            kf.update(gps)
            kf.state[1, 0] = imu_only.vx
            kf.state[3, 0] = imu_only.vy
            history["gps_fixes"].append((gps[0], gps[1]))

        fx, fy = kf.position_xy()
        history["t"].append(t)
        history["true_x"].append(car.x)
        history["true_y"].append(car.y)
        history["true_psi"].append(car.psi)
        history["imu_x"].append(imu_only.x)
        history["imu_y"].append(imu_only.y)
        history["pred_x"].append(px)
        history["pred_y"].append(py)
        history["fused_x"].append(fx)
        history["fused_y"].append(fy)
        history["P_xy"].append(kf.covariance_xy().copy())

    return path_x, path_y, history, car
