"""Top-down car + LCKF visualization."""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle

import lckf_config as cfg
from car_world import OBSTACLES


def _car_body_xy(x, y, psi):
    hw, hl = 0.9, 1.8
    corners = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw],
    ])
    c, s = math.cos(psi), math.sin(psi)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + [x, y]


def _covariance_ellipse(ax, mean_x, mean_y, p_xy, n_std=2.0, **kwargs):
    p = 0.5 * (p_xy + p_xy.T)
    vals, vecs = np.linalg.eigh(p)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse((mean_x, mean_y), width, height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ell)
    return ell


def animate_simulation(path_x, path_y, history, car, interval_ms=50):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(-5, 55)
    ax.set_ylim(-20, 25)

    for x0, y0, x1, y1 in OBSTACLES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k", lw=1.5, zorder=1))

    ax.plot(path_x, path_y, "r.", markersize=4, label="Reference path", zorder=2)
    imu_line, = ax.plot([], [], color="#e67e22", linewidth=1.2, label="IMU (dead reckoning)", zorder=3)
    pred_line, = ax.plot([], [], color="#17a2b8", linestyle="--", linewidth=1.0, label="KF predict (before GPS)", zorder=4)
    fused_line, = ax.plot([], [], color="#27ae60", linewidth=2.0, label="KF fused output", zorder=5)
    gps_scatter = ax.scatter([], [], s=14, c="b", label="GPS measurements", zorder=6)

    car_body = Polygon(
        _car_body_xy(0, 0, 0),
        closed=True,
        facecolor="0.15",
        edgecolor="k",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(car_body)
    car_center, = ax.plot([], [], "o", color="#27ae60", markersize=5, zorder=11)
    speed_text = ax.text(0, 0, "", fontsize=9, zorder=12)
    title = ax.set_title("Time = 0.00[s]")
    ax.legend(loc="upper right", fontsize=8)

    frames = len(history["t"])
    step = max(1, frames // 120)
    ellipse_artist = None

    def update(frame_idx):
        nonlocal ellipse_artist
        i = min(frame_idx * step, frames - 1)
        t = history["t"][i]
        title.set_text(f"Time = {t:.2f}[s]")

        sl = slice(0, i + 1)
        imu_line.set_data(history["imu_x"][sl], history["imu_y"][sl])
        pred_line.set_data(history["pred_x"][sl], history["pred_y"][sl])
        fused_line.set_data(history["fused_x"][sl], history["fused_y"][sl])

        gps_step = max(1, int(cfg.FS_IMU / cfg.FS_GPS))
        n_fix = min(len(history["gps_fixes"]), i // gps_step + 1)
        if n_fix > 0:
            fx, fy = zip(*history["gps_fixes"][:n_fix])
            gps_scatter.set_offsets(np.column_stack([fx, fy]))

        if ellipse_artist is not None:
            ellipse_artist.remove()
            ellipse_artist = None
        ellipse_artist = _covariance_ellipse(
            ax,
            history["fused_x"][i],
            history["fused_y"][i],
            history["P_xy"][i],
            edgecolor="r",
            linewidth=1.2,
            alpha=0.9,
            zorder=7,
        )

        tx = history["true_x"][i]
        ty = history["true_y"][i]
        psi = history["true_psi"][i]
        car_body.set_xy(_car_body_xy(tx, ty, psi))
        car_center.set_data([tx], [ty])
        speed_text.set_position((tx + 1.0, ty - 1.2))
        speed_text.set_text(f"Speed: {car.speed_kmh:.1f}[km/h]")

        return imu_line, pred_line, fused_line, gps_scatter, car_body, car_center, speed_text, title

    import matplotlib.animation as animation

    ani = animation.FuncAnimation(fig, update, frames=120, interval=interval_ms, blit=False, repeat=True)
    plt.tight_layout()
    return fig, ani
