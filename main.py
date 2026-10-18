"""Run loose-coupled Kalman filter car path-following demo."""

import matplotlib.pyplot as plt

from car_lckf_sim import run_simulation
from car_visualization import animate_simulation


def main():
    path_x, path_y, history, car = run_simulation(duration_s=6.0)
    _, ani = animate_simulation(path_x, path_y, history, car)
    plt.show()


if __name__ == "__main__":
    main()
