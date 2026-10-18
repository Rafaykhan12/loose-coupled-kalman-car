"""Render simulation preview GIF for README (run once after code changes)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from car_lckf_sim import run_simulation
from car_visualization import animate_simulation


def main():
    out = Path(__file__).resolve().parent / "docs" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    path_x, path_y, history, car = run_simulation(duration_s=6.0)
    fig, ani = animate_simulation(path_x, path_y, history, car, interval_ms=50)
    ani.save(out, writer="pillow", fps=18, dpi=90)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
