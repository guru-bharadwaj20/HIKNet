import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import DATA_DIR, FS, load_dataset

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
PRE_TRIGGER = 50
AXES = ["x", "y", "z"]
COLORS = ["tab:blue", "tab:orange", "tab:olive"]


def time_axis(n):
    return (np.arange(n) - PRE_TRIGGER) / FS


def plot_traces(ax, t, xyz, ylabel):
    for trace, name, color in zip(xyz, AXES, COLORS):
        ax.plot(t, trace, color=color, label=name)
    ax.plot(t, np.linalg.norm(xyz, axis=0), color="k", label="mag")
    ax.set_xlim(-0.01, 0.1)
    ax.set_ylabel(ylabel)


def plot_time_domain(X, y, impact_i, false_i, out):
    impacts = np.flatnonzero(y == 1)
    falses = np.flatnonzero(y == 0)
    picks = [("True impact", impacts[impact_i]), ("False positive", falses[false_i])]
    t = time_axis(X.shape[2])

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, (title, idx) in enumerate(picks):
        plot_traces(axs[0, col], t, X[idx, 3:], "rad/s")
        plot_traces(axs[1, col], t, X[idx, :3], "g")
        axs[0, col].set_title(f"{title} (sample {idx})")
        axs[1, col].set_xlabel("Time (s)")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--impact", type=int, default=4)
    parser.add_argument("--false", type=int, default=15)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    FIG_DIR.mkdir(exist_ok=True)
    plot_time_domain(X, y, args.impact, args.false, FIG_DIR / "time_domain.png")


if __name__ == "__main__":
    main()
