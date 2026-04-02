#!/usr/bin/env python3
"""
Plot the real Silverstone track from the updated Silverstone.csv and
overlay logged car trajectories from the episode CSV files.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("logs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACK_COLUMNS = ["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]


def load_track(track_path):
    """Load the updated Silverstone.csv format with a commented header row."""
    track_df = pd.read_csv(track_path, comment="#", header=None, names=TRACK_COLUMNS)
    track_df = track_df.dropna().reset_index(drop=True)

    required = set(TRACK_COLUMNS)
    missing = required - set(track_df.columns)
    if missing:
        raise ValueError(f"Track file missing columns: {sorted(missing)}")

    return track_df.astype(float)


def load_trajectory(logs_dir, episode):
    """Load one logged trajectory file."""
    traj_path = Path(logs_dir) / f"episode_{episode}_traj.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")

    traj_df = pd.read_csv(traj_path)
    required = {"x", "y"}
    missing = required - set(traj_df.columns)
    if missing:
        raise ValueError(f"Trajectory file missing columns: {sorted(missing)}")

    return traj_df


def compute_track_edges(track_df):
    """Compute left and right edges from centerline and track widths."""
    center = track_df[["x_m", "y_m"]].to_numpy()

    prev_points = np.roll(center, 1, axis=0)
    next_points = np.roll(center, -1, axis=0)
    tangents = next_points - prev_points

    tangent_norms = np.linalg.norm(tangents, axis=1)
    tangent_norms[tangent_norms == 0.0] = 1.0
    tangents = tangents / tangent_norms[:, np.newaxis]

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    right = center + normals * track_df["w_tr_right_m"].to_numpy()[:, np.newaxis]
    left = center - normals * track_df["w_tr_left_m"].to_numpy()[:, np.newaxis]
    return center, left, right


def draw_real_track(ax, track_df):
    """Draw the actual track surface and both edges from Silverstone.csv."""
    _, left_edge, right_edge = compute_track_edges(track_df)

    surface_x = np.concatenate([right_edge[:, 0], left_edge[::-1, 0]])
    surface_y = np.concatenate([right_edge[:, 1], left_edge[::-1, 1]])

    ax.fill(surface_x, surface_y, color="#d1d5db", alpha=0.8, zorder=0)
    ax.plot(left_edge[:, 0], left_edge[:, 1], color="black", linewidth=1.4, zorder=1)
    ax.plot(right_edge[:, 0], right_edge[:, 1], color="black", linewidth=1.4, zorder=1)


def plot_trajectories(track_path, logs_dir, episodes, output_path):
    """Plot the real track and overlay the logged car trajectories."""
    track_df = load_track(track_path)

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_real_track(ax, track_df)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(episodes), 1)))
    for index, episode in enumerate(episodes):
        traj_df = load_trajectory(logs_dir, episode)
        color = colors[index % len(colors)]

        ax.plot(
            traj_df["x"].to_numpy(),
            traj_df["y"].to_numpy(),
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"Episode {episode}",
            zorder=2,
        )
        ax.scatter(traj_df["x"].iloc[0], traj_df["y"].iloc[0], color=color, s=20, zorder=3)
        ax.scatter(traj_df["x"].iloc[-1], traj_df["y"].iloc[-1], color=color, s=70, marker="*", zorder=3)

    ax.set_title("Silverstone Track With Logged Trajectories", fontweight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot updated Silverstone track geometry with logged trajectories"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="Episode numbers to plot",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="Silverstone.csv",
        help="Updated Silverstone track CSV",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing episode trajectory CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path",
    )
    args = parser.parse_args()

    output_path = args.output or f"silverstone_real_track_{'_'.join(map(str, args.episodes))}.png"
    plot_trajectories(args.track, args.logs_dir, args.episodes, output_path)


if __name__ == "__main__":
    main()
