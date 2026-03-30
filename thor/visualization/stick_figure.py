"""
3D stick-figure visualization of the THOR humanoid.

Generates animated GIFs showing the robot's physical structure
and motion during simulation.
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from ..model.robot_model import RobotModel
from ..model.kinematics import forward_kinematics, body_position


# Link rendering groups (for coloring)
TORSO_IDS = [0, 1, 2]
HEAD_IDS = [3, 4]
L_ARM_IDS = list(range(5, 12))
R_ARM_IDS = list(range(12, 19))
L_LEG_IDS = list(range(19, 25))
R_LEG_IDS = list(range(25, 31))


def _get_body_positions(q: np.ndarray, model: RobotModel) -> np.ndarray:
    """Get world positions of all bodies. Returns (n_bodies, 3)."""
    X_world, _ = forward_kinematics(q, model)
    positions = np.empty((model.n_bodies, 3))
    for i in range(model.n_bodies):
        positions[i] = body_position(X_world[i])
    return positions


def _draw_robot_2d(ax, positions: np.ndarray, model: RobotModel,
                   view: str = "front"):
    """Draw 2D stick figure projection of the robot."""
    if view == "front":
        xi, yi = 1, 2  # y-z plane
        xlabel, ylabel = "y [m]", "z [m]"
    else:  # side
        xi, yi = 0, 2  # x-z plane
        xlabel, ylabel = "x [m]", "z [m]"

    # Draw links (parent-child connections)
    groups = [
        (TORSO_IDS, "k", 4),
        (HEAD_IDS, "gray", 2),
        (L_ARM_IDS, "C0", 2.5),
        (R_ARM_IDS, "C1", 2.5),
        (L_LEG_IDS, "C2", 3.5),
        (R_LEG_IDS, "C3", 3.5),
    ]

    for body_ids, color, lw in groups:
        for bid in body_ids:
            pid = model.parent[bid]
            if pid >= 0:
                ax.plot([positions[pid, xi], positions[bid, xi]],
                        [positions[pid, yi], positions[bid, yi]],
                        color=color, linewidth=lw, solid_capstyle="round")

    # Draw joints
    for i in range(model.n_bodies):
        size = 4 if i in L_LEG_IDS + R_LEG_IDS else 3
        ax.plot(positions[i, xi], positions[i, yi], "ko", markersize=size, zorder=5)

    # Pelvis marker
    ax.plot(positions[0, xi], positions[0, yi], "rs", markersize=8, zorder=6)

    # Ground line
    ax.axhline(0, color="brown", linewidth=2, linestyle="-", alpha=0.5)
    ax.fill_between([-2, 2], [-0.05, -0.05], [0, 0], color="brown", alpha=0.1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def generate_standing_gif(
    result: dict,
    model: RobotModel,
    output_dir: str = "output/gifs",
    fps: int = 20,
    dpi: int = 100,
) -> str:
    """Generate animated GIF of the THOR robot standing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    q_traj = result.get("q", None)
    time_arr = result["time"]

    if q_traj is None:
        # Use com trajectory to infer — generate static frame
        from ..simulation.standing import default_standing_config
        q0 = default_standing_config(model)
        q_traj = np.tile(q0, (len(time_arr), 1))

    dt_frame = 1.0 / fps
    dt_sim = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 0.002
    skip = max(1, int(dt_frame / dt_sim))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), dpi=dpi)
    frames = []

    com_trail = deque(maxlen=100)

    for idx in range(0, len(time_arr), skip):
        ax1.cla()
        ax2.cla()

        q = q_traj[idx]
        positions = _get_body_positions(q, model)

        # Front view
        _draw_robot_2d(ax1, positions, model, view="front")
        ax1.set_title("Front View (Y-Z)")
        ax1.set_xlim(-0.8, 0.8)
        ax1.set_ylim(-0.2, 2.0)

        # Side view
        _draw_robot_2d(ax2, positions, model, view="side")
        ax2.set_title("Side View (X-Z)")
        ax2.set_xlim(-0.8, 0.8)
        ax2.set_ylim(-0.2, 2.0)

        # Info
        t = time_arr[idx]
        com = result["com"][idx] if "com" in result else positions[0]
        fz = result.get("contact_fz", np.zeros(len(time_arr)))[idx]

        fig.suptitle(
            f"THOR 34-DOF Humanoid | t={t:.2f}s | "
            f"CoM z={com[2]:.3f}m | Fz={fz:.0f}N",
            fontsize=12, fontweight="bold")

        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w, h),
                              fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(img)

    plt.close(fig)

    path = f"{output_dir}/thor_standing.gif"
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                        duration=int(1000 / fps), loop=0, optimize=False)
    return path


def generate_static_figure(
    model: RobotModel,
    output_dir: str = "output/plots",
    dpi: int = 150,
) -> str:
    """Generate a static figure showing the THOR robot structure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from ..simulation.standing import default_standing_config
    q0 = default_standing_config(model)
    positions = _get_body_positions(q0, model)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=dpi)

    _draw_robot_2d(ax1, positions, model, view="front")
    ax1.set_title("THOR 34-DOF — Front View (Y-Z Plane)", fontsize=13)
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-0.2, 2.0)

    # Add joint labels for key joints
    labels = {0: "Pelvis", 2: "Chest", 4: "Head",
              21: "L Hip", 22: "L Knee", 24: "L Ankle",
              27: "R Hip", 28: "R Knee", 30: "R Ankle"}
    for bid, name in labels.items():
        if bid < len(positions):
            ax1.annotate(name, (positions[bid, 1], positions[bid, 2]),
                         fontsize=7, ha="left", va="bottom",
                         xytext=(5, 5), textcoords="offset points")

    _draw_robot_2d(ax2, positions, model, view="side")
    ax2.set_title("THOR 34-DOF — Side View (X-Z Plane)", fontsize=13)
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.2, 2.0)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="k", lw=4, label="Torso"),
        Line2D([0], [0], color="C0", lw=2.5, label="Left Arm"),
        Line2D([0], [0], color="C1", lw=2.5, label="Right Arm"),
        Line2D([0], [0], color="C2", lw=3.5, label="Left Leg"),
        Line2D([0], [0], color="C3", lw=3.5, label="Right Leg"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle("THOR 34-DOF Humanoid Robot — Kinematic Structure\n"
                 "35 Bodies | 40 DOF | 67.2 kg | 1.78 m",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = f"{output_dir}/thor_structure.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
