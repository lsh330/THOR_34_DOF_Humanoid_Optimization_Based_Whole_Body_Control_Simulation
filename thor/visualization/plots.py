"""
Publication-quality plots for the THOR humanoid simulation.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_standing_analysis(result: dict, output_dir: str = "output/plots",
                           dpi: int = 150) -> str:
    """Generate standing simulation analysis figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = result["time"]

    # CoM trajectory
    ax = axes[0, 0]
    com = result["com_trajectory"]
    ax.plot(t, com[:, 0], label="x", linewidth=1.5)
    ax.plot(t, com[:, 1], label="y", linewidth=1.5)
    ax.plot(t, com[:, 2], label="z", linewidth=2, color="C2")
    ax.set_ylabel("CoM Position [m]")
    ax.set_title("Center of Mass Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Joint error
    ax = axes[0, 1]
    ax.plot(t, result["joint_error"], color="C3", linewidth=1.5)
    ax.set_ylabel("Joint Error Norm [rad]")
    ax.set_title("Joint Position Error")
    ax.grid(True, alpha=0.3)

    # Torque RMS over time
    ax = axes[1, 0]
    tau = result["torques"]
    tau_rms = np.sqrt(np.mean(tau**2, axis=1))
    ax.plot(t, tau_rms, color="C1", linewidth=1.5)
    ax.set_ylabel("RMS Torque [N·m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Effort")
    ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[1, 1]
    ax.plot(t, result["energy"], color="C4", linewidth=2)
    ax.set_ylabel("Total Energy [J]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Mechanical Energy")
    ax.grid(True, alpha=0.3)

    fig.suptitle("THOR 34-DOF Humanoid — Static Standing Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = f"{output_dir}/standing_analysis.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
