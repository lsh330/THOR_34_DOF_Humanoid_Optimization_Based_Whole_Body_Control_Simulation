"""Enhanced walking and standing animation generator for THOR humanoid.

Extends stick_figure.py with:
  - Contact force arrows (proportional to Fz)
  - CoM trajectory trail (ghost markers)
  - Contact phase overlay label
  - Camera tracking (follows robot forward position)
  - Information overlay (t, CoM_z, Fz, phase)

Generates animated GIF via PIL frame accumulation (same approach as
stick_figure.py) to avoid FuncAnimation backend dependencies.
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from ..model.robot_model import RobotModel
from ..model.kinematics import forward_kinematics, body_position
from .stick_figure import _get_body_positions, _draw_robot_2d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _phase_label(n_contacts: int) -> tuple[str, str]:
    """Return (label, colour) for the current contact phase."""
    if n_contacts >= 2:
        return 'Double Support', '#2563EB'
    elif n_contacts == 1:
        return 'Single Support', '#D97706'
    else:
        return 'Flight', '#DC2626'


def _draw_contact_arrows(ax, positions, model, fz: float,
                         view: str = 'front', max_arrow_len: float = 0.25):
    """Draw vertical force arrows at foot positions.

    Arrow length is proportional to Fz (clipped to max_arrow_len in figure units).

    Args:
        ax: Matplotlib axes.
        positions: (n_bodies, 3) world positions.
        model: RobotModel instance.
        fz: Scalar vertical contact force [N].
        view: 'front' or 'side'.
        max_arrow_len: Maximum arrow length in metres.
    """
    if fz <= 0.0:
        return

    foot_ids = list(model.foot_link_ids)
    fz_per_foot = fz / max(len(foot_ids), 1)
    arrow_len = min(fz_per_foot / 2000.0, 1.0) * max_arrow_len  # normalise

    xi, yi = (1, 2) if view == 'front' else (0, 2)

    for fid in foot_ids:
        if fid < 0 or fid >= len(positions):
            continue
        px = positions[fid, xi]
        pz = positions[fid, yi]
        ax.annotate(
            '',
            xy=(px, pz + arrow_len),
            xytext=(px, pz),
            arrowprops=dict(
                arrowstyle='->', color='#059669',
                lw=2.0, mutation_scale=12,
            ),
        )
        ax.text(px + 0.04, pz + arrow_len / 2,
                f'{fz_per_foot:.0f}N', fontsize=7, color='#059669', va='center')


def _draw_com_trail(ax, com_trail: deque, view: str = 'front',
                    alpha_min: float = 0.1):
    """Draw CoM trajectory ghost markers fading from old to recent.

    Args:
        ax: Matplotlib axes.
        com_trail: deque of (3,) CoM positions (oldest first).
        view: 'front' or 'side'.
        alpha_min: Minimum alpha for the oldest marker.
    """
    if len(com_trail) < 2:
        return
    trail = list(com_trail)
    n = len(trail)
    xi, yi = (1, 2) if view == 'front' else (0, 2)
    for k, pos in enumerate(trail):
        alpha = min(1.0, alpha_min + (1.0 - alpha_min) * k / (n - 1))
        size  = 3 + 4 * k / (n - 1)
        ax.plot(pos[xi], pos[yi], 'o',
                color='#7C3AED', markersize=size,
                alpha=alpha, zorder=4)


def _add_info_overlay(ax, t: float, com_z: float, fz: float,
                      n_contacts: int) -> None:
    """Add text information box in the upper-left corner of the axes.

    Args:
        ax: Matplotlib axes.
        t: Current time [s].
        com_z: CoM height [m].
        fz: Vertical contact force [N].
        n_contacts: Number of active contact feet.
    """
    phase_lbl, phase_col = _phase_label(n_contacts)
    info = (
        f't = {t:.2f} s\n'
        f'CoM z = {com_z:.3f} m\n'
        f'Fz = {fz:.0f} N\n'
        f'{phase_lbl}'
    )
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75,
                 edgecolor=phase_col, linewidth=1.5)
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', bbox=props, color='black',
            fontfamily='monospace')


def _camera_xlim(x_center: float, view_half: float = 0.7) -> tuple[float, float]:
    """Compute axis limits centred on the robot's lateral/forward position."""
    return x_center - view_half, x_center + view_half


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_walking_gif(
    result: dict,
    model: RobotModel,
    save_path: str = 'output/gifs/thor_walking_enhanced.gif',
    fps: int = 20,
    dpi: int = 100,
    com_trail_len: int = 60,
) -> str:
    """Generate an enhanced walking animation GIF.

    Features beyond stick_figure.generate_standing_gif:
    - Contact force arrows (green) at each foot
    - CoM trajectory trail (purple ghost markers)
    - Contact phase label overlay
    - Camera follows robot forward (x) position

    Args:
        result: dict from run_contact_implicit_simulation.
                Required keys: 'time', 'q', 'com'.
                Optional: 'contact_fz', 'n_contacts'.
        model: RobotModel instance.
        save_path: Output GIF path.
        fps: Frames per second.
        dpi: Render DPI.
        com_trail_len: Number of past CoM positions to show as trail.

    Returns:
        Absolute path of the saved GIF.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    time_arr   = result['time']
    q_traj     = result['q']
    com_arr    = result.get('com', result.get('com_trajectory',
                             np.zeros((len(time_arr), 3))))
    fz_arr     = result.get('contact_fz', np.zeros(len(time_arr)))
    nc_arr     = result.get('n_contacts', np.full(len(time_arr), 2, dtype=int))

    dt_frame   = 1.0 / fps
    dt_sim     = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 0.002
    skip       = max(1, int(dt_frame / dt_sim))

    com_trail_f = deque(maxlen=com_trail_len)
    com_trail_s = deque(maxlen=com_trail_len)

    frames = []
    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)

    for idx in range(0, len(time_arr), skip):
        ax_f.cla()
        ax_s.cla()

        q        = q_traj[idx]
        positions = _get_body_positions(q, model)
        t_cur    = time_arr[idx]
        com_cur  = com_arr[idx]
        fz_cur   = float(fz_arr[idx])
        nc_cur   = int(nc_arr[idx])

        com_trail_f.append(com_cur.copy())
        com_trail_s.append(com_cur.copy())

        # Camera tracking: follow forward position (x)
        x_center_f = positions[0, 1]   # front view: y-axis
        x_center_s = positions[0, 0]   # side view: x-axis
        xlim_f = _camera_xlim(x_center_f, 0.65)
        xlim_s = _camera_xlim(x_center_s, 0.65)

        # --- Front view ---
        _draw_robot_2d(ax_f, positions, model, view='front')
        _draw_com_trail(ax_f, com_trail_f, view='front')
        _draw_contact_arrows(ax_f, positions, model, fz_cur, view='front')
        _add_info_overlay(ax_f, t_cur, float(com_cur[2]), fz_cur, nc_cur)
        ax_f.set_xlim(*xlim_f)
        ax_f.set_ylim(-0.15, 2.0)
        ax_f.set_title('Front View (Y-Z)')

        # --- Side view ---
        _draw_robot_2d(ax_s, positions, model, view='side')
        _draw_com_trail(ax_s, com_trail_s, view='side')
        _draw_contact_arrows(ax_s, positions, model, fz_cur, view='side')
        ax_s.set_xlim(*xlim_s)
        ax_s.set_ylim(-0.15, 2.0)
        ax_s.set_title('Side View (X-Z)')

        # Phase label in title
        phase_lbl, phase_col = _phase_label(nc_cur)
        fig.suptitle(
            f'THOR 34-DOF — Walking | {phase_lbl} | '
            f't={t_cur:.2f} s | CoM z={com_cur[2]:.3f} m',
            fontsize=11, fontweight='bold', color=phase_col,
        )
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes('RGBA', (w, h),
                              fig.canvas.buffer_rgba()).convert('RGB')
        frames.append(img)

    plt.close(fig)

    out = str(Path(save_path).resolve())
    if frames:
        frames[0].save(
            out, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0, optimize=False,
        )
    return out


def generate_standing_gif(
    result: dict,
    model: RobotModel,
    save_path: str = 'output/gifs/thor_standing_enhanced.gif',
    fps: int = 20,
    dpi: int = 100,
    com_trail_len: int = 80,
) -> str:
    """Generate an enhanced standing animation GIF.

    Features beyond stick_figure.generate_standing_gif:
    - Contact force arrows at each foot
    - CoM trajectory trail
    - Phase and numeric info overlay

    Args:
        result: dict from run_contact_implicit_simulation or
                run_standing_simulation.
        model: RobotModel instance.
        save_path: Output GIF path.
        fps: Frames per second.
        dpi: Render DPI.
        com_trail_len: Trail length in frames.

    Returns:
        Absolute path of the saved GIF.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    time_arr = result['time']
    q_traj   = result.get('q', None)
    com_arr  = result.get('com', result.get('com_trajectory',
                           np.zeros((len(time_arr), 3))))
    fz_arr   = result.get('contact_fz', np.zeros(len(time_arr)))
    nc_arr   = result.get('n_contacts', np.full(len(time_arr), 2, dtype=int))

    if q_traj is None:
        from ..simulation.standing import default_standing_config
        q0 = default_standing_config(model)
        q_traj = np.tile(q0, (len(time_arr), 1))

    dt_frame = 1.0 / fps
    dt_sim   = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 0.002
    skip     = max(1, int(dt_frame / dt_sim))

    com_trail_f = deque(maxlen=com_trail_len)
    com_trail_s = deque(maxlen=com_trail_len)

    frames = []
    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(12, 7), dpi=dpi)

    for idx in range(0, len(time_arr), skip):
        ax_f.cla()
        ax_s.cla()

        q         = q_traj[idx]
        positions  = _get_body_positions(q, model)
        t_cur     = time_arr[idx]
        com_cur   = com_arr[idx]
        fz_cur    = float(fz_arr[idx])
        nc_cur    = int(nc_arr[idx])

        com_trail_f.append(com_cur.copy())
        com_trail_s.append(com_cur.copy())

        # --- Front view ---
        _draw_robot_2d(ax_f, positions, model, view='front')
        _draw_com_trail(ax_f, com_trail_f, view='front')
        _draw_contact_arrows(ax_f, positions, model, fz_cur, view='front')
        _add_info_overlay(ax_f, t_cur, float(com_cur[2]), fz_cur, nc_cur)
        ax_f.set_xlim(-0.8, 0.8)
        ax_f.set_ylim(-0.2, 2.0)
        ax_f.set_title('Front View (Y-Z)')

        # --- Side view ---
        _draw_robot_2d(ax_s, positions, model, view='side')
        _draw_com_trail(ax_s, com_trail_s, view='side')
        _draw_contact_arrows(ax_s, positions, model, fz_cur, view='side')
        ax_s.set_xlim(-0.8, 0.8)
        ax_s.set_ylim(-0.2, 2.0)
        ax_s.set_title('Side View (X-Z)')

        fig.suptitle(
            f'THOR 34-DOF — Standing | t={t_cur:.2f} s | '
            f'CoM z={com_cur[2]:.3f} m | Fz={fz_cur:.0f} N',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes('RGBA', (w, h),
                              fig.canvas.buffer_rgba()).convert('RGB')
        frames.append(img)

    plt.close(fig)

    out = str(Path(save_path).resolve())
    if frames:
        frames[0].save(
            out, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0, optimize=False,
        )
    return out
