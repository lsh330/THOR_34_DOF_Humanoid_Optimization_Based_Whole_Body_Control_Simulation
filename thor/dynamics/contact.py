"""
Contact dynamics engine for the THOR humanoid.

Implements a compliant contact model (Spring-Damper / Kelvin-Voigt)
that is continuously differentiable and Numba-compatible.

Normal force:  f_n = k_n * max(0, -phi) + d_n * max(0, -dphi/dt)
Tangential:    f_t = -mu * f_n * tanh(v_t / v_stiction)

This smooth approximation avoids the discontinuities of rigid
contact (LCP) while capturing the essential contact physics.

Contact points: 4 corners per foot × 2 feet = 8 points.
Each point contributes a 3D force (normal + 2 tangential).

Reference:
    Marhefka, D.W. & Orin, D.E. (1999). "A Compliant Contact Model
    with Nonlinear Damping for Simulation of Robotic Systems."
    IEEE Trans. Systems, Man, Cybernetics, 29(6), 566-572.
"""

import math

import numpy as np
from numpy.typing import NDArray

from ..core.constants import MU_DEFAULT


# Contact model parameters
CONTACT_STIFFNESS: float = 10000.0    # Normal stiffness [N/m]
CONTACT_DAMPING: float = 500.0        # Normal damping [N*s/m]
MAX_CONTACT_FORCE: float = 5000.0     # Max force per point [N]
STICTION_VEL: float = 0.01            # Stiction velocity threshold [m/s]
GROUND_Z: float = 0.0                 # Ground plane height [m]


def compute_foot_contact_points(
    foot_pos: NDArray,
    foot_rot: NDArray,
    foot_length: float = 0.22,
    foot_width: float = 0.10,
) -> NDArray:
    """Compute 4 corner contact points of a rectangular foot.

    Args:
        foot_pos: (3,) foot center position in world frame
        foot_rot: (3,3) foot orientation in world frame
        foot_length: Foot length along x [m]
        foot_width: Foot width along y [m]

    Returns:
        points: (4, 3) corner positions in world frame
    """
    half_l = foot_length / 2.0
    half_w = foot_width / 2.0

    # Local corner positions (foot frame)
    corners_local = np.array([
        [half_l, half_w, 0.0],
        [half_l, -half_w, 0.0],
        [-half_l, half_w, 0.0],
        [-half_l, -half_w, 0.0],
    ])

    # Transform to world frame
    points = np.empty((4, 3))
    for i in range(4):
        points[i] = foot_pos + foot_rot @ corners_local[i]

    return points


def contact_force_single(
    pos: NDArray,
    vel: NDArray,
    mu: float = MU_DEFAULT,
    k_n: float = CONTACT_STIFFNESS,
    d_n: float = CONTACT_DAMPING,
    v_s: float = STICTION_VEL,
    ground_z: float = GROUND_Z,
) -> NDArray:
    """Compute contact force for a single point.

    Args:
        pos: (3,) contact point position in world frame
        vel: (3,) contact point velocity in world frame
        mu: Friction coefficient
        k_n, d_n: Normal stiffness and damping
        v_s: Stiction velocity threshold
        ground_z: Ground plane height

    Returns:
        f: (3,) contact force in world frame [fx, fy, fz]
    """
    f = np.zeros(3)

    # Penetration depth (negative = penetrating)
    phi = pos[2] - ground_z
    dphi = vel[2]

    if phi >= 0.0:
        return f  # No contact

    # Normal force (Spring-Damper) with clamping
    f_n = k_n * (-phi) + d_n * max(0.0, -dphi)
    f_n = max(0.0, min(f_n, MAX_CONTACT_FORCE))  # Clamp to prevent explosion
    f[2] = f_n

    # Tangential force (continuous Coulomb friction)
    v_t = math.sqrt(vel[0]**2 + vel[1]**2)
    if v_t > 1e-10 and f_n > 0.0:
        friction_mag = mu * f_n * math.tanh(v_t / v_s)
        f[0] = -friction_mag * vel[0] / v_t
        f[1] = -friction_mag * vel[1] / v_t

    return f


def compute_all_contact_forces(
    foot_positions: list[NDArray],
    foot_rotations: list[NDArray],
    foot_velocities: list[NDArray],
    foot_ang_velocities: list[NDArray],
    mu: float = MU_DEFAULT,
) -> tuple[NDArray, NDArray, int]:
    """Compute contact forces for all foot contact points.

    Args:
        foot_positions: List of (3,) foot center positions
        foot_rotations: List of (3,3) foot orientations
        foot_velocities: List of (3,) foot linear velocities
        foot_ang_velocities: List of (3,) foot angular velocities

    Returns:
        forces: (n_contacts, 3) contact forces in world frame
        points: (n_contacts, 3) contact point positions
        n_active: Number of active contacts
    """
    all_forces = []
    all_points = []
    n_active = 0

    for i in range(len(foot_positions)):
        corners = compute_foot_contact_points(
            foot_positions[i], foot_rotations[i])

        for j in range(4):
            # Point velocity = foot_vel + omega × (point - foot_center)
            r = corners[j] - foot_positions[i]
            point_vel = foot_velocities[i] + np.cross(foot_ang_velocities[i], r)

            f = contact_force_single(corners[j], point_vel, mu)
            all_forces.append(f)
            all_points.append(corners[j])

            if f[2] > 0.0:
                n_active += 1

    return np.array(all_forces), np.array(all_points), n_active
