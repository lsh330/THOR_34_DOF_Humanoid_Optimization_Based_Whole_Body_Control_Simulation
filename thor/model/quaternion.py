"""
Quaternion utilities for base orientation.

Convention: [w, x, y, z] (scalar-first, Hamilton convention).

Reference: Diebel, J. (2006). "Representing Attitude: Euler Angles,
Unit Quaternions, and Rotation Vectors." Stanford.
"""

import numpy as np
from numpy.typing import NDArray


def quat_to_rot(quat: NDArray) -> NDArray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_integrate(quat: NDArray, omega: NDArray, dt: float) -> NDArray:
    """Integrate quaternion with angular velocity omega over dt.

    Uses first-order quaternion derivative:
    dq/dt = 0.5 * [0, omega] * q (quaternion multiplication)
    """
    w, x, y, z = quat
    dquat = 0.5 * dt * np.array([
        -omega[0]*x - omega[1]*y - omega[2]*z,
        omega[0]*w + omega[2]*y - omega[1]*z,
        omega[1]*w - omega[2]*x + omega[0]*z,
        omega[2]*w + omega[1]*x - omega[0]*y,
    ])
    q_new = quat + dquat
    norm = np.linalg.norm(q_new)
    if norm > 1e-10:
        q_new /= norm
    return q_new


def quat_identity() -> NDArray:
    """Return identity quaternion [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0])
