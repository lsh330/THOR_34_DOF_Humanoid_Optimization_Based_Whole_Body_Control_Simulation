"""
Rotation matrices and skew-symmetric utilities.

Reference: Featherstone (2008), Appendix A.
"""

import numpy as np
from numpy.typing import NDArray


Vec3 = NDArray[np.float64]
RotMatrix = NDArray[np.float64]


def skew(v: Vec3) -> NDArray:
    """Skew-symmetric matrix from 3-vector: [v]_x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def skew_vec(S: NDArray) -> Vec3:
    """Extract 3-vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot_x(theta: float) -> RotMatrix:
    """Rotation matrix about x-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta: float) -> RotMatrix:
    """Rotation matrix about y-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta: float) -> RotMatrix:
    """Rotation matrix about z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
