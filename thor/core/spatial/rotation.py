"""
Rotation matrices and skew-symmetric utilities.

Optimized for speed: uses pre-allocated arrays and avoids
np.array() constructor in hot paths.

Reference: Featherstone (2008), Appendix A.
"""

import math

import numpy as np
from numpy.typing import NDArray


Vec3 = NDArray[np.float64]
RotMatrix = NDArray[np.float64]


def skew(v: Vec3) -> NDArray:
    """Skew-symmetric matrix from 3-vector: [v]_x.

    Optimized: direct element assignment instead of np.array constructor.
    """
    S = np.empty((3, 3))
    S[0, 0] = 0.0;    S[0, 1] = -v[2]; S[0, 2] = v[1]
    S[1, 0] = v[2];   S[1, 1] = 0.0;   S[1, 2] = -v[0]
    S[2, 0] = -v[1];  S[2, 1] = v[0];  S[2, 2] = 0.0
    return S


def skew_vec(S: NDArray) -> Vec3:
    """Extract 3-vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot_x(theta: float) -> RotMatrix:
    """Rotation matrix about x-axis. Uses math.cos/sin (faster than np)."""
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.empty((3, 3))
    R[0, 0] = 1.0; R[0, 1] = 0.0; R[0, 2] = 0.0
    R[1, 0] = 0.0; R[1, 1] = c;   R[1, 2] = -s
    R[2, 0] = 0.0; R[2, 1] = s;   R[2, 2] = c
    return R


def rot_y(theta: float) -> RotMatrix:
    """Rotation matrix about y-axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.empty((3, 3))
    R[0, 0] = c;   R[0, 1] = 0.0; R[0, 2] = s
    R[1, 0] = 0.0; R[1, 1] = 1.0; R[1, 2] = 0.0
    R[2, 0] = -s;  R[2, 1] = 0.0; R[2, 2] = c
    return R


def rot_z(theta: float) -> RotMatrix:
    """Rotation matrix about z-axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.empty((3, 3))
    R[0, 0] = c;   R[0, 1] = -s;  R[0, 2] = 0.0
    R[1, 0] = s;   R[1, 1] = c;   R[1, 2] = 0.0
    R[2, 0] = 0.0; R[2, 1] = 0.0; R[2, 2] = 1.0
    return R
