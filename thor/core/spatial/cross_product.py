"""
Spatial cross products for motion and force vectors.

Reference: Featherstone (2008), Eq. 2.31-2.32.
"""

import numpy as np
from numpy.typing import NDArray

from .rotation import skew

SpatialVector = NDArray[np.float64]
SpatialMatrix = NDArray[np.float64]


def spatial_cross_motion(v: SpatialVector) -> SpatialMatrix:
    """Spatial cross product for motion: [v]_x.

    [v]_x = [[omega_x, 0], [v_lin_x, omega_x]]
    """
    omega_x = skew(v[:3])
    v_lin_x = skew(v[3:])
    X = np.zeros((6, 6))
    X[:3, :3] = omega_x
    X[3:, :3] = v_lin_x
    X[3:, 3:] = omega_x
    return X


def spatial_cross_force(v: SpatialVector) -> SpatialMatrix:
    """Spatial cross product for force: [v]_x* = -[v]_x^T."""
    return -spatial_cross_motion(v).T
