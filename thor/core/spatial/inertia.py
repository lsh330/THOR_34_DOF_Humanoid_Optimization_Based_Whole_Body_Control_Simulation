"""
Spatial inertia construction. Optimized with np.empty.

Reference: Featherstone (2008), Eq. 2.63.
"""

import numpy as np
from numpy.typing import NDArray

from .rotation import skew, Vec3

SpatialMatrix = NDArray[np.float64]


def spatial_inertia(mass: float, com: Vec3, I_cm: NDArray) -> SpatialMatrix:
    """6x6 spatial inertia from mass, CoM, and rotational inertia."""
    cx = skew(com)
    cxT = cx.T
    I_s = np.empty((6, 6))
    I_s[:3, :3] = I_cm + mass * (cx @ cxT)
    I_s[:3, 3:] = mass * cx
    I_s[3:, :3] = mass * cxT
    I_s[3:, 3:] = np.empty((3, 3))
    I_s[3, 3] = mass; I_s[3, 4] = 0.0; I_s[3, 5] = 0.0
    I_s[4, 3] = 0.0; I_s[4, 4] = mass; I_s[4, 5] = 0.0
    I_s[5, 3] = 0.0; I_s[5, 4] = 0.0; I_s[5, 5] = mass
    return I_s
