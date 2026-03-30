"""
Spatial inertia construction.

Reference: Featherstone (2008), Eq. 2.63.
"""

import numpy as np
from numpy.typing import NDArray

from .rotation import skew, Vec3

SpatialMatrix = NDArray[np.float64]


def spatial_inertia(mass: float, com: Vec3, I_cm: NDArray) -> SpatialMatrix:
    """6x6 spatial inertia from mass, CoM, and rotational inertia.

    I_s = [[I_cm + m*[c]_x*[c]_x^T, m*[c]_x],
           [m*[c]_x^T,               m*I_3 ]]
    """
    cx = skew(com)
    I_s = np.zeros((6, 6))
    I_s[:3, :3] = I_cm + mass * (cx @ cx.T)
    I_s[:3, 3:] = mass * cx
    I_s[3:, :3] = mass * cx.T
    I_s[3:, 3:] = mass * np.eye(3)
    return I_s
