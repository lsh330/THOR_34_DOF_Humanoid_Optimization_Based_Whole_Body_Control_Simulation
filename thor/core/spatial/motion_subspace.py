"""
Joint motion subspace vectors.

Reference: Featherstone (2008), Table 4.1.
"""

import numpy as np
from numpy.typing import NDArray

SpatialVector = NDArray[np.float64]


def motion_subspace_revolute(axis: int) -> SpatialVector:
    """Motion subspace S for a revolute joint.

    axis: 0='x', 1='y', 2='z'
    S = [e_axis; 0] (rotation about axis, no translation).
    """
    S = np.zeros(6)
    S[axis] = 1.0
    return S


def motion_subspace_prismatic(axis: int) -> SpatialVector:
    """Motion subspace S for a prismatic joint.

    axis: 0='x', 1='y', 2='z'
    S = [0; e_axis] (translation along axis, no rotation).
    """
    S = np.zeros(6)
    S[3 + axis] = 1.0
    return S
