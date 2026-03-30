"""
Spatial coordinate transforms (Plucker).

Reference: Featherstone (2008), Eq. 2.25.
"""

import numpy as np
from numpy.typing import NDArray

from .rotation import skew, RotMatrix, Vec3

SpatialMatrix = NDArray[np.float64]


def spatial_transform(R: RotMatrix, p: Vec3) -> SpatialMatrix:
    """Spatial transform X from rotation R and translation p.

    X = [[R, 0], [-R*[p]_x, R]]
    Maps motion vectors from frame B to frame A.
    """
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -R @ skew(p)
    return X


def spatial_transform_inv(X: SpatialMatrix) -> SpatialMatrix:
    """Inverse of spatial transform."""
    R = X[:3, :3]
    Rt = R.T
    X_inv = np.zeros((6, 6))
    X_inv[:3, :3] = Rt
    X_inv[3:, 3:] = Rt
    X_inv[3:, :3] = -Rt @ X[3:, :3] @ Rt
    return X_inv
