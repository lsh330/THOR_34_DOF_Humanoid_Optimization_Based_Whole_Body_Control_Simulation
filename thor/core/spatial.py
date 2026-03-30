"""
Spatial algebra for rigid body dynamics.

Implements Featherstone's spatial vector notation:
- Spatial motion vectors (twists): v = [omega; v_linear]
- Spatial force vectors (wrenches): f = [tau; f_linear]
- Spatial transforms: X ∈ R^{6×6}
- Spatial inertia: I_s ∈ R^{6×6}

Convention: Plücker coordinates, motion-type ordering [angular; linear].

Reference:
    Featherstone, R. (2008). "Rigid Body Dynamics Algorithms." Springer.
    Ch. 2: Spatial Vector Algebra.
"""

import numpy as np
import numba as nb
from numpy.typing import NDArray


# Type aliases
SpatialVector = NDArray[np.float64]   # (6,)
SpatialMatrix = NDArray[np.float64]   # (6, 6)
RotMatrix = NDArray[np.float64]       # (3, 3)
Vec3 = NDArray[np.float64]            # (3,)


def skew(v: Vec3) -> NDArray:
    """Skew-symmetric matrix from 3-vector.

    [v]× = [[0, -vz, vy],
             [vz, 0, -vx],
             [-vy, vx, 0]]
    """
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


def spatial_transform(R: RotMatrix, p: Vec3) -> SpatialMatrix:
    """Construct spatial transform X from rotation R and translation p.

    X = [[R, 0], [-R*[p]×, R]]

    Transforms spatial motion vectors from frame B to frame A:
        v_A = X_{AB} * v_B

    Reference: Featherstone (2008), Eq. (2.25).
    """
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -R @ skew(p)
    return X


def spatial_transform_inv(X: SpatialMatrix) -> SpatialMatrix:
    """Inverse of spatial transform: X^{-1}.

    For X = [[R, 0], [-R*[p]×, R]]:
    X^{-1} = [[R^T, 0], [[p]× * R^T, R^T]]

    More efficient than general 6x6 inverse.
    """
    R = X[:3, :3]
    Rt = R.T
    X_inv = np.zeros((6, 6))
    X_inv[:3, :3] = Rt
    X_inv[3:, 3:] = Rt
    # Extract -R*[p]× from X[3:, :3], recover [p]× = -R^T * X[3:,:3]
    skew_p_R = -X[3:, :3]  # = R*[p]×
    X_inv[3:, :3] = (skew_p_R @ Rt).T  # = ([p]×*R^T)^T... need careful derivation
    # Simpler: X_inv[3:,:3] = skew(p) @ R^T where p is recovered
    # Actually: X^{-1} = [[R^T, 0], [skew(p)@R^T, R^T]] where p = -R^T @ skew_vec(...)
    # Use the identity directly
    X_inv[3:, :3] = -Rt @ X[3:, :3] @ Rt
    return X_inv


def spatial_inertia(mass: float, com: Vec3, I_cm: NDArray) -> SpatialMatrix:
    """Construct 6×6 spatial inertia matrix.

    I_s = [[I_cm + m*[c]×*[c]×^T, m*[c]×],
           [m*[c]×^T,               m*I_3]]

    Where c is the center of mass position in the body frame.

    Reference: Featherstone (2008), Eq. (2.63).
    """
    cx = skew(com)
    I_s = np.zeros((6, 6))
    I_s[:3, :3] = I_cm + mass * (cx @ cx.T)
    I_s[:3, 3:] = mass * cx
    I_s[3:, :3] = mass * cx.T
    I_s[3:, 3:] = mass * np.eye(3)
    return I_s


def spatial_cross_motion(v: SpatialVector) -> SpatialMatrix:
    """Spatial cross product for motion vectors: [v]×.

    [v]× = [[omega×, 0    ],
            [v_lin×, omega×]]

    Such that: [v]× * m = v × m  (spatial cross product).
    """
    omega_x = skew(v[:3])
    v_lin_x = skew(v[3:])
    X = np.zeros((6, 6))
    X[:3, :3] = omega_x
    X[3:, :3] = v_lin_x
    X[3:, 3:] = omega_x
    return X


def spatial_cross_force(v: SpatialVector) -> SpatialMatrix:
    """Spatial cross product for force vectors: [v]×*.

    [v]×* = -[v]×^T

    Such that: [v]×* * f = v ×* f  (spatial force cross product).
    """
    return -spatial_cross_motion(v).T


def motion_subspace_revolute(axis: int) -> SpatialVector:
    """Motion subspace vector S for a revolute joint.

    axis: 0='x', 1='y', 2='z'
    S = [e_axis; 0] where e_axis is the unit vector along the rotation axis.
    """
    S = np.zeros(6)
    S[axis] = 1.0
    return S


def motion_subspace_prismatic(axis: int) -> SpatialVector:
    """Motion subspace vector S for a prismatic joint.

    axis: 0='x', 1='y', 2='z'
    S = [0; e_axis]
    """
    S = np.zeros(6)
    S[3 + axis] = 1.0
    return S
