"""JIT-compiled forward kinematics.

Provides three @njit kernels:
    forward_kinematics_jit  — writes X_world for all bodies into a pre-allocated
                               buffer (avoids repeated allocation in hot loops).
    body_position_jit       — extracts 3-D position from a 6×6 spatial transform.
    com_position_jit        — computes centre-of-mass from FK results.

Spatial transform convention (Featherstone 2008, §2.8):
    X = [[R,   0       ],
         [-R*skew(p), R]]

    Given X, position p is recovered via:
        -skew(p) = R^T @ X[3:, :3]
        p[0] = -(-skew(p))[2, 1] = (R^T @ X[3:,:3])[2,1]  (with sign flip)

The world transform X_world[i] satisfies:
    X_world[i] = X_parent_i @ X_world[parent[i]]

so that the spatial velocity in body frame is:
    v_i = X_world[i] @ v_world
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def forward_kinematics_jit(
    n_bodies: int,
    parent: np.ndarray,           # (n_bodies,) int32
    joint_types: np.ndarray,      # (n_bodies,) int32
    joint_offsets: np.ndarray,    # (n_bodies, 3) float64  [m]
    joint_rotations: np.ndarray,  # (n_bodies, 3, 3) float64
    q: np.ndarray,                # (n_q,) float64  [p(3), quat(4), joints(N-1)]
    X_world_buf: np.ndarray,      # (n_bodies, 6, 6) float64  OUTPUT
) -> None:
    """Compute forward kinematics into pre-allocated buffer.

    Writes X_world_buf[i] (6×6 spatial transform, world → body frame)
    for all bodies i in 0 .. n_bodies-1.

    Args:
        n_bodies: Number of rigid bodies.
        parent: Parent body indices (-1 for root).
        joint_types: Joint type codes (0=FIXED,1=REV_X,2=REV_Y,3=REV_Z,4=FLOATING).
        joint_offsets: Fixed offset from parent origin to child joint [m].
        joint_rotations: Fixed rotation from parent frame to child joint frame.
        q: Configuration [p_base(3), quat_base(4), q_joints(N-1)].
        X_world_buf: Pre-allocated output buffer (n_bodies, 6, 6); overwritten.
    """
    quat    = q[3:7]    # [w, x, y, z]
    q_joints = q[7:]    # revolute joint angles [rad]

    # ------------------------------------------------------------------ #
    # Body 0: floating base                                                #
    # ------------------------------------------------------------------ #
    w = quat[0]; x = quat[1]; y = quat[2]; z = quat[3]

    R = np.zeros((3, 3))
    R[0, 0] = 1.0 - 2.0*(y*y + z*z)
    R[0, 1] = 2.0*(x*y - w*z)
    R[0, 2] = 2.0*(x*z + w*y)
    R[1, 0] = 2.0*(x*y + w*z)
    R[1, 1] = 1.0 - 2.0*(x*x + z*z)
    R[1, 2] = 2.0*(y*z - w*x)
    R[2, 0] = 2.0*(x*z - w*y)
    R[2, 1] = 2.0*(y*z + w*x)
    R[2, 2] = 1.0 - 2.0*(x*x + y*y)

    px = q[0]; py = q[1]; pz = q[2]

    X_world_buf[0, :3, :3] = R
    X_world_buf[0, :3, 3:] = 0.0
    X_world_buf[0, 3:, 3:] = R
    for r in range(3):
        X_world_buf[0, 3+r, 0] = -(R[r, 1]*pz - R[r, 2]*py)
        X_world_buf[0, 3+r, 1] = -(R[r, 2]*px - R[r, 0]*pz)
        X_world_buf[0, 3+r, 2] = -(R[r, 0]*py - R[r, 1]*px)

    # ------------------------------------------------------------------ #
    # Bodies 1 .. n_bodies-1                                              #
    # ------------------------------------------------------------------ #
    for i in range(1, n_bodies):
        qi = 0.0
        idx = i - 1
        if idx < len(q_joints):
            qi = q_joints[idx]

        jtype = joint_types[i]
        c = math.cos(qi)
        s = math.sin(qi)

        Rj = np.zeros((3, 3))
        if jtype == 1:          # REVOLUTE_X
            Rj[0, 0] = 1.0
            Rj[1, 1] =  c;  Rj[1, 2] = -s
            Rj[2, 1] =  s;  Rj[2, 2] =  c
        elif jtype == 2:        # REVOLUTE_Y
            Rj[0, 0] =  c;  Rj[0, 2] =  s
            Rj[1, 1] = 1.0
            Rj[2, 0] = -s;  Rj[2, 2] =  c
        elif jtype == 3:        # REVOLUTE_Z
            Rj[0, 0] =  c;  Rj[0, 1] = -s
            Rj[1, 0] =  s;  Rj[1, 1] =  c
            Rj[2, 2] = 1.0
        else:                   # FIXED
            Rj[0, 0] = 1.0
            Rj[1, 1] = 1.0
            Rj[2, 2] = 1.0

        # R_total = joint_rotations[i] @ Rj
        Rt = np.zeros((3, 3))
        for r in range(3):
            for cc in range(3):
                for k in range(3):
                    Rt[r, cc] += joint_rotations[i, r, k] * Rj[k, cc]

        ox = joint_offsets[i, 0]
        oy = joint_offsets[i, 1]
        oz = joint_offsets[i, 2]

        # X_parent_i (local transform, parent → body i)
        Xp = np.zeros((6, 6))
        Xp[:3, :3] = Rt
        Xp[3:, 3:] = Rt
        for r in range(3):
            Xp[3+r, 0] = -(Rt[r, 1]*oz - Rt[r, 2]*oy)
            Xp[3+r, 1] = -(Rt[r, 2]*ox - Rt[r, 0]*oz)
            Xp[3+r, 2] = -(Rt[r, 0]*oy - Rt[r, 1]*ox)

        # X_world[i] = Xp @ X_world[parent[i]]
        par = parent[i]
        for r in range(6):
            for cc in range(6):
                acc = 0.0
                for k in range(6):
                    acc += Xp[r, k] * X_world_buf[par, k, cc]
                X_world_buf[i, r, cc] = acc


@njit(cache=True)
def body_position_jit(X_world_i: np.ndarray) -> np.ndarray:
    """Extract 3-D position from a 6×6 spatial transform.

    For X = [[R, 0], [-R*skew(p), R]]:
        X[3:, :3] = -R*skew(p)
        => -skew(p) = R^T @ X[3:, :3]
        => p extracted from skew-symmetric matrix elements.

    skew(p) = [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
    -skew(p)[2,1] = px, -skew(p)[0,2] = py, -skew(p)[1,0] = pz
    but we have -skew(p), so:
        neg_skew[2,1] = px → p[0] = -neg_skew[2,1]? No:
        neg_skew = R^T @ X[3:,:3]
        neg_skew = -skew(p)
        -skew(p)[0,1] = pz  → p[2] = neg_skew[0,1]  (with + sign)
        etc.

    Derivation:
        skew(p) = [[0, -p2, p1],
                   [p2, 0, -p0],
                   [-p1, p0, 0]]
        -skew(p)[2,1] = p0  → p[0] = neg_skew_p[2,1]  wait: -skew[2,1] = -p0 → p0 = -neg_skew[2,1]
        Actually neg_skew_p = R^T @ X[3:,:3] = -skew(p):
            -skew(p)[0,1] = p[2]   → p[2] = -(-skew(p))[0,1] needs checking.
    Let me be explicit:
        neg_skew_p = -skew(p)
        = [[0,  p2, -p1],
           [-p2, 0,  p0],
           [p1, -p0,  0]]
    So: neg_skew_p[2,1] = -p0 → p[0] = -neg_skew_p[2,1]
        neg_skew_p[0,2] = -p1 → p[1] = -neg_skew_p[0,2]
        neg_skew_p[1,0] = -p2 → p[2] = -neg_skew_p[1,0]

    Args:
        X_world_i: (6, 6) spatial transform for body i.

    Returns:
        p: (3,) position of body origin in world frame [m].
    """
    R = X_world_i[:3, :3]
    # neg_skew_p = R^T @ X[3:, :3]
    neg_skew_p = np.zeros((3, 3))
    for r in range(3):
        for cc in range(3):
            acc = 0.0
            for k in range(3):
                acc += R[k, r] * X_world_i[3+k, cc]   # R^T[r,k] = R[k,r]
            neg_skew_p[r, cc] = acc

    p = np.zeros(3)
    p[0] = -neg_skew_p[2, 1]
    p[1] = -neg_skew_p[0, 2]
    p[2] = -neg_skew_p[1, 0]
    return p


@njit(cache=True)
def com_position_jit(
    n_bodies: int,
    total_mass: float,
    masses: np.ndarray,      # (n_bodies,) float64  [kg]
    coms: np.ndarray,        # (n_bodies, 3) float64 [m], in body frame
    X_world: np.ndarray,     # (n_bodies, 6, 6) float64
) -> np.ndarray:
    """Compute centre of mass position in world frame.

    c = (1/M_total) * sum_i  m_i * (p_i + R_i @ com_body_i)

    where p_i is the body origin position and R_i is the rotation.

    Args:
        n_bodies: Number of rigid bodies.
        total_mass: Sum of all link masses [kg].
        masses: Per-body masses [kg].
        coms: Per-body CoM offset from body origin in body frame [m].
        X_world: World spatial transforms from forward_kinematics_jit.

    Returns:
        com: (3,) CoM position in world frame [m].
    """
    com = np.zeros(3)
    for i in range(n_bodies):
        p_i = body_position_jit(X_world[i])
        R_i = X_world[i, :3, :3]

        # com_world_i = p_i + R_i @ coms[i]
        for k in range(3):
            val = p_i[k]
            for j in range(3):
                val += R_i[k, j] * coms[i, j]
            com[k] += masses[i] * val

    for k in range(3):
        com[k] /= total_mass
    return com
