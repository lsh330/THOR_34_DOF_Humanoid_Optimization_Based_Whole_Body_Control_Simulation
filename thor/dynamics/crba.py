"""
Composite Rigid Body Algorithm (CRBA) for mass matrix computation.

Computes: M(q) = CRBA(model, q)

where M is the n×n joint-space inertia matrix.
Complexity: O(N*d) where N = bodies, d = tree depth.

The mass matrix M appears in the equations of motion:
    M(q)*a + h(q,v) = S^T*τ + J_c^T*f_c

M is symmetric and positive-definite.

Reference:
    Featherstone, R. (2008). Ch. 6: Forward Dynamics.
    Algorithm 6.2 (Composite Rigid Body Algorithm).
"""

import numpy as np
from numpy.typing import NDArray

from ..core.spatial import (
    spatial_inertia, motion_subspace_revolute,
)
from ..model.robot_model import RobotModel
from ..model.joint_types import JointType
from ..model.kinematics import joint_transform, quat_to_rot


def crba(model: RobotModel, q: NDArray) -> NDArray:
    """Composite Rigid Body Algorithm.

    Args:
        model: Robot model
        q: Configuration vector

    Returns:
        M: (n_dof, n_dof) symmetric positive-definite mass matrix
    """
    n = model.n_bodies
    n_dof = model.n_dof
    q_joints = q[7:]

    # Precompute transforms and spatial inertias
    _eye6 = np.eye(6)
    X_up = [_eye6.copy() for _ in range(n)]
    I_c = []  # Composite inertias (mutable copies)

    for i in range(n):
        I_c.append(model.spatial_inertias[i].copy())  # Use cached, copy for mutation

    # Compute parent-to-child transforms
    R_base = quat_to_rot(q[3:7])
    from ..core.spatial import spatial_transform
    X_up[0] = spatial_transform(R_base, q[:3])

    for i in range(1, n):
        q_i = q_joints[i - 1] if (i - 1) < len(q_joints) else 0.0
        X_up[i] = joint_transform(i, q_i, model)

    # === Pass 1: Accumulate composite inertias (tips to root) ===
    for i in range(n - 1, 0, -1):
        parent = model.parent[i]
        I_c[parent] = I_c[parent] + X_up[i].T @ I_c[i] @ X_up[i]

    # === Pass 2: Compute mass matrix entries ===
    M = np.zeros((n_dof, n_dof))

    # Floating base block (6×6)
    M[:6, :6] = I_c[0]

    # Joint columns
    for i in range(1, n):
        link = model.links[i]
        jtype = link.joint_type
        if jtype not in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                         JointType.REVOLUTE_Z):
            continue

        S_i = motion_subspace_revolute(link.joint_axis)
        dof_i = 5 + i
        if dof_i >= n_dof:
            continue

        # F_i = I_c[i] * S_i (in body i frame)
        F_i = I_c[i] @ S_i

        # Diagonal element
        M[dof_i, dof_i] = S_i @ F_i

        # Off-diagonal: walk up to root
        F_up = F_i.copy()
        j = i
        while True:
            F_up = X_up[j].T @ F_up
            j = model.parent[j]

            if j == 0:
                # Floating base coupling
                M[:6, dof_i] = F_up
                M[dof_i, :6] = F_up
                break
            elif j < 0:
                break

            link_j = model.links[j]
            jtype_j = link_j.joint_type
            if jtype_j in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                           JointType.REVOLUTE_Z):
                S_j = motion_subspace_revolute(link_j.joint_axis)
                dof_j = 5 + j
                if dof_j < n_dof:
                    val = S_j @ F_up
                    M[dof_i, dof_j] = val
                    M[dof_j, dof_i] = val

    return M
