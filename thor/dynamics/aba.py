"""
Articulated Body Algorithm (ABA) for forward dynamics.

Computes: ddq = ABA(model, q, v, tau, f_ext)

Given configuration q, velocity v, and applied forces tau/f_ext,
computes the resulting joint accelerations ddq.

Three passes:
    Pass 1 (outward): Compute velocities and bias forces
    Pass 2 (inward): Compute articulated-body inertias
    Pass 3 (outward): Compute accelerations

Complexity: O(N) where N is the number of bodies.

Reference:
    Featherstone, R. (2008). Ch. 7: Forward Dynamics.
    Algorithm 7.1 (Articulated-Body Algorithm).
"""

import numpy as np
from numpy.typing import NDArray

from ..core.spatial import (
    spatial_inertia, spatial_cross_motion, spatial_cross_force,
    motion_subspace_revolute,
)
from ..core.constants import GRAVITY_VEC
from ..model.robot_model import RobotModel
from ..model.joint_types import JointType
from ..model.kinematics import joint_transform, quat_to_rot


def aba(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
    tau: NDArray,
    f_ext: NDArray | None = None,
) -> NDArray:
    """Articulated Body Algorithm for forward dynamics.

    Args:
        model: Robot model
        q: Configuration [p(3), quat(4), q_joints(N-1)]
        v: Velocity [v_base(6), dq(N-1)]
        tau: Applied forces [f_base(6), tau_joints(N-1)]
        f_ext: External spatial forces (N, 6) or None

    Returns:
        ddq: Accelerations [a_base(6), ddq_joints(N-1)]
    """
    n = model.n_bodies
    n_dof = model.n_dof

    q_joints = q[7:]
    v_base = v[:6]
    dq = v[6:]

    # Gravity as fictitious base acceleration
    a_grav = np.zeros(6)
    a_grav[3:] = -GRAVITY_VEC

    # Precompute per-body data
    X_up = [np.eye(6) for _ in range(n)]
    S = [np.zeros(6) for _ in range(n)]       # Motion subspace
    vel = [np.zeros(6) for _ in range(n)]     # Spatial velocity
    c_bias = [np.zeros(6) for _ in range(n)]  # Velocity-dependent bias
    pA = [np.zeros(6) for _ in range(n)]      # Articulated bias force
    IA = []                                     # Articulated inertia

    for i in range(n):
        IA.append(model.spatial_inertias[i].copy())  # Cached, copy for mutation

    # === Pass 1: Velocity propagation (base → tips) ===

    R_base = quat_to_rot(q[3:7])
    from ..core.spatial import spatial_transform
    X_up[0] = spatial_transform(R_base, q[:3])
    vel[0] = v_base

    for i in range(1, n):
        q_i = q_joints[i - 1] if (i - 1) < len(q_joints) else 0.0
        dq_i = dq[i - 1] if (i - 1) < len(dq) else 0.0

        X_up[i] = joint_transform(i, q_i, model)
        parent = model.parent[i]

        link = model.links[i]
        if link.joint_type in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                                JointType.REVOLUTE_Z):
            S[i] = motion_subspace_revolute(link.joint_axis)

        vel[i] = X_up[i] @ vel[parent] + S[i] * dq_i
        c_bias[i] = spatial_cross_motion(vel[i]) @ (S[i] * dq_i)

    # === Pass 2: Articulated inertia (tips → base) ===

    for i in range(n):
        Iv = IA[i] @ vel[i]
        pA[i] = spatial_cross_force(vel[i]) @ Iv
        if f_ext is not None:
            pA[i] -= f_ext[i]

    # U, D, u arrays for joint-space inversion
    U = [np.zeros(6) for _ in range(n)]
    D = np.zeros(n)
    u = np.zeros(n)

    for i in range(n - 1, 0, -1):
        link = model.links[i]
        if link.joint_type not in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                                     JointType.REVOLUTE_Z):
            parent = model.parent[i]
            IA[parent] += X_up[i].T @ IA[i] @ X_up[i]
            pA[parent] += X_up[i].T @ pA[i]
            continue

        U[i] = IA[i] @ S[i]
        D[i] = S[i] @ U[i]

        dof_i = 5 + i
        tau_i = tau[dof_i] if dof_i < n_dof else 0.0
        u[i] = tau_i - S[i] @ pA[i]

        if abs(D[i]) > 1e-12:
            Ia = IA[i] - np.outer(U[i], U[i]) / D[i]
            pa = pA[i] + Ia @ c_bias[i] + U[i] * (u[i] / D[i])
        else:
            Ia = IA[i].copy()
            pa = pA[i] + Ia @ c_bias[i]

        parent = model.parent[i]
        IA[parent] += X_up[i].T @ Ia @ X_up[i]
        pA[parent] += X_up[i].T @ pa

    # === Pass 3: Acceleration propagation (base → tips) ===

    ddq_out = np.zeros(n_dof)
    acc = [np.zeros(6) for _ in range(n)]

    # Base acceleration
    if abs(np.linalg.det(IA[0])) > 1e-12:
        acc[0] = np.linalg.solve(IA[0], tau[:6] - pA[0]) + a_grav
    else:
        acc[0] = a_grav
    ddq_out[:6] = acc[0]

    for i in range(1, n):
        parent = model.parent[i]
        link = model.links[i]

        a_parent = X_up[i] @ acc[parent] + c_bias[i]

        if link.joint_type in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                                JointType.REVOLUTE_Z):
            dof_i = 5 + i
            if abs(D[i]) > 1e-12:
                ddq_i = (u[i] - U[i] @ a_parent) / D[i]
            else:
                ddq_i = 0.0

            if dof_i < n_dof:
                ddq_out[dof_i] = ddq_i
            acc[i] = a_parent + S[i] * ddq_i
        else:
            acc[i] = a_parent

    return ddq_out
