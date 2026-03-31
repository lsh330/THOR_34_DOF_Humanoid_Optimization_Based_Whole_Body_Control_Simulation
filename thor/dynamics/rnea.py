"""
Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics.

Computes: τ = RNEA(model, q, v, a, f_ext)

where τ is the vector of generalized forces required to produce
acceleration a given configuration q and velocity v.

The algorithm has O(N) complexity where N is the number of bodies.

Two passes:
    Forward pass: Propagate velocities and accelerations from base to tips
    Backward pass: Accumulate forces from tips to base

Special case: h = RNEA(model, q, v, 0) gives the bias force
(Coriolis + gravity), which is used in the EOM:
    M(q)*a + h(q,v) = S^T*τ + J_c^T*f_c

Reference:
    Featherstone, R. (2008). Ch. 5: Inverse Dynamics.
    Algorithm 5.3 (Recursive Newton-Euler).
"""

import numpy as np
from numpy.typing import NDArray

from ..core.spatial import (
    spatial_transform, rot_x, rot_y, rot_z,
    spatial_inertia, spatial_cross_motion, spatial_cross_force,
    motion_subspace_revolute,
)
from ..core.constants import GRAVITY_VEC
from ..model.robot_model import RobotModel
from ..model.joint_types import JointType


def rnea(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
    a: NDArray,
    f_ext: NDArray | None = None,
) -> NDArray:
    """Recursive Newton-Euler Algorithm.

    Args:
        model: Robot model (N bodies)
        q: Configuration [p_base(3), quat(4), q_joints(N-1)] ∈ R^{N+6}
        v: Velocity [v_base(6), dq_joints(N-1)] ∈ R^{N+5}
        a: Acceleration [a_base(6), ddq_joints(N-1)] ∈ R^{N+5}
        f_ext: External spatial forces on each body (N, 6) or None

    Returns:
        tau: Generalized forces [f_base(6), tau_joints(N-1)] ∈ R^{N+5}
    """
    from ..model.kinematics import joint_transform, quat_to_rot

    n = model.n_bodies
    n_dof = model.n_dof

    # Parse configuration
    p_base = q[:3]
    quat_base = q[3:7]
    q_joints = q[7:]

    # Parse velocity/acceleration
    v_base = v[:6]
    dq = v[6:]
    a_base = a[:6]
    ddq = a[6:]

    # Spatial gravity (acts as acceleration on the base)
    a_grav = np.zeros(6)
    a_grav[3:] = -GRAVITY_VEC  # Negative because we add it to base acceleration

    # Per-body arrays (list of 1D arrays — faster than 2D slicing in Python)
    vel = [np.zeros(6) for _ in range(n)]
    acc = [np.zeros(6) for _ in range(n)]
    f = [np.zeros(6) for _ in range(n)]
    _eye6 = np.eye(6)
    X_up = [_eye6.copy() for _ in range(n)]

    # Use cached spatial inertias from model (no recomputation)
    I_s = model.spatial_inertias

    # === Forward pass: velocities and accelerations ===

    # Body 0: floating base
    R_base = quat_to_rot(quat_base)
    X_up[0] = spatial_transform(R_base, p_base)

    vel[0] = v_base
    acc[0] = a_base + a_grav  # Include gravity as fictitious acceleration

    # Bodies 1..N-1
    for i in range(1, n):
        q_i = q_joints[i - 1] if (i - 1) < len(q_joints) else 0.0
        dq_i = dq[i - 1] if (i - 1) < len(dq) else 0.0
        ddq_i = ddq[i - 1] if (i - 1) < len(ddq) else 0.0

        X_up[i] = joint_transform(i, q_i, model)
        parent = model.parent[i]

        # Motion subspace
        link = model.links[i]
        jtype = link.joint_type
        if jtype in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                     JointType.REVOLUTE_Z):
            S_i = motion_subspace_revolute(link.joint_axis)
        else:
            S_i = np.zeros(6)

        # Velocity: v_i = X_up * v_parent + S_i * dq_i
        vel[i] = X_up[i] @ vel[parent] + S_i * dq_i

        # Acceleration: a_i = X_up * a_parent + S_i * ddq_i + v_i × S_i * dq_i
        vxS = spatial_cross_motion(vel[i]) @ (S_i * dq_i)
        acc[i] = X_up[i] @ acc[parent] + S_i * ddq_i + vxS

    # === Backward pass: forces ===

    for i in range(n):
        # f_i = I_i * a_i + v_i ×* (I_i * v_i)
        Iv = I_s[i] @ vel[i]
        f[i] = I_s[i] @ acc[i] + spatial_cross_force(vel[i]) @ Iv

        # Subtract external forces
        if f_ext is not None:
            f[i] -= f_ext[i]

    # Accumulate from tips to root
    for i in range(n - 1, 0, -1):
        parent = model.parent[i]
        f[parent] += X_up[i].T @ f[i]

    # === Extract generalized forces ===
    tau = np.zeros(n_dof)

    # Floating base force
    tau[:6] = f[0]

    # Joint torques: tau_i = S_i^T * f_i
    for i in range(1, n):
        link = model.links[i]
        jtype = link.joint_type
        if jtype in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                     JointType.REVOLUTE_Z):
            S_i = motion_subspace_revolute(link.joint_axis)
            dof_idx = 5 + i
            if dof_idx < n_dof:
                tau[dof_idx] = S_i @ f[i]

    return tau


def bias_forces(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
) -> NDArray:
    """Compute bias forces h(q, v) = RNEA(q, v, 0).

    h contains Coriolis, centrifugal, and gravitational terms.
    Used in EOM: M*a + h = S^T*τ + J_c^T*f_c
    """
    a_zero = np.zeros(model.n_dof)
    return rnea(model, q, v, a_zero)


def gravity_forces(
    model: RobotModel,
    q: NDArray,
) -> NDArray:
    """Compute gravity forces g(q) = RNEA(q, 0, 0).

    Used for gravity compensation in standing control.
    """
    v_zero = np.zeros(model.n_dof)
    a_zero = np.zeros(model.n_dof)
    return rnea(model, q, v_zero, a_zero)
