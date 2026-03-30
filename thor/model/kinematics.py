"""
Forward kinematics for the THOR floating-base humanoid.

Computes body-frame spatial transforms, Jacobians, and
center of mass position for the full 34-DOF kinematic tree.

The configuration vector q has the structure:
    q = [p_base(3), quat_base(4), q_joints(34)]  ∈ R^41

The velocity vector v has the structure:
    v = [v_base(3), omega_base(3), dq_joints(34)] ∈ R^40

Reference:
    Featherstone, R. (2008). Ch. 4: Forward Kinematics.
"""

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.spatial import (
    spatial_transform, rot_x, rot_y, rot_z,
    motion_subspace_revolute, skew,
)
from .robot_model import RobotModel
from .joint_types import JointType


from .quaternion import quat_to_rot  # noqa: E402 — re-exported for compatibility


def joint_transform(link_idx: int, q_joint: float,
                     model: RobotModel) -> NDArray:
    """Compute the spatial transform for a single revolute joint.

    X_J(q) = X_tree * X_rot(q)

    where X_tree is the fixed transform from parent and
    X_rot is the rotation due to joint angle q.
    """
    link = model.links[link_idx]

    # Joint rotation
    jtype = link.joint_type
    if jtype == JointType.REVOLUTE_X:
        R_joint = rot_x(q_joint)
    elif jtype == JointType.REVOLUTE_Y:
        R_joint = rot_y(q_joint)
    elif jtype == JointType.REVOLUTE_Z:
        R_joint = rot_z(q_joint)
    else:
        R_joint = np.eye(3)

    # Combined: fixed offset rotation * joint rotation
    R_total = link.joint_rotation @ R_joint
    return spatial_transform(R_total, link.joint_offset)


def forward_kinematics(
    q: NDArray,
    model: RobotModel,
) -> tuple[list[NDArray], list[NDArray]]:
    """Compute forward kinematics for all bodies.

    Args:
        q: Configuration [p_base(3), quat_base(4), q_joints(34)]
        model: Robot model

    Returns:
        X_world: List of spatial transforms from world to each body frame
        X_parent: List of spatial transforms from parent to each body frame
    """
    n = model.n_bodies
    p_base = q[:3]
    quat_base = q[3:7]
    q_joints = q[7:]

    X_parent = [np.eye(6) for _ in range(n)]
    X_world = [np.eye(6) for _ in range(n)]

    # Body 0: floating base
    R_base = quat_to_rot(quat_base)
    X_world[0] = spatial_transform(R_base, p_base)
    X_parent[0] = X_world[0].copy()

    # Bodies 1..N: revolute joints
    for i in range(1, n):
        q_i = q_joints[i - 1] if (i - 1) < len(q_joints) else 0.0
        X_parent[i] = joint_transform(i, q_i, model)

        parent = model.parent[i]
        X_world[i] = X_parent[i] @ X_world[parent]

    return X_world, X_parent


def body_position(X_world_i: NDArray) -> NDArray:
    """Extract 3D position from a spatial transform.

    For X = [[R, 0], [-R*skew(p), R]]:
        X[3:,:3] = -R*skew(p)
        => skew(p) = -R^T * X[3:,:3]
        => p = extract_from_skew(-R^T * X[3:,:3])
    """
    R = X_world_i[:3, :3]
    neg_skew_p = R.T @ X_world_i[3:, :3]  # = -skew(p)
    # Extract p from -skew(p): negate the standard extraction
    p = np.array([-neg_skew_p[2, 1], -neg_skew_p[0, 2], -neg_skew_p[1, 0]])
    return p


def com_position(q: NDArray, model: RobotModel) -> NDArray:
    """Compute center of mass position in world frame.

    c = (1/M) * sum_i m_i * p_i

    where p_i is the world position of body i's center of mass.
    """
    X_world, _ = forward_kinematics(q, model)

    com = np.zeros(3)
    for i in range(model.n_bodies):
        p_i = body_position(X_world[i])
        link = model.links[i]
        # Transform CoM from body frame to world
        R_i = X_world[i][:3, :3]
        com_world = p_i + R_i @ link.com
        com += link.mass * com_world

    return com / model.total_mass


def body_jacobian(
    body_idx: int,
    q: NDArray,
    model: RobotModel,
    X_world: Optional[list[NDArray]] = None,
) -> NDArray:
    """Compute the 6×n_dof body Jacobian for a specific body.

    J maps generalized velocities v to the spatial velocity of the body:
        v_body = J * v

    The Jacobian columns are the motion subspace vectors transformed
    to the body frame, along the kinematic chain from body to root.

    Args:
        body_idx: Target body index
        q: Configuration vector
        model: Robot model
        X_world: Pre-computed world transforms (optional)

    Returns:
        J: (6, n_dof) Jacobian matrix
    """
    n_dof = model.n_dof

    if X_world is None:
        X_world, _ = forward_kinematics(q, model)

    J = np.zeros((6, n_dof))

    # Walk up the kinematic chain from body_idx to root
    i = body_idx
    while i >= 0:
        link = model.links[i]

        if i == 0:
            # Floating base: 6 DOF columns (0-5)
            # Transform from base frame to target body frame
            X_bi = X_world[body_idx] @ np.linalg.inv(X_world[0])
            J[:, :6] = X_bi  # All 6 spatial DOFs
        else:
            # Revolute joint: single DOF column
            jtype = link.joint_type
            if jtype in (JointType.REVOLUTE_X, JointType.REVOLUTE_Y,
                         JointType.REVOLUTE_Z):
                axis = link.joint_axis
                S_i = motion_subspace_revolute(axis)

                # Transform S from joint frame to target body frame
                X_bi = X_world[body_idx] @ np.linalg.inv(X_world[i])
                col_idx = 5 + i  # DOF index (6 base + joint index)
                if col_idx < n_dof:
                    J[:, col_idx] = X_bi @ S_i

        i = model.parent[i]

    return J
