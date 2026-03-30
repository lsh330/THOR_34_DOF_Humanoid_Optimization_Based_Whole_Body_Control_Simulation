"""
Centroidal Momentum Matrix (CMM) computation.

The centroidal momentum h_G relates generalized velocities to
the 6D momentum at the center of mass:

    h_G = A_G(q) * v

where A_G ∈ R^{6×n_dof} is the Centroidal Momentum Matrix,
and h_G = [k (angular momentum), l (linear momentum)]^T.

The time derivative gives the Newton-Euler equations at CoM:
    dh_G/dt = sum_i f_ext_i + [0; m*g]

Reference:
    Orin, D.E., Goswami, A. & Lee, S.-H. (2013). "Centroidal
    Dynamics of a Humanoid Robot." Autonomous Robots, 35(2-3).
"""

import numpy as np
from numpy.typing import NDArray

from ..core.spatial import spatial_inertia, skew
from ..model.robot_model import RobotModel
from ..model.kinematics import forward_kinematics, com_position, body_position


def centroidal_momentum_matrix(
    q: NDArray,
    model: RobotModel,
) -> NDArray:
    """Compute the Centroidal Momentum Matrix A_G(q).

    A_G maps generalized velocities v to centroidal momentum h_G:
        h_G = A_G * v

    Implementation uses the body Jacobians and spatial inertias:
        A_G = sum_i X_G_i^{-T} * I_i * J_i

    where X_G_i transforms from body i to the CoM frame.

    Returns:
        A_G: (6, n_dof) Centroidal Momentum Matrix
    """
    from ..model.kinematics import body_jacobian

    n_dof = model.n_dof
    X_world, _ = forward_kinematics(q, model)

    # Compute CoM position
    com = com_position(q, model)

    A_G = np.zeros((6, n_dof))

    for i in range(model.n_bodies):
        link = model.links[i]
        if link.mass < 1e-10:
            continue

        # Body position and orientation in world frame
        R_i = X_world[i][:3, :3]
        p_i = body_position(X_world[i])

        # CoM of body i in world frame
        com_i = p_i + R_i @ link.com

        # Vector from total CoM to body i CoM
        r_i = com_i - com

        # Body Jacobian (6 × n_dof)
        J_i = body_jacobian(i, q, model, X_world)

        # Linear momentum contribution: m_i * v_com_i
        # v_com_i = J_v_i * v (linear part of body Jacobian)
        J_v_i = J_i[3:, :]  # Linear velocity part

        # Angular momentum contribution about CoM:
        # L_i = I_i * omega_i + r_i × m_i * v_com_i
        J_w_i = J_i[:3, :]  # Angular velocity part

        # Centroidal contribution from body i:
        # h_G_i = [I_i*omega_i + r_i × m_i*v_i; m_i*v_i]
        A_G[3:, :] += link.mass * J_v_i                    # Linear momentum
        A_G[:3, :] += (R_i @ link.inertia @ R_i.T) @ J_w_i  # Angular momentum (inertia)
        A_G[:3, :] += link.mass * skew(r_i) @ J_v_i         # Angular momentum (cross term)

    return A_G


def centroidal_momentum(
    q: NDArray,
    v: NDArray,
    model: RobotModel,
) -> NDArray:
    """Compute centroidal momentum h_G = A_G(q) * v.

    Returns:
        h_G: (6,) = [angular_momentum(3), linear_momentum(3)]
    """
    A_G = centroidal_momentum_matrix(q, model)
    return A_G @ v
