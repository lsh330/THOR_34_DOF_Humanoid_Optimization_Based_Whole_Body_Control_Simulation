"""
Whole-Body QP Inverse Dynamics Controller (Layer 2).

Solves a Quadratic Program at 1 kHz to compute joint torques
that track desired task-space accelerations while respecting
physical constraints.

Decision variables: x = [ddq(n_dof), tau(n_actuated), f_c(n_contact)]

    min_x  sum_i w_i * ||J_i*ddq + dJ_i*v - ddx_i^des||^2 + w_reg * ||tau||^2

    s.t.  M*ddq + h = S^T*tau + J_c^T*f_c     (equations of motion)
          f_c ∈ Friction Cone                    (linearized Coulomb friction)
          tau_min <= tau <= tau_max               (actuator limits)
          J_c*ddq + dJ_c*v = 0                   (contact acceleration = 0)

Reference:
    Herzog, A. et al. (2016). Autonomous Robots, 40, 473-491.
    Escande, A. et al. (2014). IJRR, 33(7), 1006-1028.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..model.robot_model import RobotModel
from ..core.constants import MU_DEFAULT, FRICTION_CONE_FACES


def _build_friction_cone(mu: float, n_faces: int = FRICTION_CONE_FACES) -> NDArray:
    """Build linearized friction cone matrix.

    For a single contact point with normal along z:
    f_z >= 0
    ||f_xy|| <= mu * f_z

    Linearized as n_faces inequalities:
    A_fc * f <= 0 where f = [fx, fy, fz]

    Returns:
        A_fc: (n_faces + 1, 3) inequality matrix
    """
    A_fc = np.zeros((n_faces + 1, 3))

    # Friction cone facets
    for i in range(n_faces):
        angle = 2.0 * np.pi * i / n_faces
        A_fc[i, 0] = np.cos(angle)
        A_fc[i, 1] = np.sin(angle)
        A_fc[i, 2] = -mu

    # Normal force non-negativity: -fz <= 0
    A_fc[n_faces, 2] = -1.0

    return A_fc


class WholeBodyQP:
    """Whole-Body QP Inverse Dynamics Controller.

    Formulates and solves a QP at each control step to compute
    joint torques tracking multiple task-space objectives.
    """

    __slots__ = (
        "_model", "_n_dof", "_n_actuated", "_n_contacts",
        "_S", "_mu", "_fc_matrix",
    )

    def __init__(self, model: RobotModel, n_contacts: int = 8,
                 mu: float = MU_DEFAULT):
        """Initialize WB-QP controller.

        Args:
            model: Robot model
            n_contacts: Number of contact points (default: 8 = 4 per foot × 2)
            mu: Friction coefficient
        """
        self._model = model
        self._n_dof = model.n_dof
        self._n_actuated = model.n_dof - 6  # Exclude floating base
        self._n_contacts = n_contacts
        self._mu = mu

        # Selection matrix S: maps actuated torques to full generalized forces
        # tau_full = S^T * tau_actuated
        # S = [0_{n_act × 6}, I_{n_act}]
        self._S = np.zeros((self._n_actuated, self._n_dof))
        self._S[:, 6:] = np.eye(self._n_actuated)

        # Friction cone for each contact point
        self._fc_matrix = _build_friction_cone(mu)

    def solve(
        self,
        M: NDArray,
        h: NDArray,
        J_c: NDArray,
        dJ_c_v: NDArray,
        task_jacobians: list[NDArray],
        task_targets: list[NDArray],
        task_weights: list[float],
        v: NDArray,
        tau_max: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Solve the whole-body QP.

        Args:
            M: (n_dof, n_dof) mass matrix
            h: (n_dof,) bias forces
            J_c: (n_c*3, n_dof) stacked contact Jacobians
            dJ_c_v: (n_c*3,) contact Jacobian derivative times velocity
            task_jacobians: List of (6, n_dof) task Jacobians
            task_targets: List of (6,) desired task-space accelerations
            task_weights: List of scalar weights per task
            v: (n_dof,) current velocity
            tau_max: (n_actuated,) torque limits (optional)

        Returns:
            ddq: (n_dof,) joint accelerations
            tau: (n_actuated,) joint torques
            f_c: (n_c*3,) contact forces
        """
        n_dof = self._n_dof
        n_act = self._n_actuated
        n_cf = J_c.shape[0] if J_c.shape[0] > 0 else 0

        # Decision variable: x = [ddq, f_c]
        # We solve for ddq and f_c, then recover tau from EOM
        n_x = n_dof + n_cf

        # === Build cost function ===
        # H = sum_i w_i * J_i^T * J_i
        # g = sum_i w_i * J_i^T * (dJ_i*v - ddx_des_i)
        H = np.zeros((n_x, n_x))
        g_vec = np.zeros(n_x)

        # Task costs (only on ddq part)
        for J_task, ddx_des, w in zip(task_jacobians, task_targets, task_weights):
            if J_task.shape[0] > 0:
                H[:n_dof, :n_dof] += w * (J_task.T @ J_task)
                g_vec[:n_dof] += w * (J_task.T @ (-ddx_des))

        # Regularization on ddq
        H[:n_dof, :n_dof] += 1e-6 * np.eye(n_dof)

        # Regularization on contact forces (minimize force magnitude)
        if n_cf > 0:
            H[n_dof:, n_dof:] += 1e-4 * np.eye(n_cf)

        # === Equality constraints: EOM + contact ===
        # M*ddq + h = S^T*tau + J_c^T*f_c
        # => M*ddq - J_c^T*f_c = S^T*tau - h
        # But tau is derived: tau = (S*M)*ddq + S*h - S*J_c^T*f_c ... complex
        # Simpler: treat as M*ddq - J_c^T*f_c + h = S^T * tau
        # where tau = S * (M*ddq + h - J_c^T*f_c)
        # This means the floating-base rows of EOM must be satisfied:
        # M_fb*ddq + h_fb = J_c_fb^T * f_c (no actuation on floating base)

        # Floating base constraint (6 equations):
        # M[:6,:] * ddq + h[:6] = J_c[:,:6]^T * f_c... wait, that's wrong
        # The EOM is: M*ddq + h = [0_{6}; tau] + J_c^T * f_c
        # So floating base: M[:6,:]*ddq + h[:6] = J_c^T[:6,:]*f_c
        n_eq = 6  # Floating base
        if n_cf > 0:
            n_eq += n_cf  # Contact: J_c*ddq + dJ_c*v = 0

        A_eq = np.zeros((n_eq, n_x))
        b_eq = np.zeros(n_eq)

        # Floating base dynamics
        A_eq[:6, :n_dof] = M[:6, :]
        if n_cf > 0:
            J_c_T = J_c.T
            A_eq[:6, n_dof:] = -J_c_T[:6, :]
        b_eq[:6] = -h[:6]

        # Contact constraint: J_c * ddq = -dJ_c * v
        if n_cf > 0:
            A_eq[6:6+n_cf, :n_dof] = J_c
            b_eq[6:6+n_cf] = -dJ_c_v

        # === Solve as least-squares with equality constraints ===
        # Using scipy's minimize with equality constraints
        from scipy.optimize import minimize as sp_minimize

        def cost(x):
            return 0.5 * x @ H @ x + g_vec @ x

        def cost_grad(x):
            return H @ x + g_vec

        constraints = [
            {"type": "eq", "fun": lambda x: A_eq @ x - b_eq,
             "jac": lambda x: A_eq}
        ]

        x0 = np.zeros(n_x)
        result = sp_minimize(cost, x0, jac=cost_grad, method="SLSQP",
                             constraints=constraints,
                             options={"maxiter": 100, "ftol": 1e-10})

        x_opt = result.x
        ddq = x_opt[:n_dof]
        f_c = x_opt[n_dof:] if n_cf > 0 else np.zeros(0)

        # Recover torques from EOM: tau = S*(M*ddq + h - J_c^T*f_c)
        tau_full = M @ ddq + h
        if n_cf > 0:
            tau_full -= J_c.T @ f_c
        tau = tau_full[6:]  # Remove floating base

        # Clamp torques
        if tau_max is not None:
            tau = np.clip(tau, -tau_max, tau_max)

        return ddq, tau, f_c
