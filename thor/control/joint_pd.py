"""
Joint-level PD + gravity compensation controller (Layer 3).

tau = g(q) + Kp*(q_des - q) + Kd*(dq_des - dq)

Reference:
    Hopkins & Leonessa (2015). IJHR, Sec. IV-A.
"""

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..dynamics.rnea import gravity_forces


class JointPDController:
    """Joint-level PD controller with gravity compensation."""

    __slots__ = ("_model", "_Kp", "_Kd", "_q_des", "_n_dof")

    def __init__(self, model: RobotModel, q_des: NDArray,
                 kp_legs: float = 500.0, kd_legs: float = 50.0,
                 kp_arms: float = 100.0, kd_arms: float = 10.0,
                 kp_other: float = 200.0, kd_other: float = 20.0):
        self._model = model
        self._n_dof = model.n_dof
        self._q_des = q_des.copy()

        n_joints = model.n_dof - 6
        self._Kp = np.ones(n_joints) * kp_other
        self._Kd = np.ones(n_joints) * kd_other

        for i in range(n_joints):
            if i + 1 < model.n_bodies:
                name = model.links[i + 1].name
                if "leg" in name:
                    self._Kp[i] = kp_legs
                    self._Kd[i] = kd_legs
                elif "arm" in name:
                    self._Kp[i] = kp_arms
                    self._Kd[i] = kd_arms

    def compute(self, q: NDArray, v: NDArray, t: float) -> NDArray:
        """Compute control torques."""
        tau = np.zeros(self._n_dof)

        g = gravity_forces(self._model, q)
        q_err = q[7:] - self._q_des[7:]
        dq = v[6:]

        tau_joints = g[6:] - self._Kp * q_err - self._Kd * dq

        # Clamp torques
        for i in range(len(tau_joints)):
            if i + 1 < self._model.n_bodies:
                lim = self._model.links[i + 1].tau_max
                tau_joints[i] = np.clip(tau_joints[i], -lim, lim)

        tau[6:] = tau_joints
        return tau

    def set_target(self, q_des: NDArray) -> None:
        self._q_des = q_des.copy()
