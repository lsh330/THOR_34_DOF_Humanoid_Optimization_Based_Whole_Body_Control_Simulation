"""
Centroidal LQR Controller (Layer 1 — simplified MPC).

Controls the center of mass position and linear momentum
using LQR feedback on the centroidal dynamics:

    m * ddc = sum(f_i) + m*g
    dL/dt = sum((p_i - c) × f_i)

Simplified to linear inverted pendulum model (LIPM) for
CoM height regulation:

    ddc_x = (g/z_0) * (c_x - p_zmp_x)
    ddc_y = (g/z_0) * (c_y - p_zmp_y)

LQR on this 2D system provides reference CoM accelerations
that are tracked by the whole-body controller (Layer 2).

Reference:
    Kajita, S. et al. (2001). "The 3D Linear Inverted Pendulum Mode."
    IEEE/RSJ IROS.
    Wieber, P.-B. (2006). "Trajectory Free Linear Model Predictive
    Control for Stable Walking." Humanoids.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from ..core.constants import GRAVITY


class CentroidalLQR:
    """LQR controller for centroidal dynamics (CoM regulation)."""

    __slots__ = ("_K_x", "_K_y", "_z0", "_com_des", "_dcom_des")

    def __init__(self, z0: float = 0.7, q_pos: float = 100.0,
                 q_vel: float = 10.0, r: float = 1.0):
        """Initialize centroidal LQR.

        Args:
            z0: Nominal CoM height [m]
            q_pos, q_vel: LQR state weights
            r: LQR input weight
        """
        self._z0 = z0

        # LIPM dynamics: x_dot = A*x + B*u
        # x = [c, dc], u = ddc
        omega2 = GRAVITY / z0
        A = np.array([[0.0, 1.0], [omega2, 0.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.diag([q_pos, q_vel])
        R = np.array([[r]])

        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)

        self._K_x = K.flatten()
        self._K_y = K.flatten()
        self._com_des = np.zeros(3)
        self._dcom_des = np.zeros(3)

    def set_target(self, com_des: NDArray, dcom_des: NDArray | None = None):
        self._com_des = com_des.copy()
        self._dcom_des = dcom_des.copy() if dcom_des is not None else np.zeros(3)

    def compute(self, com: NDArray, dcom: NDArray) -> NDArray:
        """Compute desired CoM acceleration.

        Returns:
            ddc_des: (3,) desired CoM acceleration [m/s^2]
        """
        ddc = np.zeros(3)

        # X-axis LQR
        state_x = np.array([com[0] - self._com_des[0],
                            dcom[0] - self._dcom_des[0]])
        ddc[0] = -self._K_x @ state_x

        # Y-axis LQR
        state_y = np.array([com[1] - self._com_des[1],
                            dcom[1] - self._dcom_des[1]])
        ddc[1] = -self._K_y @ state_y

        # Z-axis: PD on height
        kp_z = 200.0
        kd_z = 40.0
        ddc[2] = -kp_z * (com[2] - self._com_des[2]) - kd_z * dcom[2]

        return ddc
