"""
Biomechanically accurate walking controller for THOR humanoid.

Uses CONTINUOUS joint trajectory generation that smoothly transitions
between gait phases without snapping back to standing configuration.

Each leg maintains a continuous angle profile over the full gait cycle:
- Swing phase: follows biomechanical swing trajectory (Winter 1991)
- Stance phase: follows stance trajectory (smooth extension)
- Double support: smooth interpolation between phases

Reference:
    Winter, D.A. (1991). Biomechanics and Motor Control of Human Movement.
    Perry, J. (1992). Gait Analysis: Normal and Pathological Function.
"""

import math

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..dynamics.rnea import gravity_forces
from .gait.swing_trajectory import swing_leg_angles, stance_leg_angles


# Gait timing
DS_DURATION: float = 0.25
SWING_DURATION: float = 0.55
STEP_CYCLE: float = SWING_DURATION + DS_DURATION  # 0.8s per step


def _compute_leg_targets(
    t: float,
    n_steps: int,
    q_stand_joints: NDArray,
) -> tuple[NDArray, NDArray]:
    """Compute continuous joint targets for both legs.

    Returns target joint angles for left and right leg pitch joints
    (hip_p, kn_p, an_p) as deviations from standing configuration.
    """
    # Left leg pitch indices (in q[7:] space)
    L_HIP_P, L_KN_P, L_AN_P = 20, 21, 22
    R_HIP_P, R_KN_P, R_AN_P = 26, 27, 28

    q_target = q_stand_joints.copy()

    # Determine what each leg should do at time t
    if t < DS_DURATION:
        # Initial DS: hold standing
        return q_target

    t_rel = t - DS_DURATION
    step_idx = int(t_rel / STEP_CYCLE)
    t_in_cycle = t_rel - step_idx * STEP_CYCLE

    if step_idx >= n_steps:
        return q_target

    is_left_swing = (step_idx % 2 == 0)

    if t_in_cycle < SWING_DURATION:
        # Swing phase
        s = t_in_cycle / SWING_DURATION
        swing_hip, swing_kn, swing_an = swing_leg_angles(s)
        stance_hip, stance_kn, stance_an = stance_leg_angles(s)

        if is_left_swing:
            q_target[L_HIP_P] = q_stand_joints[L_HIP_P] + swing_hip
            q_target[L_KN_P] = q_stand_joints[L_KN_P] + swing_kn
            q_target[L_AN_P] = q_stand_joints[L_AN_P] + swing_an
            q_target[R_HIP_P] = q_stand_joints[R_HIP_P] + stance_hip
            q_target[R_KN_P] = q_stand_joints[R_KN_P] + stance_kn
            q_target[R_AN_P] = q_stand_joints[R_AN_P] + stance_an
        else:
            q_target[R_HIP_P] = q_stand_joints[R_HIP_P] + swing_hip
            q_target[R_KN_P] = q_stand_joints[R_KN_P] + swing_kn
            q_target[R_AN_P] = q_stand_joints[R_AN_P] + swing_an
            q_target[L_HIP_P] = q_stand_joints[L_HIP_P] + stance_hip
            q_target[L_KN_P] = q_stand_joints[L_KN_P] + stance_kn
            q_target[L_AN_P] = q_stand_joints[L_AN_P] + stance_an
    else:
        # DS transition: smooth interpolation toward next phase's start
        s_ds = (t_in_cycle - SWING_DURATION) / DS_DURATION

        # End of current swing phase (s=1)
        swing_hip_end, swing_kn_end, swing_an_end = swing_leg_angles(1.0)
        stance_hip_end, stance_kn_end, stance_an_end = stance_leg_angles(1.0)

        # Start of next swing phase (s=0 of next step)
        # Next step swings the OTHER leg
        swing_hip_start, swing_kn_start, swing_an_start = swing_leg_angles(0.0)
        stance_hip_start, stance_kn_start, stance_an_start = stance_leg_angles(0.0)

        # Smooth interpolation (cosine blend)
        blend = 0.5 - 0.5 * math.cos(math.pi * s_ds)

        if is_left_swing:
            # Left was swinging → now becomes stance (next step swings right)
            l_hip = swing_hip_end * (1 - blend) + stance_hip_start * blend
            l_kn = swing_kn_end * (1 - blend) + stance_kn_start * blend
            l_an = swing_an_end * (1 - blend) + stance_an_start * blend
            r_hip = stance_hip_end * (1 - blend) + swing_hip_start * blend
            r_kn = stance_kn_end * (1 - blend) + swing_kn_start * blend
            r_an = stance_an_end * (1 - blend) + swing_an_start * blend
        else:
            r_hip = swing_hip_end * (1 - blend) + stance_hip_start * blend
            r_kn = swing_kn_end * (1 - blend) + stance_kn_start * blend
            r_an = swing_an_end * (1 - blend) + stance_an_start * blend
            l_hip = stance_hip_end * (1 - blend) + swing_hip_start * blend
            l_kn = stance_kn_end * (1 - blend) + swing_kn_start * blend
            l_an = stance_an_end * (1 - blend) + swing_an_start * blend

        q_target[L_HIP_P] = q_stand_joints[L_HIP_P] + l_hip
        q_target[L_KN_P] = q_stand_joints[L_KN_P] + l_kn
        q_target[L_AN_P] = q_stand_joints[L_AN_P] + l_an
        q_target[R_HIP_P] = q_stand_joints[R_HIP_P] + r_hip
        q_target[R_KN_P] = q_stand_joints[R_KN_P] + r_kn
        q_target[R_AN_P] = q_stand_joints[R_AN_P] + r_an

    return q_target


def _compute_leg_velocity_targets(
    t: float, n_steps: int, q_stand_joints: NDArray,
) -> NDArray:
    """Compute target joint velocities via finite difference."""
    dt_fd = 0.002
    q_now = _compute_leg_targets(t, n_steps, q_stand_joints)
    t_next = min(t + dt_fd, DS_DURATION + n_steps * STEP_CYCLE - 0.001)
    q_next = _compute_leg_targets(t_next, n_steps, q_stand_joints)
    return (q_next - q_now) / dt_fd


class WalkingController:
    """Continuous-trajectory walking controller."""

    __slots__ = (
        "_model", "_q_stand", "_n_steps", "_total_duration",
        "_kp_leg", "_kd_leg", "_kp_other", "_kd_other",
    )

    def __init__(
        self,
        model: RobotModel,
        q_standing: NDArray,
        n_steps: int = 6,
        kp_leg: float = 600.0,
        kd_leg: float = 60.0,
    ):
        self._model = model
        self._q_stand = q_standing.copy()
        self._n_steps = n_steps
        self._kp_leg = kp_leg
        self._kd_leg = kd_leg
        self._kp_other = 300.0
        self._kd_other = 30.0
        self._total_duration = DS_DURATION + n_steps * STEP_CYCLE

    def _get_phase(self, t: float) -> dict:
        """Get phase info for logging."""
        from .gait.phase_detector import detect_phase
        return detect_phase(t, DS_DURATION, SWING_DURATION, self._n_steps)

    def compute(self, q: NDArray, v: NDArray, t: float) -> NDArray:
        """Compute walking torques with continuous trajectory tracking."""
        n_dof = self._model.n_dof
        n_joints = n_dof - 6
        tau = np.zeros(n_dof)

        # Gravity compensation
        g = gravity_forces(self._model, q)
        tau[6:] = g[6:]

        # Continuous target trajectory
        q_target = _compute_leg_targets(t, self._n_steps, self._q_stand[7:])
        dq_target = _compute_leg_velocity_targets(t, self._n_steps, self._q_stand[7:])

        # PD tracking with feedforward velocity
        q_err = q[7:] - q_target
        dq_err = v[6:] - dq_target

        for i in range(n_joints):
            if i + 1 < self._model.n_bodies:
                name = self._model.links[i + 1].name
                if "leg" in name:
                    kp, kd = self._kp_leg, self._kd_leg
                else:
                    kp, kd = self._kp_other, self._kd_other

                tau[6 + i] -= kp * q_err[i] + kd * dq_err[i]
                lim = self._model.links[i + 1].tau_max
                tau[6 + i] = np.clip(tau[6 + i], -lim, lim)

        return tau

    @property
    def total_duration(self) -> float:
        return self._total_duration
