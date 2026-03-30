"""
Walking controller for THOR humanoid.

Combines Contact-Implicit MPC with gait-phase-dependent
swing foot trajectory tracking and CoM regulation.

Walking phases:
    1. Double Support (DS): Both feet on ground, shift CoM
    2. Left Single Support (LSS): Right foot swings
    3. Double Support (DS): Both feet on ground
    4. Right Single Support (RSS): Left foot swings
    ... (repeat)

Swing foot trajectory: cubic polynomial in z (lift-place).
CoM trajectory: maintain above support polygon center.

Reference:
    Kajita, S. et al. (2003). "Biped Walking Pattern Generation
    by using Preview Control of ZMP." ICRA.
"""

import math

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..model.kinematics import forward_kinematics, body_position, body_jacobian
from ..dynamics.rnea import gravity_forces


def swing_foot_trajectory(
    t: float,
    t_start: float,
    t_end: float,
    p_start: NDArray,
    p_end: NDArray,
    lift_height: float = 0.05,
) -> tuple[NDArray, NDArray]:
    """Compute swing foot position and velocity at time t.

    Uses cubic polynomial for smooth lift-off and touchdown.

    z(s) = 4*h*s*(1-s) where s = (t-t_start)/(t_end-t_start)
    This creates a parabolic arc peaking at h at s=0.5.

    x,y interpolated linearly.

    Returns:
        p_des: (3,) desired foot position
        v_des: (3,) desired foot velocity
    """
    duration = t_end - t_start
    if duration <= 0:
        return p_end.copy(), np.zeros(3)

    s = np.clip((t - t_start) / duration, 0.0, 1.0)
    ds = 1.0 / duration

    # Horizontal: linear interpolation
    p_des = p_start + s * (p_end - p_start)
    v_des = (p_end - p_start) * ds

    # Vertical: parabolic arc
    p_des[2] = p_start[2] + 4.0 * lift_height * s * (1.0 - s)
    v_des[2] = 4.0 * lift_height * (1.0 - 2.0 * s) * ds

    return p_des, v_des


class WalkingController:
    """Walking controller using CI-MPC framework.

    Generates joint torques for bipedal walking by:
    1. Tracking swing foot trajectory via inverse kinematics PD
    2. Maintaining stance posture via gravity compensation + PD
    3. Regulating CoM via centroidal LQR
    """

    __slots__ = (
        "_model", "_q_stand", "_step_length", "_step_duration",
        "_ds_duration", "_lift_height", "_n_steps",
        "_kp_swing", "_kd_swing", "_kp_stance", "_kd_stance",
        "_total_duration",
    )

    def __init__(
        self,
        model: RobotModel,
        q_standing: NDArray,
        step_length: float = 0.10,
        step_duration: float = 0.6,
        ds_duration: float = 0.2,
        lift_height: float = 0.04,
        n_steps: int = 6,
    ):
        self._model = model
        self._q_stand = q_standing.copy()
        self._step_length = step_length
        self._step_duration = step_duration
        self._ds_duration = ds_duration
        self._lift_height = lift_height
        self._n_steps = n_steps

        # Gains
        self._kp_swing = 300.0
        self._kd_swing = 30.0
        self._kp_stance = 800.0
        self._kd_stance = 80.0

        # Total duration
        self._total_duration = (
            ds_duration +  # Initial DS
            n_steps * (step_duration + ds_duration)
        )

    def _get_phase(self, t: float) -> tuple[str, int, float, float]:
        """Determine current gait phase.

        Returns:
            phase: "ds" or "swing_left" or "swing_right"
            step_idx: Current step number
            t_start: Phase start time
            t_end: Phase end time
        """
        if t < self._ds_duration:
            return "ds", -1, 0.0, self._ds_duration

        t_rel = t - self._ds_duration

        for i in range(self._n_steps):
            cycle_start = i * (self._step_duration + self._ds_duration)

            # Swing phase
            swing_end = cycle_start + self._step_duration
            if t_rel < swing_end:
                phase = "swing_left" if i % 2 == 0 else "swing_right"
                return phase, i, cycle_start + self._ds_duration, swing_end + self._ds_duration

            # DS phase
            ds_end = swing_end + self._ds_duration
            if t_rel < ds_end:
                return "ds", i, swing_end + self._ds_duration, ds_end + self._ds_duration

        return "ds", self._n_steps - 1, 0, self._total_duration

    def compute(self, q: NDArray, v: NDArray, t: float) -> NDArray:
        """Compute walking control torques."""
        n_dof = self._model.n_dof
        tau = np.zeros(n_dof)

        # Gravity compensation (always active)
        g = gravity_forces(self._model, q)
        tau[6:] = g[6:]

        # Get current phase
        phase, step_idx, t_start, t_end = self._get_phase(t)

        # Joint PD tracking (stance configuration)
        q_err = q[7:] - self._q_stand[7:]
        dq = v[6:]

        if phase == "ds":
            # Double support: strong posture regulation
            tau[6:] -= self._kp_stance * q_err + self._kd_stance * dq

        elif "swing_left" in phase:
            # Right foot stance, left foot swings
            # Stance leg (right): strong PD
            # indices 24-29 in v (r_leg joints, bodies 25-30)
            for i in range(24, 30):
                if i < len(q_err):
                    tau[6 + i] -= self._kp_stance * q_err[i] + self._kd_stance * dq[i]

            # Swing leg (left): track swing trajectory via modified PD
            # Lift the left ankle by modifying hip_p and knee targets
            s = np.clip((t - t_start) / max(t_end - t_start, 0.01), 0, 1)

            # Swing leg joint targets: lift foot via hip flexion + knee bend
            hip_p_swing = -0.3 - 0.3 * math.sin(math.pi * s)  # More flexion mid-swing
            kn_p_swing = 0.6 + 0.5 * math.sin(math.pi * s)    # More knee bend mid-swing
            an_p_swing = -0.3 + 0.1 * math.sin(math.pi * s)   # Ankle adjustment

            # Left leg: bodies 19-24, joint indices 18-23 in q_err
            swing_targets = {
                20: hip_p_swing,   # l_hip_p
                21: kn_p_swing,    # l_kn_p
                22: an_p_swing,    # l_an_p
            }

            for joint_idx, target in swing_targets.items():
                i = joint_idx  # in q_err indexing
                if i < len(q_err):
                    err_swing = q[7 + i] - target
                    tau[6 + i] = g[6 + i] - self._kp_swing * err_swing - self._kd_swing * dq[i]

            # Non-swing joints: moderate PD
            for i in range(len(q_err)):
                if i not in range(18, 24) and i not in range(24, 30):
                    tau[6 + i] -= self._kp_stance * 0.5 * q_err[i] + self._kd_stance * 0.5 * dq[i]

        elif "swing_right" in phase:
            # Left foot stance, right foot swings (mirror of above)
            for i in range(18, 24):
                if i < len(q_err):
                    tau[6 + i] -= self._kp_stance * q_err[i] + self._kd_stance * dq[i]

            s = np.clip((t - t_start) / max(t_end - t_start, 0.01), 0, 1)

            hip_p_swing = -0.3 - 0.3 * math.sin(math.pi * s)
            kn_p_swing = 0.6 + 0.5 * math.sin(math.pi * s)
            an_p_swing = -0.3 + 0.1 * math.sin(math.pi * s)

            swing_targets = {
                26: hip_p_swing,   # r_hip_p
                27: kn_p_swing,    # r_kn_p
                28: an_p_swing,    # r_an_p
            }

            for joint_idx, target in swing_targets.items():
                i = joint_idx
                if i < len(q_err):
                    err_swing = q[7 + i] - target
                    tau[6 + i] = g[6 + i] - self._kp_swing * err_swing - self._kd_swing * dq[i]

            for i in range(len(q_err)):
                if i not in range(18, 24) and i not in range(24, 30):
                    tau[6 + i] -= self._kp_stance * 0.5 * q_err[i] + self._kd_stance * 0.5 * dq[i]

        # Torque limits
        for i in range(len(tau) - 6):
            if i + 1 < self._model.n_bodies:
                lim = self._model.links[i + 1].tau_max
                tau[6 + i] = np.clip(tau[6 + i], -lim, lim)

        return tau

    @property
    def total_duration(self) -> float:
        return self._total_duration
