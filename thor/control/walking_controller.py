"""
Biomechanically accurate walking controller for THOR humanoid.

Joint angle trajectories derived from human gait biomechanics:
- Winter, D.A. (1991). "Biomechanics and Motor Control of Human Movement."
- Perry, J. (1992). "Gait Analysis: Normal and Pathological Function."

Gait cycle convention (starting at heel strike):
    0-15%:   Loading Response (double support)
    15-50%:  Midstance + Terminal Stance (single support)
    50-65%:  Pre-swing (double support)
    65-100%: Swing (initial→mid→terminal swing)

Joint angle profiles at 0.5 m/s (slow walking):
    Hip pitch:   -7 deg (extension) to +25 deg (flexion)
    Knee pitch:  0 deg (extension) to +55 deg (flexion in swing)
    Ankle pitch: -15 deg (plantarflexion) to +10 deg (dorsiflexion)
"""

import math

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..dynamics.rnea import gravity_forces
from .gait.phase_detector import detect_phase
from .gait.swing_trajectory import (
    swing_leg_angles as _gait_phase_angles,
    stance_leg_angles as _stance_leg_angles,
    HIP_STANCE_EXT, HIP_SWING_FLEX, KNEE_STANCE, KNEE_SWING_FLEX,
    ANKLE_PUSH_OFF, ANKLE_SWING,
)

# Gait timing (0.5 m/s, Winter 1991 scaled)
STEP_LENGTH: float = 0.15
STEP_DURATION: float = 0.8
DS_DURATION: float = 0.25
SWING_DURATION: float = 0.55
FOOT_CLEARANCE: float = 0.04


class WalkingController:
    """Biomechanically accurate walking controller.

    Uses gait-phase-dependent joint angle trajectories derived
    from human walking biomechanics (Winter 1991).
    """

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

        # Total duration: initial DS + n_steps * (swing + DS)
        self._total_duration = DS_DURATION + n_steps * (SWING_DURATION + DS_DURATION)

    def _get_phase(self, t: float) -> dict:
        """Determine gait phase at time t (delegates to gait.phase_detector)."""
        return detect_phase(t, DS_DURATION, SWING_DURATION, self._n_steps)

    def compute(self, q: NDArray, v: NDArray, t: float) -> NDArray:
        """Compute walking control torques with biomechanical targets."""
        n_dof = self._model.n_dof
        n_joints = n_dof - 6
        tau = np.zeros(n_dof)

        # Gravity compensation
        g = gravity_forces(self._model, q)
        tau[6:] = g[6:]

        phase_info = self._get_phase(t)
        phase = phase_info["phase"]
        s = phase_info["s"]

        # Joint target array (deviations from standing config)
        q_target = self._q_stand[7:].copy()
        dq = v[6:]

        # Left leg joint indices in q[7:] array
        # l_hip_y=18, l_hip_r=19, l_hip_p=20, l_kn_p=21, l_an_p=22, l_an_r=23
        # r_hip_y=24, r_hip_r=25, r_hip_p=26, r_kn_p=27, r_an_p=28, r_an_r=29
        L_HIP_P, L_KN_P, L_AN_P = 20, 21, 22
        R_HIP_P, R_KN_P, R_AN_P = 26, 27, 28

        if phase == "ds":
            # Double support: interpolate both legs toward standing
            pass  # Use standing config (q_target already set)

        elif phase == "swing_left":
            # Left leg swings, right leg stance
            swing_hip, swing_kn, swing_an = _gait_phase_angles(s)
            stance_hip, stance_kn, stance_an = _stance_leg_angles(s)

            # Left leg (swing): biomechanical swing trajectory
            q_target[L_HIP_P] = self._q_stand[7 + L_HIP_P] + swing_hip
            q_target[L_KN_P] = self._q_stand[7 + L_KN_P] + swing_kn
            q_target[L_AN_P] = self._q_stand[7 + L_AN_P] + swing_an

            # Right leg (stance): slight adjustments
            q_target[R_HIP_P] = self._q_stand[7 + R_HIP_P] + stance_hip
            q_target[R_KN_P] = self._q_stand[7 + R_KN_P] + stance_kn
            q_target[R_AN_P] = self._q_stand[7 + R_AN_P] + stance_an

        elif phase == "swing_right":
            # Right leg swings, left leg stance
            swing_hip, swing_kn, swing_an = _gait_phase_angles(s)
            stance_hip, stance_kn, stance_an = _stance_leg_angles(s)

            # Right leg (swing)
            q_target[R_HIP_P] = self._q_stand[7 + R_HIP_P] + swing_hip
            q_target[R_KN_P] = self._q_stand[7 + R_KN_P] + swing_kn
            q_target[R_AN_P] = self._q_stand[7 + R_AN_P] + swing_an

            # Left leg (stance)
            q_target[L_HIP_P] = self._q_stand[7 + L_HIP_P] + stance_hip
            q_target[L_KN_P] = self._q_stand[7 + L_KN_P] + stance_kn
            q_target[L_AN_P] = self._q_stand[7 + L_AN_P] + stance_an

        # PD tracking torques
        q_err = q[7:] - q_target
        for i in range(n_joints):
            if i + 1 < self._model.n_bodies:
                name = self._model.links[i + 1].name
                if "leg" in name:
                    kp, kd = self._kp_leg, self._kd_leg
                else:
                    kp, kd = self._kp_other, self._kd_other

                tau[6 + i] -= kp * q_err[i] + kd * dq[i]

                # Torque limits
                lim = self._model.links[i + 1].tau_max
                tau[6 + i] = np.clip(tau[6 + i], -lim, lim)

        return tau

    @property
    def total_duration(self) -> float:
        return self._total_duration
