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


# Biomechanical gait parameters at 0.5 m/s (Winter 1991, scaled)
STEP_LENGTH: float = 0.15       # Reduced for stability in simulation [m]
STEP_DURATION: float = 0.8      # Per step [s]
DS_DURATION: float = 0.25       # Double support transition [s]
SWING_DURATION: float = 0.55    # Single leg swing [s]
FOOT_CLEARANCE: float = 0.04    # Max foot lift [m]

# Joint angle profiles (radians) — from biomechanics literature
# Convention: positive = flexion for hip/knee, dorsiflexion for ankle
HIP_STANCE_EXT: float = math.radians(-5)    # Peak extension at terminal stance
HIP_SWING_FLEX: float = math.radians(20)    # Peak flexion at terminal swing
KNEE_STANCE: float = math.radians(5)        # Near extension during stance
KNEE_SWING_FLEX: float = math.radians(45)   # Peak flexion during swing
ANKLE_PUSH_OFF: float = math.radians(-10)   # Plantarflexion at push-off
ANKLE_SWING: float = math.radians(5)        # Dorsiflexion during swing


def _gait_phase_angles(s: float) -> tuple[float, float, float]:
    """Compute hip, knee, ankle angles for a swing leg at phase s in [0,1].

    s=0: toe-off (start of swing)
    s=0.5: mid-swing (peak knee flexion, hip crossing neutral)
    s=1: heel strike (end of swing)

    The profiles approximate Winter's normative data:
    - Hip: sinusoidal from extension to flexion
    - Knee: rapid flexion then extension (asymmetric bell)
    - Ankle: dorsiflexion for foot clearance

    Returns:
        (hip_p, kn_p, an_p) in radians
    """
    # Hip pitch: smooth transition from extension to flexion
    # At s=0: hip at stance extension angle
    # At s=1: hip at swing flexion angle
    hip_p = HIP_STANCE_EXT + (HIP_SWING_FLEX - HIP_STANCE_EXT) * (
        0.5 - 0.5 * math.cos(math.pi * s))

    # Knee pitch: rapid flexion in early swing, then extension
    # Peak flexion at s ≈ 0.4 (initial-to-mid swing)
    # Bell curve with early peak
    kn_p = KNEE_STANCE + (KNEE_SWING_FLEX - KNEE_STANCE) * (
        math.sin(math.pi * s) ** 0.8)  # Slightly asymmetric

    # Ankle pitch: dorsiflexion during swing for foot clearance
    # Neutral at toe-off, dorsiflexed during mid-swing, neutral at heel strike
    an_p = ANKLE_SWING * math.sin(math.pi * s)

    return hip_p, kn_p, an_p


def _stance_leg_angles(s_stance: float) -> tuple[float, float, float]:
    """Compute stance leg joint angles.

    s_stance: phase within stance (0=heel strike, 1=toe-off)

    During stance, joints are relatively stable with small excursions:
    - Hip: starts flexed, gradually extends
    - Knee: slight flexion for shock absorption, then near-extension
    - Ankle: progresses from dorsiflexion to plantarflexion push-off
    """
    # Hip: flexion → extension during stance
    hip_p = HIP_SWING_FLEX * (1.0 - s_stance) + HIP_STANCE_EXT * s_stance

    # Knee: slight bend at loading, near-extension at midstance
    kn_p = KNEE_STANCE + math.radians(10) * math.sin(math.pi * s_stance * 0.5)

    # Ankle: dorsiflexion at midstance, plantarflexion at push-off
    an_p = math.radians(5) * math.sin(math.pi * s_stance) + \
           ANKLE_PUSH_OFF * s_stance**2

    return hip_p, kn_p, an_p


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
        """Determine gait phase at time t."""
        if t < DS_DURATION:
            return {"phase": "ds", "step": -1, "s": t / DS_DURATION}

        t_rel = t - DS_DURATION
        step_cycle = SWING_DURATION + DS_DURATION

        step_idx = int(t_rel / step_cycle)
        t_in_cycle = t_rel - step_idx * step_cycle

        if step_idx >= self._n_steps:
            return {"phase": "ds", "step": self._n_steps, "s": 1.0}

        if t_in_cycle < SWING_DURATION:
            is_left = (step_idx % 2 == 0)
            s = t_in_cycle / SWING_DURATION
            return {
                "phase": "swing_left" if is_left else "swing_right",
                "step": step_idx,
                "s": s,  # 0→1 within swing
            }
        else:
            s = (t_in_cycle - SWING_DURATION) / DS_DURATION
            return {"phase": "ds", "step": step_idx, "s": s}

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
