"""
Swing leg joint trajectory generation.

Biomechanically accurate joint angle profiles for the swing phase
of bipedal walking, derived from Winter (1991) normative data.

Convention: positive = flexion (hip/knee), dorsiflexion (ankle).
"""

import math

# Biomechanical parameters (radians)
HIP_STANCE_EXT: float = math.radians(-5)
HIP_SWING_FLEX: float = math.radians(20)
KNEE_STANCE: float = math.radians(5)
KNEE_SWING_FLEX: float = math.radians(45)
ANKLE_PUSH_OFF: float = math.radians(-10)
ANKLE_SWING: float = math.radians(5)


def swing_leg_angles(s: float) -> tuple[float, float, float]:
    """Compute swing leg joint angles at phase s in [0, 1].

    s=0: toe-off, s=0.5: mid-swing, s=1: heel strike.

    Returns (hip_pitch, knee_pitch, ankle_pitch) deviations [rad].
    """
    hip_p = HIP_STANCE_EXT + (HIP_SWING_FLEX - HIP_STANCE_EXT) * (
        0.5 - 0.5 * math.cos(math.pi * s))
    kn_p = KNEE_STANCE + (KNEE_SWING_FLEX - KNEE_STANCE) * (
        math.sin(math.pi * s) ** 0.8)
    an_p = ANKLE_SWING * math.sin(math.pi * s)
    return hip_p, kn_p, an_p


def stance_leg_angles(s: float) -> tuple[float, float, float]:
    """Compute stance leg joint angles at phase s in [0, 1].

    s=0: heel strike, s=1: toe-off.

    Returns (hip_pitch, knee_pitch, ankle_pitch) deviations [rad].
    """
    hip_p = HIP_SWING_FLEX * (1.0 - s) + HIP_STANCE_EXT * s
    kn_p = KNEE_STANCE + math.radians(10) * math.sin(math.pi * s * 0.5)
    an_p = math.radians(5) * math.sin(math.pi * s) + ANKLE_PUSH_OFF * s**2
    return hip_p, kn_p, an_p
