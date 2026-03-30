"""
Contact Sequence Planner (Layer 0).

Generates gait patterns (contact schedules) for bipedal locomotion.
Outputs timing and foot placement for the higher-level controllers.

Supported gaits:
    - Standing: both feet in contact
    - Stepping: alternating single support (in-place)
    - Walking: forward progression with step length

Reference:
    Kajita, S. et al. (2003). "Biped Walking Pattern Generation by
    using Preview Control of ZMP." ICRA.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ContactPhase:
    """Single phase of a contact sequence."""
    t_start: float          # Phase start time [s]
    t_end: float            # Phase end time [s]
    left_foot_contact: bool  # Left foot on ground
    right_foot_contact: bool # Right foot on ground
    left_foot_pos: NDArray   # Left foot target position (3,)
    right_foot_pos: NDArray  # Right foot target position (3,)


def generate_standing_plan(duration: float = 5.0) -> list[ContactPhase]:
    """Generate standing (double support) contact plan."""
    return [ContactPhase(
        t_start=0.0, t_end=duration,
        left_foot_contact=True, right_foot_contact=True,
        left_foot_pos=np.array([0.0, 0.093, 0.0]),
        right_foot_pos=np.array([0.0, -0.093, 0.0]),
    )]


def generate_stepping_plan(
    n_steps: int = 6,
    step_duration: float = 0.8,
    double_support_time: float = 0.2,
) -> list[ContactPhase]:
    """Generate in-place stepping contact plan."""
    phases = []
    t = 0.0
    l_pos = np.array([0.0, 0.093, 0.0])
    r_pos = np.array([0.0, -0.093, 0.0])

    # Initial double support
    phases.append(ContactPhase(t, t + double_support_time,
                               True, True, l_pos.copy(), r_pos.copy()))
    t += double_support_time

    for i in range(n_steps):
        is_left_swing = (i % 2 == 0)

        # Single support phase
        phases.append(ContactPhase(
            t, t + step_duration,
            not is_left_swing, not (not is_left_swing),
            l_pos.copy(), r_pos.copy(),
        ))
        t += step_duration

        # Double support
        phases.append(ContactPhase(
            t, t + double_support_time,
            True, True, l_pos.copy(), r_pos.copy(),
        ))
        t += double_support_time

    return phases


def generate_walking_plan(
    n_steps: int = 6,
    step_length: float = 0.15,
    step_duration: float = 0.8,
    double_support_time: float = 0.2,
    lateral_offset: float = 0.093,
) -> list[ContactPhase]:
    """Generate forward walking contact plan."""
    phases = []
    t = 0.0
    l_x, r_x = 0.0, 0.0

    # Initial double support
    l_pos = np.array([l_x, lateral_offset, 0.0])
    r_pos = np.array([r_x, -lateral_offset, 0.0])
    phases.append(ContactPhase(t, t + double_support_time,
                               True, True, l_pos.copy(), r_pos.copy()))
    t += double_support_time

    for i in range(n_steps):
        is_left_swing = (i % 2 == 0)

        if is_left_swing:
            l_x += step_length
            l_pos = np.array([l_x, lateral_offset, 0.0])
        else:
            r_x += step_length
            r_pos = np.array([r_x, -lateral_offset, 0.0])

        # Single support
        phases.append(ContactPhase(
            t, t + step_duration,
            not is_left_swing, is_left_swing,
            l_pos.copy(), r_pos.copy(),
        ))
        t += step_duration

        # Double support
        phases.append(ContactPhase(
            t, t + double_support_time,
            True, True, l_pos.copy(), r_pos.copy(),
        ))
        t += double_support_time

    return phases


def get_current_phase(phases: list[ContactPhase], t: float) -> ContactPhase:
    """Get the active contact phase at time t."""
    for phase in phases:
        if phase.t_start <= t < phase.t_end:
            return phase
    return phases[-1]  # Default to last phase
