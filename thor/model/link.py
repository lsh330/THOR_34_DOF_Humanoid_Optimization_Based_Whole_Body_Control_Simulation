"""
Link (rigid body) data structure.

Separated from robot_model.py for single responsibility.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .joint_types import JointType


@dataclass(slots=True)
class LinkData:
    """Single rigid body (link) in the kinematic tree."""
    name: str
    mass: float                          # [kg]
    com: NDArray = field(default_factory=lambda: np.zeros(3))
    inertia: NDArray = field(default_factory=lambda: np.eye(3) * 0.001)
    parent_id: int = -1
    joint_type: JointType = JointType.FIXED
    joint_axis: int = 2
    joint_offset: NDArray = field(default_factory=lambda: np.zeros(3))
    joint_rotation: NDArray = field(default_factory=lambda: np.eye(3))
    q_min: float = -np.pi
    q_max: float = np.pi
    tau_max: float = 100.0
    is_actuated: bool = True
