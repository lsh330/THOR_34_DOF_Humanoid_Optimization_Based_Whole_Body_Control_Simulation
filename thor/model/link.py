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
    joint_axis: int = -1                 # Auto-derived from joint_type in __post_init__
    joint_offset: NDArray = field(default_factory=lambda: np.zeros(3))
    joint_rotation: NDArray = field(default_factory=lambda: np.eye(3))
    q_min: float = -np.pi
    q_max: float = np.pi
    tau_max: float = 100.0
    is_actuated: bool = True

    def __post_init__(self):
        """Auto-derive joint_axis from joint_type if not explicitly set."""
        if self.joint_axis == -1:
            # REVOLUTE_X=1 → axis=0, REVOLUTE_Y=2 → axis=1, REVOLUTE_Z=3 → axis=2
            _type_to_axis = {
                JointType.REVOLUTE_X: 0,
                JointType.REVOLUTE_Y: 1,
                JointType.REVOLUTE_Z: 2,
                # Future: PRISMATIC_X: 0, PRISMATIC_Y: 1, PRISMATIC_Z: 2
            }
            self.joint_axis = _type_to_axis.get(self.joint_type, 2)
