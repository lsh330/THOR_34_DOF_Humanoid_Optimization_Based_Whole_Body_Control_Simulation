"""
Joint type definitions for the THOR humanoid robot.

Supports revolute joints (all 34 DOF of THOR are revolute)
and the 6-DOF floating base.
"""

from enum import IntEnum, auto


class JointType(IntEnum):
    """Joint types supported by the dynamics engine."""
    FIXED = 0
    REVOLUTE_X = 1   # Rotation about local x-axis
    REVOLUTE_Y = 2   # Rotation about local y-axis
    REVOLUTE_Z = 3   # Rotation about local z-axis
    FLOATING = 4     # 6-DOF floating base (special)
