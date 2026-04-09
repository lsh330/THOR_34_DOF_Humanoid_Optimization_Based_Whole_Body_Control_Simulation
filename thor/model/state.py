"""Robot state representation with named field access.

Provides index-slice namespaces (QIndex, VIndex) for the configuration
and velocity vectors, and a RobotState dataclass that wraps (q, v, t)
with convenience properties for common sub-vectors.

Configuration vector q  --  dimension 41
    [0:3]   World position [m]
    [3:7]   Base quaternion [w, x, y, z]
    [7:41]  Joint angles [rad]

Velocity vector v  --  dimension 40  (body-frame, Featherstone convention)
    [0:3]   Angular velocity of base [rad/s]
    [3:6]   Linear velocity of base [m/s]
    [6:40]  Joint velocities [rad/s]

Joint sub-vector ordering (matches robot_model.py DOF mapping):
    waist (2)  |  head (2)  |  l_arm (7)  |  r_arm (7)
    l_leg (6)  |  r_leg (6)  |  l_grip (2)  |  r_grip (2)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Index namespaces
# ---------------------------------------------------------------------------

class QIndex:
    """Named slices into configuration vector q (dim 41)."""
    BASE_POS  = slice(0, 3)    # World position [m]
    BASE_QUAT = slice(3, 7)    # Quaternion [w, x, y, z]
    JOINTS    = slice(7, 41)   # All joint angles [rad]
    WAIST     = slice(7, 9)    # Waist [yaw, pitch]
    HEAD      = slice(9, 11)   # Head [yaw, pitch]
    L_ARM     = slice(11, 18)  # Left arm (7 DOF)
    R_ARM     = slice(18, 25)  # Right arm (7 DOF)
    L_LEG     = slice(25, 31)  # Left leg (6 DOF)
    R_LEG     = slice(31, 37)  # Right leg (6 DOF)
    L_GRIP    = slice(37, 39)  # Left gripper (2 DOF)
    R_GRIP    = slice(39, 41)  # Right gripper (2 DOF)


class VIndex:
    """Named slices into velocity vector v (dim 40)."""
    BASE_ANG = slice(0, 3)    # Angular velocity [rad/s]
    BASE_LIN = slice(3, 6)    # Linear velocity [m/s]
    JOINTS   = slice(6, 40)   # All joint velocities [rad/s]
    WAIST    = slice(6, 8)
    HEAD     = slice(8, 10)
    L_ARM    = slice(10, 17)
    R_ARM    = slice(17, 24)
    L_LEG    = slice(24, 30)
    R_LEG    = slice(30, 36)
    L_GRIP   = slice(36, 38)
    R_GRIP   = slice(38, 40)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RobotState:
    """Full robot state with named field access.

    Attributes
    ----------
    q : NDArray[float64], shape (41,)
        Configuration vector: 3 base position + 4 quaternion + 34 joints.
    v : NDArray[float64], shape (40,)
        Velocity vector: 6 base twist (angular first) + 34 joint rates.
    t : float
        Simulation time [s].
    """
    q: NDArray
    v: NDArray
    t: float = 0.0

    # --- Base ---

    @property
    def base_position(self) -> NDArray:
        """World-frame base position [m], shape (3,)."""
        return self.q[QIndex.BASE_POS]

    @property
    def base_quaternion(self) -> NDArray:
        """Base orientation as quaternion [w, x, y, z], shape (4,)."""
        return self.q[QIndex.BASE_QUAT]

    @property
    def joint_positions(self) -> NDArray:
        """All 34 joint angles [rad], shape (34,)."""
        return self.q[QIndex.JOINTS]

    @property
    def base_angular_velocity(self) -> NDArray:
        """Body-frame angular velocity [rad/s], shape (3,)."""
        return self.v[VIndex.BASE_ANG]

    @property
    def base_linear_velocity(self) -> NDArray:
        """Body-frame linear velocity [m/s], shape (3,)."""
        return self.v[VIndex.BASE_LIN]

    @property
    def joint_velocities(self) -> NDArray:
        """All 34 joint velocities [rad/s], shape (34,)."""
        return self.v[VIndex.JOINTS]

    # --- Limb shortcuts ---

    @property
    def left_leg_q(self) -> NDArray:
        """Left leg joint angles [hip_y, hip_r, hip_p, kn_p, an_p, an_r] [rad]."""
        return self.q[QIndex.L_LEG]

    @property
    def right_leg_q(self) -> NDArray:
        """Right leg joint angles [hip_y, hip_r, hip_p, kn_p, an_p, an_r] [rad]."""
        return self.q[QIndex.R_LEG]

    @property
    def left_arm_q(self) -> NDArray:
        """Left arm joint angles (7 DOF) [rad]."""
        return self.q[QIndex.L_ARM]

    @property
    def right_arm_q(self) -> NDArray:
        """Right arm joint angles (7 DOF) [rad]."""
        return self.q[QIndex.R_ARM]

    @property
    def left_leg_v(self) -> NDArray:
        """Left leg joint velocities [rad/s]."""
        return self.v[VIndex.L_LEG]

    @property
    def right_leg_v(self) -> NDArray:
        """Right leg joint velocities [rad/s]."""
        return self.v[VIndex.R_LEG]

    # --- Factory methods ---

    @classmethod
    def from_default_standing(cls, base_height: float = 1.02) -> "RobotState":
        """Create default standing configuration.

        Sets quaternion w=1 (identity orientation) and applies nominal
        knee-bend angles so the robot is balanced at *base_height* [m].

        Joint layout within L_LEG / R_LEG (offset from slice start):
            0: hip_y   1: hip_r   2: hip_p
            3: kn_p    4: an_p    5: an_r

        Args:
            base_height: Pelvis height above ground [m].

        Returns:
            RobotState at nominal standing posture, zero velocity.
        """
        q = np.zeros(41, dtype=np.float64)
        q[2] = base_height        # Base z position [m]
        q[3] = 1.0                # Quaternion w = 1 (identity)

        # Nominal leg bend: hip_p = -0.3 rad, kn_p = 0.6 rad, an_p = -0.3 rad
        L0 = QIndex.L_LEG.start   # = 25
        R0 = QIndex.R_LEG.start   # = 31
        q[L0 + 2] = -0.3          # l_hip_p
        q[L0 + 3] =  0.6          # l_kn_p
        q[L0 + 4] = -0.3          # l_an_p
        q[R0 + 2] = -0.3          # r_hip_p
        q[R0 + 3] =  0.6          # r_kn_p
        q[R0 + 4] = -0.3          # r_an_p

        v = np.zeros(40, dtype=np.float64)
        return cls(q=q, v=v, t=0.0)

    @classmethod
    def zeros(cls) -> "RobotState":
        """Create an all-zero state (useful as a placeholder)."""
        q = np.zeros(41, dtype=np.float64)
        q[3] = 1.0  # Valid unit quaternion
        return cls(q=q, v=np.zeros(40, dtype=np.float64), t=0.0)
