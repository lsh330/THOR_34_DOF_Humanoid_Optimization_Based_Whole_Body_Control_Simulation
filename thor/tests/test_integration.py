"""
Integration and simulation correctness tests.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.model.quaternion import quat_to_rot, quat_integrate, quat_identity
from thor.simulation.standing import default_standing_config


class TestQuaternionIntegration:
    def test_constant_rotation(self):
        """Constant angular velocity should produce uniform rotation."""
        q = quat_identity()
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s about z
        dt = 0.001
        for _ in range(1000):
            q = quat_integrate(q, omega, dt)
        # After 1s at 1 rad/s, total angle = 1 rad about z
        R = quat_to_rot(q)
        # R should rotate x-axis toward y-axis by ~1 rad
        angle = np.arccos(np.clip(R[0, 0], -1, 1))
        assert abs(angle - 1.0) < 0.05  # Within 5%

    def test_unit_norm_preserved(self):
        """Quaternion norm should stay 1 after many integrations."""
        q = quat_identity()
        omega = np.array([0.5, -0.3, 0.8])
        for _ in range(10000):
            q = quat_integrate(q, omega, 0.001)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


class TestStandingConfig:
    def test_knee_bend(self):
        """Standing config should have knee bend."""
        model = RobotModel()
        q = default_standing_config(model)
        l_knee = q[7 + 21]  # l_kn_p
        r_knee = q[7 + 27]  # r_kn_p
        assert l_knee > 0.3  # Positive = flexion
        assert r_knee > 0.3

    def test_symmetric(self):
        """Left and right leg joints should be symmetric."""
        model = RobotModel()
        q = default_standing_config(model)
        for l_idx, r_idx in [(20, 26), (21, 27), (22, 28)]:
            assert abs(q[7+l_idx] - q[7+r_idx]) < 1e-10

    def test_base_quaternion_unit(self):
        model = RobotModel()
        q = default_standing_config(model)
        assert abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-15
