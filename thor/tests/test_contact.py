"""
Tests for contact dynamics (spring-damper and contact-implicit).
"""

import numpy as np
import pytest

from thor.dynamics.contact import contact_force_single, CONTACT_STIFFNESS as CONTACT_K, CONTACT_DAMPING as CONTACT_D


class TestSpringDamperContact:
    def test_no_contact_above_ground(self):
        """No force when above ground."""
        pos = np.array([0.0, 0.0, 0.1])  # Above ground
        vel = np.zeros(3)
        f = contact_force_single(pos, vel)
        np.testing.assert_allclose(f, 0, atol=1e-15)

    def test_normal_force_proportional(self):
        """Normal force proportional to penetration depth."""
        for depth in [0.01, 0.02, 0.05]:
            pos = np.array([0.0, 0.0, -depth])
            vel = np.zeros(3)
            f = contact_force_single(pos, vel)
            expected_fz = CONTACT_K * depth
            assert abs(f[2] - min(expected_fz, 3000.0)) < 1.0

    def test_damping_force(self):
        """Damping adds force when moving into ground."""
        pos = np.array([0.0, 0.0, -0.01])
        vel_into = np.array([0.0, 0.0, -1.0])  # Moving downward
        vel_away = np.array([0.0, 0.0, 1.0])   # Moving upward

        f_into = contact_force_single(pos, vel_into)
        f_away = contact_force_single(pos, vel_away)

        assert f_into[2] > f_away[2]  # More force when pressing in

    def test_friction_opposes_motion(self):
        """Tangential force opposes sliding velocity."""
        pos = np.array([0.0, 0.0, -0.01])
        vel = np.array([1.0, 0.0, 0.0])  # Sliding in +x
        f = contact_force_single(pos, vel)
        assert f[0] < 0  # Friction in -x direction

    def test_no_adhesion(self):
        """Normal force is always >= 0 (no sticking to ground)."""
        pos = np.array([0.0, 0.0, -0.001])
        vel = np.array([0.0, 0.0, 10.0])  # Moving away fast
        f = contact_force_single(pos, vel)
        assert f[2] >= 0


class TestContactImplicit:
    def test_standing_stability(self):
        """CI time-stepping should maintain standing for 100 steps."""
        from thor.model.robot_model import RobotModel
        from thor.simulation.standing import default_standing_config
        from thor.dynamics.contact_implicit import contact_implicit_step
        from thor.model.kinematics import com_position

        model = RobotModel()
        q = default_standing_config(model)
        v = np.zeros(model.n_dof)

        from thor.control.joint_pd import JointPDController
        pd = JointPDController(model, q)

        com_initial = com_position(q, model)[2]

        for _ in range(100):
            tau = pd.compute(q, v, 0.0)
            q, v, _, _ = contact_implicit_step(model, q, v, tau, 0.002)

        com_final = com_position(q, model)[2]
        assert abs(com_final - com_initial) < 0.2  # Less than 20cm drift (100 steps only)
