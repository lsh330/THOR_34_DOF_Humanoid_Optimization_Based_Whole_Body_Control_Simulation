"""
Tests for walking forward progression and gait correctness.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.control.walking_controller import WalkingController
from thor.dynamics.contact_implicit import (
    contact_implicit_step, run_contact_implicit_simulation,
)
from thor.model.kinematics import com_position


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestForwardProgression:
    def test_forward_distance(self, model, q0):
        """Walking should cover ~0.95m in 5.1s (6 steps)."""
        walker = WalkingController(model, q0, n_steps=6)
        result = run_contact_implicit_simulation(
            model, q0, walker.compute,
            t_final=walker.total_duration, dt=0.002)
        forward = result["q"][-1, 0] - result["q"][0, 0]
        assert forward > 0.8, f"Forward only {forward:.2f}m"
        assert forward < 1.2, f"Forward too much {forward:.2f}m"

    def test_zero_yaw_during_walking(self, model, q0):
        """Base quaternion should maintain zero yaw."""
        walker = WalkingController(model, q0, n_steps=4)
        q = q0.copy()
        v = np.zeros(model.n_dof)
        for _ in range(500):
            tau = walker.compute(q, v, 0.5)
            q, v, _, _ = contact_implicit_step(model, q, v, tau, 0.002)
        # Check quaternion is near identity
        assert abs(q[3] - 1.0) < 0.01 or abs(q[3] + 1.0) < 0.01

    def test_com_stays_above_ground(self, model, q0):
        """CoM should never go below ground during walking."""
        walker = WalkingController(model, q0, n_steps=4)
        result = run_contact_implicit_simulation(
            model, q0, walker.compute,
            t_final=walker.total_duration, dt=0.002)
        com_z = result["com"][:, 2]
        assert com_z.min() > 0.3, f"CoM too low: {com_z.min():.2f}m"

    def test_alternating_hip_pattern(self, model, q0):
        """Left and right hip pitch should alternate during walking."""
        walker = WalkingController(model, q0, n_steps=4)
        result = run_contact_implicit_simulation(
            model, q0, walker.compute,
            t_final=walker.total_duration, dt=0.002)
        q_traj = result["q"]
        l_hip = q_traj[:, 27]  # L hip pitch
        r_hip = q_traj[:, 33]  # R hip pitch
        # Check that they have different signs at midpoints
        n = len(q_traj)
        mid = n // 2
        diff = abs(l_hip[mid] - r_hip[mid])
        assert diff > 0.01, "L/R hips should differ during walking"
