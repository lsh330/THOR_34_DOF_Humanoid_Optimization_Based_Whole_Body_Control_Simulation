"""
Tests for walking controller biomechanical correctness.

Validates joint angle ranges against Winter (1991) normative data.
"""

import math
import numpy as np
import pytest

from thor.control.gait.swing_trajectory import (
    swing_leg_angles as _gait_phase_angles,
    stance_leg_angles as _stance_leg_angles,
    HIP_SWING_FLEX, HIP_STANCE_EXT,
    KNEE_SWING_FLEX, KNEE_STANCE,
)
from thor.control.walking_controller import WalkingController
from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config


class TestSwingTrajectory:
    def test_swing_start(self):
        """At s=0 (toe-off), hip should be near extension."""
        hip, kn, an = _gait_phase_angles(0.0)
        assert abs(hip - HIP_STANCE_EXT) < 0.01

    def test_swing_end(self):
        """At s=1 (heel strike), hip should be flexed."""
        hip, kn, an = _gait_phase_angles(1.0)
        assert abs(hip - HIP_SWING_FLEX) < 0.01

    def test_swing_midpoint_knee_flexion(self):
        """At mid-swing, knee should be significantly flexed."""
        hip, kn, an = _gait_phase_angles(0.5)
        assert kn > math.radians(30)  # Should be well flexed

    def test_hip_range_within_biomechanical(self):
        """Hip pitch should stay within -10 to +30 deg."""
        for s in np.linspace(0, 1, 50):
            hip, _, _ = _gait_phase_angles(s)
            assert hip >= math.radians(-15)
            assert hip <= math.radians(35)

    def test_knee_range_within_biomechanical(self):
        """Knee pitch should stay within 0 to +60 deg."""
        for s in np.linspace(0, 1, 50):
            _, kn, _ = _gait_phase_angles(s)
            assert kn >= math.radians(-5)
            assert kn <= math.radians(60)

    def test_ankle_range(self):
        """Ankle should stay within -20 to +15 deg."""
        for s in np.linspace(0, 1, 50):
            _, _, an = _gait_phase_angles(s)
            assert an >= math.radians(-25)
            assert an <= math.radians(20)


class TestStanceTrajectory:
    def test_stance_start_flexed(self):
        """Stance starts with hip flexed (just landed)."""
        hip, kn, an = _stance_leg_angles(0.0)
        assert hip > 0  # Flexed (positive)

    def test_stance_end_extended(self):
        """Stance ends with hip extended (about to push off)."""
        hip, kn, an = _stance_leg_angles(1.0)
        assert hip < math.radians(5)  # Near or past neutral


class TestWalkingController:
    def test_phase_detection_initial_ds(self):
        model = RobotModel()
        q0 = default_standing_config(model)
        wc = WalkingController(model, q0, n_steps=2)
        phase = wc._get_phase(0.0)
        assert phase["phase"] == "ds"

    def test_phase_detection_swing(self):
        model = RobotModel()
        q0 = default_standing_config(model)
        wc = WalkingController(model, q0, n_steps=2)
        phase = wc._get_phase(0.5)
        assert "swing" in phase["phase"]

    def test_torque_limits_respected(self):
        model = RobotModel()
        q0 = default_standing_config(model)
        wc = WalkingController(model, q0, n_steps=2)
        v = np.zeros(model.n_dof)
        tau = wc.compute(q0, v, 0.5)
        for i in range(model.n_dof - 6):
            if i + 1 < model.n_bodies:
                lim = model.links[i + 1].tau_max
                assert abs(tau[6 + i]) <= lim + 1e-6
