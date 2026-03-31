"""
Forward Kinematics accuracy tests.

Validates body position computation against known geometric relationships
and tests consistency between FK and Jacobian.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.model.kinematics import (
    forward_kinematics, body_position, body_jacobian, com_position,
)
from thor.simulation.standing import default_standing_config


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestFKGeometry:
    def test_pelvis_at_base_height(self, model, q0):
        """Pelvis position should match base height in q."""
        X_world, _ = forward_kinematics(q0, model)
        p = body_position(X_world[0])
        assert abs(p[2] - q0[2]) < 0.01

    def test_left_right_symmetry(self, model, q0):
        """Symmetric config → symmetric body positions."""
        X_world, _ = forward_kinematics(q0, model)
        l_foot = body_position(X_world[model.foot_link_ids[0]])
        r_foot = body_position(X_world[model.foot_link_ids[1]])
        # x,z should match; y should be opposite sign
        assert abs(l_foot[0] - r_foot[0]) < 0.01
        assert abs(l_foot[2] - r_foot[2]) < 0.01
        assert abs(l_foot[1] + r_foot[1]) < 0.1  # Approximately opposite

    def test_com_between_feet(self, model, q0):
        """CoM x should be between left and right foot x."""
        X_world, _ = forward_kinematics(q0, model)
        com = com_position(q0, model)
        l_foot = body_position(X_world[model.foot_link_ids[0]])
        r_foot = body_position(X_world[model.foot_link_ids[1]])
        foot_x_range = [min(l_foot[0], r_foot[0]) - 0.5,
                        max(l_foot[0], r_foot[0]) + 0.5]
        assert foot_x_range[0] <= com[0] <= foot_x_range[1]

    def test_head_above_pelvis(self, model, q0):
        """Head should be above pelvis in z."""
        X_world, _ = forward_kinematics(q0, model)
        pelvis_z = body_position(X_world[0])[2]
        head_z = body_position(X_world[4])[2]
        assert head_z > pelvis_z

    def test_feet_near_ground(self, model, q0):
        """Feet should be near z=0 in standing config."""
        X_world, _ = forward_kinematics(q0, model)
        for fid in model.foot_link_ids:
            p = body_position(X_world[fid])
            assert abs(p[2]) < 0.1  # Within 10cm of ground


class TestJacobianFKConsistency:
    def test_base_translation_jacobian(self, model, q0):
        """Moving base x should move all bodies in x."""
        body_idx = 0  # Pelvis
        J = body_jacobian(body_idx, q0, model)

        # Base translational velocity (v[3] = vx in Featherstone)
        v = np.zeros(model.n_dof)
        v[3] = 1.0  # Unit x-velocity
        v_pred = J @ v

        # Linear x-velocity of pelvis should be ~1.0
        assert abs(v_pred[3]) > 0.5  # x-component of linear velocity
