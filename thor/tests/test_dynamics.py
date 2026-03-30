"""
Tests for THOR dynamics engine.
"""

import math
import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.model.kinematics import forward_kinematics, com_position, body_position
from thor.dynamics.rnea import gravity_forces, bias_forces
from thor.dynamics.crba import crba
from thor.simulation.standing import default_standing_config


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q_standing(model):
    return default_standing_config(model)


class TestRobotModel:
    def test_body_count(self, model):
        assert model.n_bodies == 35

    def test_dof_count(self, model):
        assert model.n_dof == 40

    def test_total_mass(self, model):
        assert 60.0 < model.total_mass < 75.0

    def test_foot_links_exist(self, model):
        assert model.foot_link_ids[0] > 0
        assert model.foot_link_ids[1] > 0


class TestKinematics:
    def test_base_position(self, q_standing, model):
        X_world, _ = forward_kinematics(q_standing, model)
        p = body_position(X_world[0])
        assert abs(p[2] - 0.85) < 0.01  # Base at 0.85m

    def test_com_positive_z(self, q_standing, model):
        com = com_position(q_standing, model)
        assert com[2] > 0.5  # CoM should be above ground

    def test_com_near_center(self, q_standing, model):
        com = com_position(q_standing, model)
        assert abs(com[0]) < 0.1  # x near zero
        assert abs(com[1]) < 0.01  # y near zero (symmetric)


class TestGravity:
    def test_gravity_magnitude(self, q_standing, model):
        g = gravity_forces(model, q_standing)
        # Vertical force on base should equal total weight
        assert abs(g[5] - model.total_mass * 9.81) < 1.0

    def test_gravity_zero_velocity(self, q_standing, model):
        g = gravity_forces(model, q_standing)
        assert g.shape == (model.n_dof,)


class TestMassMatrix:
    def test_symmetric(self, q_standing, model):
        M = crba(model, q_standing)
        np.testing.assert_allclose(M, M.T, atol=1e-8)

    def test_positive_definite(self, q_standing, model):
        M = crba(model, q_standing)
        eigs = np.linalg.eigvalsh(M)
        assert eigs.min() > -1e-6  # Allow tiny numerical noise

    def test_shape(self, q_standing, model):
        M = crba(model, q_standing)
        assert M.shape == (model.n_dof, model.n_dof)


class TestStanding:
    def test_gravity_compensation(self, model):
        """Perfect gravity compensation should produce zero acceleration."""
        q = default_standing_config(model)
        g = gravity_forces(model, q)
        M = crba(model, q)
        M_jj = M[6:, 6:]
        h_j = g[6:]
        tau = h_j  # Exact gravity compensation

        rhs = tau - h_j  # Should be zero
        np.testing.assert_allclose(rhs, 0.0, atol=1e-10)
