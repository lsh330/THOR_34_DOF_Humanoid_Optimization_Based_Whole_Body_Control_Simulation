"""
Advanced centroidal dynamics tests.

Validates the Centroidal Momentum Matrix against known physical laws.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.centroidal import centroidal_momentum_matrix, centroidal_momentum
from thor.dynamics.crba import crba
from thor.model.kinematics import com_position


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestCentroidalMomentum:
    def test_linear_momentum_at_rest(self, model, q0):
        """At zero velocity, centroidal momentum should be zero."""
        v = np.zeros(model.n_dof)
        h_G = centroidal_momentum(q0, v, model)
        np.testing.assert_allclose(h_G, 0, atol=1e-10)

    def test_linear_momentum_pure_translation(self, model, q0):
        """Pure base translation → linear momentum = m * v_base."""
        v = np.zeros(model.n_dof)
        v[5] = 1.0  # Base vz = 1 m/s (Featherstone: v[3:6] = linear)
        h_G = centroidal_momentum(q0, v, model)
        # Linear momentum (last 3) should be ~m * [0,0,1]
        expected_lz = model.total_mass * 1.0
        assert abs(h_G[5] - expected_lz) / expected_lz < 0.5

    def test_cmm_shape(self, model, q0):
        """CMM should be 6 x n_dof."""
        A_G = centroidal_momentum_matrix(q0, model)
        assert A_G.shape == (6, model.n_dof)

    def test_cmm_rank(self, model, q0):
        """CMM should have rank 6 (full row rank) for a humanoid."""
        A_G = centroidal_momentum_matrix(q0, model)
        rank = np.linalg.matrix_rank(A_G, tol=1e-6)
        assert rank == 6

    def test_momentum_direction_consistency(self, model, q0):
        """Positive base vz should produce positive linear momentum z."""
        v = np.zeros(model.n_dof)
        v[5] = 0.5
        h_G = centroidal_momentum(q0, v, model)
        assert h_G[5] > 0  # Positive z-momentum for positive z-velocity
