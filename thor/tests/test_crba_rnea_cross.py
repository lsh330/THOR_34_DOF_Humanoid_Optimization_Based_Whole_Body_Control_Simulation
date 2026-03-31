"""
Comprehensive CRBA-RNEA cross-validation at multiple configurations.

The fundamental identity: M(q)*ddq + h(q,dq) = RNEA(q, dq, ddq)
must hold for ANY q, dq, ddq. This test verifies it at 10 random
configurations to ensure the algorithms are consistent.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.crba import crba
from thor.dynamics.rnea import rnea, bias_forces


@pytest.fixture
def model():
    return RobotModel()


class TestCRBA_RNEA_MultiConfig:
    """Cross-validate CRBA and RNEA at 10 random configurations."""

    @pytest.mark.parametrize("seed", range(10))
    def test_consistency_random_config(self, model, seed):
        """M*ddq + h = RNEA(q, dq, ddq) at random config."""
        rng = np.random.default_rng(seed)

        q = default_standing_config(model)
        # Perturb joints randomly
        q[7:] += rng.normal(0, 0.2, model.n_bodies - 1)

        dq = rng.normal(0, 0.5, model.n_dof)
        ddq = rng.normal(0, 1.0, model.n_dof)

        M = crba(model, q)
        h = bias_forces(model, q, dq)

        tau_crba = M @ ddq + h
        tau_rnea = rnea(model, q, dq, ddq)

        np.testing.assert_allclose(
            tau_crba, tau_rnea, atol=1e-4,
            err_msg=f"CRBA-RNEA mismatch at seed={seed}")

    def test_mass_matrix_spd_random(self, model):
        """Mass matrix SPD at 5 random configs."""
        for seed in range(5):
            rng = np.random.default_rng(42 + seed)
            q = default_standing_config(model)
            q[7:] += rng.normal(0, 0.3, model.n_bodies - 1)
            M = crba(model, q)
            np.testing.assert_allclose(M, M.T, atol=1e-8)
            eigs = np.linalg.eigvalsh(M)
            assert eigs.min() > -1e-6, f"M not PSD at seed={seed}"
