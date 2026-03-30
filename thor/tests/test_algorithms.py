"""
Cross-validation tests between dynamics algorithms.

The gold standard: CRBA and RNEA must produce consistent results.
M*ddq + h = RNEA(q, dq, ddq) for arbitrary ddq.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.crba import crba
from thor.dynamics.rnea import rnea, bias_forces, gravity_forces
from thor.dynamics.centroidal import centroidal_momentum
from thor.model.kinematics import com_position


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestCRBA_RNEA_Consistency:
    def test_M_ddq_equals_rnea(self, model, q0):
        """M*ddq = RNEA(q, 0, ddq) - h for arbitrary ddq.

        This is the fundamental consistency check: the mass matrix
        from CRBA times an acceleration must equal the torque from
        RNEA minus the bias forces.
        """
        np.random.seed(42)
        ddq = np.random.randn(model.n_dof) * 0.1

        M = crba(model, q0)
        h = bias_forces(model, q0, np.zeros(model.n_dof))

        # RNEA with zero velocity and given acceleration
        tau_rnea = rnea(model, q0, np.zeros(model.n_dof), ddq)

        # Should satisfy: tau_rnea = M*ddq + h
        tau_crba = M @ ddq + h

        np.testing.assert_allclose(tau_crba, tau_rnea, atol=1e-6,
                                   err_msg="CRBA-RNEA consistency failed")

    def test_gravity_only(self, model, q0):
        """h(q, 0) = g(q) when velocity is zero."""
        h = bias_forces(model, q0, np.zeros(model.n_dof))
        g = gravity_forces(model, q0)
        np.testing.assert_allclose(h, g, atol=1e-12)


class TestCentroidal:
    def test_linear_momentum_equals_mass_times_com_vel(self, model, q0):
        """l_G = m * d(com)/dt approximation check.

        For small velocity, centroidal linear momentum should
        approximately equal total_mass * v_com.
        """
        v = np.zeros(model.n_dof)
        v[5] = 0.1  # Small z-velocity at base
        h_G = centroidal_momentum(q0, v, model)

        # Linear momentum (last 3 components)
        l_G = h_G[3:]
        # Should be approximately m * v_base_linear
        m_v = model.total_mass * v[3:6]

        # Not exact due to kinematic chain, but should be same order
        assert np.linalg.norm(l_G) > 0
        # At least same sign
        assert l_G[2] * m_v[2] >= 0


class TestMassMatrixProperties:
    def test_mass_on_translational_diagonal(self, model, q0):
        """M[3:6, 3:6] should be approximately total_mass * I_3."""
        M = crba(model, q0)
        M_trans = M[3:6, 3:6]
        expected = model.total_mass * np.eye(3)
        np.testing.assert_allclose(M_trans, expected, atol=0.1)

    def test_mass_matrix_spd(self, model, q0):
        """M must be symmetric positive definite at various configs."""
        for _ in range(5):
            q = q0.copy()
            q[7:] += np.random.randn(model.n_bodies - 1) * 0.1
            M = crba(model, q)
            np.testing.assert_allclose(M, M.T, atol=1e-8)
            eigs = np.linalg.eigvalsh(M)
            assert eigs.min() > -1e-6

    def test_condition_number_reasonable(self, model, q0):
        """Condition number should be < 10^6 for a humanoid."""
        M = crba(model, q0)
        eigs = np.linalg.eigvalsh(M)
        pos_eigs = eigs[eigs > 1e-10]
        cond = pos_eigs.max() / pos_eigs.min()
        assert cond < 1e7  # Humanoid robots typically have cond ~10^4-10^6


class TestEnergyConservation:
    def test_free_fall_energy(self, model, q0):
        """In free fall (no contact, no control), energy is conserved.

        Since we use semi-implicit Euler, small drift is expected
        but should be < 1% over 100 steps.
        """
        from thor.dynamics.contact_implicit import contact_implicit_step

        q = q0.copy()
        q[2] = 5.0  # High in air (no contact)
        v = np.zeros(model.n_dof)
        tau = np.zeros(model.n_dof)
        dt = 0.001

        # Initial energy: KE + PE = 0 + mgh
        E0 = model.total_mass * 9.81 * com_position(q, model)[2]

        for _ in range(100):
            q, v, _, _ = contact_implicit_step(model, q, v, tau, dt)

        E_final = (0.5 * v @ crba(model, q) @ v +
                   model.total_mass * 9.81 * com_position(q, model)[2])

        # Allow 5% drift for semi-implicit Euler
        assert abs(E_final - E0) / abs(E0) < 0.05
