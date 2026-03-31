"""
Computational performance benchmark tests.

Validates that dynamics algorithms meet real-time requirements
for a 40-DOF humanoid control system.
"""

import time
import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.crba import crba
from thor.dynamics.rnea import rnea, bias_forces
from thor.optimization.lcp_solver import solve_lcp_fb_newton


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestCRBAPerformance:
    def test_crba_under_50ms(self, model, q0):
        """CRBA (40x40 mass matrix) should complete in < 50ms."""
        # Warm up
        crba(model, q0)

        t0 = time.perf_counter()
        for _ in range(10):
            crba(model, q0)
        avg = (time.perf_counter() - t0) / 10

        assert avg < 0.05, f"CRBA took {avg*1000:.1f}ms (limit: 50ms)"


class TestRNEAPerformance:
    def test_rnea_under_50ms(self, model, q0):
        """RNEA (inverse dynamics) should complete in < 50ms."""
        v = np.zeros(model.n_dof)
        a = np.zeros(model.n_dof)

        rnea(model, q0, v, a)

        t0 = time.perf_counter()
        for _ in range(10):
            rnea(model, q0, v, a)
        avg = (time.perf_counter() - t0) / 10

        assert avg < 0.05, f"RNEA took {avg*1000:.1f}ms (limit: 50ms)"


class TestLCPPerformance:
    def test_lcp_2x2_under_5ms(self):
        """2x2 LCP (typical contact) should solve in < 5ms."""
        M = np.array([[0.05, 0.01], [0.01, 0.05]])
        q = np.array([-0.5, -0.3])

        solve_lcp_fb_newton(M, q)

        t0 = time.perf_counter()
        for _ in range(100):
            solve_lcp_fb_newton(M, q)
        avg = (time.perf_counter() - t0) / 100

        assert avg < 0.005, f"LCP took {avg*1000:.2f}ms (limit: 5ms)"


class TestSimulationStepPerformance:
    def test_step_under_20ms(self, model, q0):
        """Full simulation step should complete in < 20ms."""
        from thor.dynamics.contact_implicit import contact_implicit_step
        from thor.control.joint_pd import JointPDController

        pd = JointPDController(model, q0)
        v = np.zeros(model.n_dof)
        tau = pd.compute(q0, v, 0.0)

        # Warm up
        contact_implicit_step(model, q0, v, tau, 0.002)

        t0 = time.perf_counter()
        for _ in range(10):
            contact_implicit_step(model, q0, v, tau, 0.002)
        avg = (time.perf_counter() - t0) / 10

        assert avg < 0.02, f"Step took {avg*1000:.1f}ms (limit: 20ms)"
