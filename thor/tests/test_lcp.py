"""
Tests for LCP solver correctness.

Validates against known analytical solutions and benchmark problems.
"""

import numpy as np
import pytest

from thor.optimization.lcp_solver import solve_lcp_fb_newton, solve_lcp_interior_point


class TestLCPBasic:
    def test_trivial_solution(self):
        """q >= 0 implies z = 0 is the solution."""
        M = np.eye(2)
        q = np.array([1.0, 2.0])
        z, iters, res = solve_lcp_fb_newton(M, q)
        np.testing.assert_allclose(z, 0, atol=1e-6)

    def test_simple_2x2(self):
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        z, iters, res = solve_lcp_fb_newton(M, q)
        assert res < 1e-6
        w = M @ z + q
        assert np.all(z >= -1e-8)
        assert np.all(w >= -1e-8)
        assert abs(np.dot(z, w)) < 1e-6

    def test_complementarity(self):
        """z_i * w_i = 0 for all i."""
        M = np.array([[3.0, 0.5], [0.5, 3.0]])
        q = np.array([-2.0, -1.0])
        z, _, _ = solve_lcp_fb_newton(M, q)
        w = M @ z + q
        for i in range(len(z)):
            assert abs(z[i] * w[i]) < 1e-5

    def test_contact_like(self):
        """Delassus-matrix style LCP (SPD matrix)."""
        M = np.array([[0.05, 0.01], [0.01, 0.05]])
        q = np.array([-0.5, -0.3])
        z, iters, res = solve_lcp_fb_newton(M, q)
        assert np.all(z >= -1e-8)
        assert res < 1e-5


class TestLCPInteriorPoint:
    def test_matches_fb_newton(self):
        """Both solvers should give same answer."""
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        z_fb, _, _ = solve_lcp_fb_newton(M, q)
        z_ip, _, _ = solve_lcp_interior_point(M, q)
        np.testing.assert_allclose(z_fb, z_ip, atol=1e-4)

    def test_larger_problem(self):
        """4x4 LCP from friction cone."""
        np.random.seed(42)
        A = np.random.randn(4, 4)
        M = A.T @ A + 0.1 * np.eye(4)  # Ensure SPD
        q = np.random.randn(4) - 1
        z, iters, res = solve_lcp_interior_point(M, q)
        w = M @ z + q
        assert np.all(z >= -1e-6)
        assert np.all(w >= -1e-6)
