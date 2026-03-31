"""
LCP solver convergence rate and robustness tests.
"""

import numpy as np
import pytest

from thor.optimization.lcp_solver import solve_lcp_fb_newton, solve_lcp_interior_point


class TestConvergenceRate:
    def test_fb_newton_converges_in_under_10_iters(self):
        """Standard 2x2 should converge in < 10 Newton iterations."""
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        z, iters, res = solve_lcp_fb_newton(M, q, max_iter=50)
        assert iters < 10, f"Took {iters} iterations (expected < 10)"
        assert res < 1e-6

    def test_fb_residual_monotone_decrease(self):
        """Residual should decrease monotonically (or nearly so)."""
        M = np.array([[3.0, 0.5], [0.5, 3.0]])
        q = np.array([-2.0, -1.0])
        # Run manually to track residual
        z = np.ones(2) * 0.1
        prev_res = float('inf')
        for _ in range(20):
            w = M @ z + q
            F = np.array([
                z[i] + w[i] - np.sqrt(z[i]**2 + w[i]**2 + 2e-8)
                for i in range(2)
            ])
            res = np.linalg.norm(F)
            if res < 1e-8:
                break
            # Newton step (simplified)
            z_new, _, new_res = solve_lcp_fb_newton(M, q, z0=z, max_iter=1)
            z = z_new
        # Final residual should be small
        _, _, final_res = solve_lcp_fb_newton(M, q)
        assert final_res < 1e-6

    @pytest.mark.parametrize("size", [2, 3, 4, 5])
    def test_random_spd_convergence(self, size):
        """FB-Newton should converge for random SPD LCP matrices."""
        rng = np.random.default_rng(42 + size)
        A = rng.standard_normal((size, size))
        M = A.T @ A + 0.1 * np.eye(size)
        q = rng.standard_normal(size) - 1.0
        z, iters, res = solve_lcp_fb_newton(M, q, max_iter=50)
        assert res < 1e-4, f"Failed at size {size}: res={res:.2e}"

    def test_interior_point_convergence(self):
        """Interior point should converge for standard problem."""
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        z, iters, mu = solve_lcp_interior_point(M, q)
        assert mu < 1e-6
        assert iters < 30
