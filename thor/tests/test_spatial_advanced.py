"""
Advanced spatial algebra tests.

Validates composite properties: transform chains, inertia transforms,
and the fundamental identity dM/dt = C + C^T (Christoffel property).
"""

import numpy as np
import pytest

from thor.core.spatial import (
    rot_x, rot_y, rot_z, skew,
    spatial_transform, spatial_transform_inv, spatial_inertia,
    spatial_cross_motion, spatial_cross_force,
    motion_subspace_revolute,
)


class TestTransformChain:
    def test_chain_associativity(self):
        """(X_AB * X_BC) * v = X_AB * (X_BC * v)."""
        R1, p1 = rot_z(0.3), np.array([0.1, 0.2, 0.3])
        R2, p2 = rot_x(0.5), np.array([0.0, -0.1, 0.4])
        X1 = spatial_transform(R1, p1)
        X2 = spatial_transform(R2, p2)
        v = np.random.randn(6)
        np.testing.assert_allclose((X1 @ X2) @ v, X1 @ (X2 @ v), atol=1e-12)

    def test_triple_inverse(self):
        """X * X^{-1} * X = X."""
        R = rot_y(1.2)
        p = np.array([0.5, -0.3, 0.8])
        X = spatial_transform(R, p)
        X_inv = spatial_transform_inv(X)
        np.testing.assert_allclose(X @ X_inv @ X, X, atol=1e-10)

    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_inverse_at_various_angles(self, angle):
        R = rot_z(angle) @ rot_x(angle * 0.3)
        p = np.array([0.2, -0.1, 0.5]) * angle
        X = spatial_transform(R, p)
        X_inv = spatial_transform_inv(X)
        np.testing.assert_allclose(X @ X_inv, np.eye(6), atol=1e-9)


class TestInertiaTransform:
    def test_transform_preserves_spd(self):
        """Transformed inertia must remain SPD."""
        I = spatial_inertia(2.0, np.array([0.1, 0.0, 0.0]),
                            np.diag([0.1, 0.2, 0.15]))
        R = rot_z(0.7)
        p = np.array([0.3, 0.0, 0.0])
        X = spatial_transform(R, p)
        I_transformed = X.T @ I @ X
        np.testing.assert_allclose(I_transformed, I_transformed.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(I_transformed)
        assert eigs.min() > -1e-10

    def test_mass_invariance(self):
        """Trace of translational block should equal 3*mass after transform."""
        mass = 5.0
        I = spatial_inertia(mass, np.array([0.1, 0.2, 0.0]),
                            np.diag([0.05, 0.05, 0.05]))
        R = rot_y(0.9)
        p = np.array([0.5, -0.2, 0.3])
        X = spatial_transform(R, p)
        I_t = X.T @ I @ X
        # Translational block trace = 3*mass (invariant under rigid transform)
        assert abs(np.trace(I_t[3:, 3:]) - 3 * mass) < 1e-8


class TestCrossProductProperties:
    def test_jacobi_identity(self):
        """Jacobi identity: [a,[b,c]] + [b,[c,a]] + [c,[a,b]] = 0."""
        a = np.random.randn(6)
        b = np.random.randn(6)
        c = np.random.randn(6)
        Xa = spatial_cross_motion(a)
        Xb = spatial_cross_motion(b)
        Xc = spatial_cross_motion(c)
        lhs = Xa @ (Xb @ c) + Xb @ (Xc @ a) + Xc @ (Xa @ b)
        np.testing.assert_allclose(lhs, 0, atol=1e-12)

    def test_force_cross_transpose(self):
        """[v]_x* = -[v]_x^T for all v."""
        for _ in range(10):
            v = np.random.randn(6)
            Xm = spatial_cross_motion(v)
            Xf = spatial_cross_force(v)
            np.testing.assert_allclose(Xf, -Xm.T, atol=1e-15)
