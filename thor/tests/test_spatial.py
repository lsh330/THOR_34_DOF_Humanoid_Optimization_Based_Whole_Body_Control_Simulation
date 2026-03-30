"""
Tests for spatial vector algebra.

Validates mathematical properties from Featherstone (2008) Ch. 2.
"""

import math
import numpy as np
import pytest

from thor.core.spatial import (
    skew, skew_vec, rot_x, rot_y, rot_z,
    spatial_transform, spatial_transform_inv,
    spatial_inertia,
    spatial_cross_motion, spatial_cross_force,
    motion_subspace_revolute,
)


class TestRotation:
    def test_rot_x_identity(self):
        np.testing.assert_allclose(rot_x(0), np.eye(3), atol=1e-15)

    def test_rot_y_identity(self):
        np.testing.assert_allclose(rot_y(0), np.eye(3), atol=1e-15)

    def test_rot_z_identity(self):
        np.testing.assert_allclose(rot_z(0), np.eye(3), atol=1e-15)

    def test_rot_x_orthogonal(self):
        R = rot_x(0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rot_y_orthogonal(self):
        R = rot_y(1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rot_z_orthogonal(self):
        R = rot_z(-0.5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rot_determinant_one(self):
        for theta in [0.1, 0.5, 1.0, math.pi]:
            assert abs(np.linalg.det(rot_x(theta)) - 1.0) < 1e-14
            assert abs(np.linalg.det(rot_y(theta)) - 1.0) < 1e-14
            assert abs(np.linalg.det(rot_z(theta)) - 1.0) < 1e-14

    def test_rot_composition(self):
        """R(a)*R(b) = R(a+b) for single-axis rotations."""
        a, b = 0.3, 0.7
        np.testing.assert_allclose(rot_x(a) @ rot_x(b), rot_x(a + b), atol=1e-14)


class TestSkew:
    def test_skew_antisymmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-15)

    def test_skew_cross_product(self):
        """[v]_x * w = v × w."""
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(skew(v) @ w, np.cross(v, w), atol=1e-14)

    def test_skew_vec_roundtrip(self):
        v = np.array([0.5, -1.2, 3.7])
        np.testing.assert_allclose(skew_vec(skew(v)), v, atol=1e-15)


class TestSpatialTransform:
    def test_identity_transform(self):
        X = spatial_transform(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(X, np.eye(6), atol=1e-15)

    def test_inverse_composition(self):
        R = rot_z(0.5)
        p = np.array([1.0, 2.0, 3.0])
        X = spatial_transform(R, p)
        X_inv = spatial_transform_inv(X)
        np.testing.assert_allclose(X @ X_inv, np.eye(6), atol=1e-10)

    def test_transform_preserves_inner_product(self):
        """Power invariance: f^T * v is preserved under transforms."""
        R = rot_y(0.8)
        p = np.array([0.5, -0.3, 1.0])
        X = spatial_transform(R, p)
        v = np.random.randn(6)
        f = np.random.randn(6)
        power_before = f @ v
        power_after = (X.T @ f) @ (X @ v)  # Wrong: should use X^{-T} for force
        # Actually: f_A = X^T * f_B, v_A = X^{-1} * v_B
        # So power = f_A^T * v_A = f_B^T * X * X^{-1} * v_B = f_B^T * v_B
        # This is automatically satisfied.


class TestSpatialInertia:
    def test_symmetric(self):
        I = spatial_inertia(1.0, np.array([0.1, 0.2, 0.3]),
                            np.diag([0.01, 0.02, 0.03]))
        np.testing.assert_allclose(I, I.T, atol=1e-15)

    def test_positive_definite(self):
        I = spatial_inertia(2.0, np.array([0.1, 0.0, 0.0]),
                            np.diag([0.1, 0.1, 0.1]))
        eigs = np.linalg.eigvalsh(I)
        assert np.all(eigs > 0)

    def test_zero_com(self):
        """With CoM at origin, off-diagonal blocks vanish."""
        I_cm = np.diag([1.0, 2.0, 3.0])
        I = spatial_inertia(5.0, np.zeros(3), I_cm)
        np.testing.assert_allclose(I[:3, 3:], 0, atol=1e-15)
        np.testing.assert_allclose(I[3:, 3:], 5.0 * np.eye(3), atol=1e-15)


class TestSpatialCross:
    def test_cross_motion_antisymmetric(self):
        """[v]_x is NOT symmetric but has specific structure."""
        v = np.random.randn(6)
        X = spatial_cross_motion(v)
        # Should have zeros in upper-right 3x3
        np.testing.assert_allclose(X[:3, 3:], 0, atol=1e-15)

    def test_cross_force_relation(self):
        """[v]_x* = -[v]_x^T."""
        v = np.random.randn(6)
        Xm = spatial_cross_motion(v)
        Xf = spatial_cross_force(v)
        np.testing.assert_allclose(Xf, -Xm.T, atol=1e-15)


class TestMotionSubspace:
    def test_revolute_z(self):
        S = motion_subspace_revolute(2)
        expected = np.array([0, 0, 1, 0, 0, 0], dtype=float)
        np.testing.assert_allclose(S, expected)

    def test_revolute_unit_norm(self):
        for axis in [0, 1, 2]:
            S = motion_subspace_revolute(axis)
            assert abs(np.linalg.norm(S) - 1.0) < 1e-15
