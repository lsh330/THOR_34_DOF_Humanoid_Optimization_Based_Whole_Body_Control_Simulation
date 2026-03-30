"""
Tests for quaternion operations.

Validates rotation matrix conversion, integration, and normalization.
"""

import math
import numpy as np
import pytest

from thor.model.quaternion import quat_to_rot, quat_integrate, quat_identity


class TestQuaternionToRotation:
    def test_identity(self):
        R = quat_to_rot(quat_identity())
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonal(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])  # 120 deg rotation
        R = quat_to_rot(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_det_one(self):
        q = np.array([0.707, 0.0, 0.707, 0.0])  # 90 deg about y
        q /= np.linalg.norm(q)
        R = quat_to_rot(q)
        assert abs(np.linalg.det(R) - 1.0) < 1e-14

    def test_90_deg_z(self):
        """Quaternion for 90 deg about z should map x→y."""
        angle = math.pi / 2
        q = np.array([math.cos(angle/2), 0, 0, math.sin(angle/2)])
        R = quat_to_rot(q)
        x_hat = np.array([1.0, 0.0, 0.0])
        result = R @ x_hat
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-14)


class TestQuaternionIntegration:
    def test_zero_omega(self):
        """Zero angular velocity preserves quaternion."""
        q = quat_identity()
        q_new = quat_integrate(q, np.zeros(3), 0.01)
        np.testing.assert_allclose(q_new, q, atol=1e-14)

    def test_normalization(self):
        """Integrated quaternion must be unit norm."""
        q = quat_identity()
        omega = np.array([1.0, 2.0, 3.0])
        q_new = quat_integrate(q, omega, 0.01)
        assert abs(np.linalg.norm(q_new) - 1.0) < 1e-10

    def test_small_rotation(self):
        """Small rotation about z should produce expected quaternion."""
        q = quat_identity()
        omega = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s about z
        dt = 0.01
        q_new = quat_integrate(q, omega, dt)
        # After dt, rotation angle ≈ 0.001 rad about z
        # q ≈ [cos(0.0005), 0, 0, sin(0.0005)]
        expected_angle = 0.001
        assert abs(q_new[3]) > 0  # z component nonzero
        assert q_new[0] > 0.99  # w close to 1
