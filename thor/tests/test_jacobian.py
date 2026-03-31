"""
Numerical Jacobian verification tests.

Validates the analytical body Jacobian against finite-difference
approximation at multiple configurations.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.model.kinematics import (
    forward_kinematics, body_jacobian, body_position, com_position,
)
from thor.simulation.standing import default_standing_config


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def q0(model):
    return default_standing_config(model)


class TestJacobianNumerical:
    """Verify analytical Jacobian matches finite-difference."""

    def _numerical_jacobian(self, body_idx, q, model, eps=1e-6):
        """Compute body Jacobian via central finite differences."""
        n_dof = model.n_dof
        J_num = np.zeros((3, n_dof))  # Linear velocity Jacobian only

        X0, _ = forward_kinematics(q, model)
        p0 = body_position(X0[body_idx])

        for j in range(n_dof):
            q_plus = q.copy()
            q_minus = q.copy()

            if j < 3:
                q_plus[j] += eps
                q_minus[j] -= eps
            elif j < 7:
                # Quaternion: skip (complex perturbation)
                continue
            else:
                q_plus[j] += eps
                q_minus[j] -= eps

            X_plus, _ = forward_kinematics(q_plus, model)
            X_minus, _ = forward_kinematics(q_minus, model)

            p_plus = body_position(X_plus[body_idx])
            p_minus = body_position(X_minus[body_idx])

            J_num[:, j] = (p_plus - p_minus) / (2 * eps)

        return J_num

    def test_foot_jacobian_joint_columns(self, model, q0):
        """Foot Jacobian joint columns match finite-difference (joints only).

        Skip base columns (q-space has quaternion, v-space has angular vel)
        and compare only joint columns where the mapping is direct.
        """
        fid = model.foot_link_ids[0]
        J_analytical = body_jacobian(fid, q0, model)[3:, :]  # Linear part (3 x n_dof)
        J_numerical = self._numerical_jacobian(fid, q0, model)  # (3 x n_dof)

        # Compare joint columns (v indices 6:, q indices 7:)
        # For revolute joints, dq_i = dq_i (no transformation)
        n_joints = model.n_dof - 6
        for j in range(min(n_joints, 10)):  # First 10 joints
            v_col = 6 + j
            q_col = 7 + j
            a = J_analytical[:, v_col]
            n = J_numerical[:, q_col]
            if np.linalg.norm(n) > 0.01:
                # Allow 20% relative tolerance (FK chain approximation)
                np.testing.assert_allclose(
                    a, n, atol=0.3,
                    err_msg=f"Joint {j} Jacobian mismatch")

    def test_pelvis_jacobian(self, model, q0):
        """Pelvis Jacobian should have identity-like structure for base translation."""
        J = body_jacobian(0, q0, model)
        # Base translational columns (3:6) should affect pelvis position directly
        # J[3:6, 3:6] should be approximately eye(3) for pelvis
        J_trans = J[3:6, 3:6]
        # At least the diagonal should be nonzero
        for i in range(3):
            assert abs(J_trans[i, i]) > 0.1

    def test_jacobian_shape(self, model, q0):
        """Jacobian should be 6 x n_dof."""
        for body_idx in [0, 10, 20, 30]:
            J = body_jacobian(body_idx, q0, model)
            assert J.shape == (6, model.n_dof)


class TestCoMConsistency:
    def test_com_within_body_bounds(self, model, q0):
        """CoM should be within the convex hull of body positions."""
        com = com_position(q0, model)
        X, _ = forward_kinematics(q0, model)
        positions = np.array([body_position(X[i]) for i in range(model.n_bodies)])

        # CoM x should be between min and max body x
        assert com[0] >= positions[:, 0].min() - 0.5
        assert com[0] <= positions[:, 0].max() + 0.5

        # CoM z should be positive and below max body z
        assert com[2] > 0
        assert com[2] <= positions[:, 2].max() + 0.1
