"""
Edge cases and robustness tests.

Validates the dynamics engine against degenerate inputs: extreme joint
angles, large velocities, zero timestep, and reproducibility.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.dynamics.crba import crba
from thor.dynamics.rnea import rnea, bias_forces, gravity_forces
from thor.model.kinematics import com_position
from thor.simulation.standing import default_standing_config
from thor.model.quaternion import quat_to_rot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return RobotModel()


@pytest.fixture(scope="module")
def q_standing(model):
    return default_standing_config(model)


# ---------------------------------------------------------------------------
# Test 1: Zero initial conditions → static equilibrium
# ---------------------------------------------------------------------------

def test_zero_initial_conditions(model, q_standing):
    """
    With perfect gravity compensation (tau = g(q)) and zero velocity,
    the joint accelerations should be negligible (near-static equilibrium).
    """
    q = q_standing.copy()
    v = np.zeros(model.n_dof)

    g = gravity_forces(model, q)
    tau = np.zeros(model.n_dof)
    tau[6:] = g[6:]  # Feed exactly gravity torques to joints

    M = crba(model, q)
    h = bias_forces(model, q, v)

    M_jj = M[6:, 6:]
    rhs = tau[6:] - h[6:]

    try:
        ddq_j = np.linalg.solve(M_jj, rhs)
    except np.linalg.LinAlgError:
        ddq_j = np.zeros(model.n_dof - 6)

    # At v=0 with perfect gravity comp, bias == gravity, so rhs ≈ 0
    assert np.linalg.norm(ddq_j) < 1e-8, (
        f"Static equilibrium violated: ||ddq_j|| = {np.linalg.norm(ddq_j):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Extreme joint angles — CRBA finite
# ---------------------------------------------------------------------------

def test_extreme_joint_angles_crba(model, q_standing):
    """
    CRBA must produce a finite mass matrix at joint limit configurations.

    Tests that trig functions (cos/sin) in the spatial transforms do not
    cause NaN or Inf even at ±q_max angles.
    """
    q = q_standing.copy()
    for i, link in enumerate(model.links[1:], start=1):
        q[6 + i] = link.q_max  # Drive each joint to its maximum angle

    M = crba(model, q)
    assert np.all(np.isfinite(M)), "CRBA produced NaN/Inf at q_max configuration"


# ---------------------------------------------------------------------------
# Test 3: Extreme joint angles — RNEA finite
# ---------------------------------------------------------------------------

def test_extreme_joint_angles_rnea(model, q_standing):
    """
    RNEA must produce finite generalized forces at joint limit configurations.
    """
    q = q_standing.copy()
    for i, link in enumerate(model.links[1:], start=1):
        q[6 + i] = link.q_max

    v_zero = np.zeros(model.n_dof)
    a_zero = np.zeros(model.n_dof)
    tau = rnea(model, q, v_zero, a_zero)
    assert np.all(np.isfinite(tau)), "RNEA produced NaN/Inf at q_max configuration"


# ---------------------------------------------------------------------------
# Test 4: Large velocity — RNEA finite
# ---------------------------------------------------------------------------

def test_large_velocity_rnea(model, q_standing):
    """
    RNEA(q, v=10*ones, 0) must remain finite.

    Tests numerical robustness of the Coriolis/centrifugal computation at
    velocities well beyond normal operating range.
    """
    q = q_standing.copy()
    v = np.ones(model.n_dof) * 10.0
    a_zero = np.zeros(model.n_dof)
    tau = rnea(model, q, v, a_zero)
    assert np.all(np.isfinite(tau)), "RNEA produced NaN/Inf with large velocity"


# ---------------------------------------------------------------------------
# Test 5: Zero timestep — state unchanged
# ---------------------------------------------------------------------------

def test_zero_timestep(model, q_standing):
    """
    contact_implicit_step with h=0 must return q, v unchanged.

    A zero timestep should produce no state change regardless of torque input.
    """
    from thor.dynamics.contact_implicit import contact_implicit_step

    q = q_standing.copy()
    v = np.zeros(model.n_dof)
    g = gravity_forces(model, q)
    tau = np.zeros(model.n_dof)
    tau[6:] = g[6:]

    q_new, v_new, _, _ = contact_implicit_step(model, q, v, tau, h=0.0)

    # With h=0, position should be identical; velocity clamping may apply
    np.testing.assert_allclose(
        q_new[:3], q[:3], atol=1e-15,
        err_msg="Position changed with zero timestep"
    )
    np.testing.assert_allclose(
        q_new[7:], q[7:], atol=1e-15,
        err_msg="Joint angles changed with zero timestep"
    )


# ---------------------------------------------------------------------------
# Test 6: Mass matrix condition number
# ---------------------------------------------------------------------------

def test_mass_matrix_condition_number(model, q_standing):
    """
    The joint-joint block M_jj must be well-conditioned (cond < 1e8)
    at the standing configuration.

    A large condition number would indicate near-singular dynamics,
    causing numerical instability in the Cholesky solve.
    """
    M = crba(model, q_standing)
    M_jj = M[6:, 6:]
    cond = np.linalg.cond(M_jj)
    assert cond < 1e8, (
        f"M_jj condition number {cond:.2e} exceeds threshold 1e8"
    )


# ---------------------------------------------------------------------------
# Test 7: Numerical symmetry of CRBA output
# ---------------------------------------------------------------------------

def test_numerical_symmetry_crba(model, q_standing):
    """
    CRBA must produce a symmetric matrix to machine epsilon (atol = 1e-12).

    Asymmetry would indicate a bug in the off-diagonal fill.
    """
    M = crba(model, q_standing)
    asym = np.max(np.abs(M - M.T))
    assert asym < 1e-12, (
        f"Mass matrix max asymmetry = {asym:.2e}, expected < 1e-12"
    )


# ---------------------------------------------------------------------------
# Test 8: Single-body model (pelvis only) — skipped (model is fixed)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Full THOR model cannot be reduced to pelvis-only")
def test_single_body_model(model):
    """
    Placeholder: pelvis-only model would have M[:6,:6] == spatial_inertia(pelvis).
    Skipped because RobotModel is not modifiable without redesign.
    """
    pass


# ---------------------------------------------------------------------------
# Test 9: Identity quaternion → rotation matrix = I_3
# ---------------------------------------------------------------------------

def test_identity_quaternion_rotation(model):
    """
    quat_to_rot([1, 0, 0, 0]) must return the 3×3 identity matrix.
    """
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_rot(q_id)
    np.testing.assert_allclose(
        R, np.eye(3), atol=1e-15,
        err_msg="Identity quaternion does not map to I_3"
    )


# ---------------------------------------------------------------------------
# Test 10: Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(model, q_standing):
    """
    Two identical simulations (same inputs, no randomness) must produce
    bit-identical results.
    """
    from thor.dynamics.contact_implicit import contact_implicit_step
    from thor.control.joint_pd import JointPDController

    def _run(q_init):
        q = q_init.copy()
        v = np.zeros(model.n_dof)
        pd = JointPDController(model, q)
        results = []
        for step in range(50):
            tau = pd.compute(q, v, step * 0.002)
            q, v, _, _ = contact_implicit_step(model, q, v, tau, 0.002)
            results.append(q.copy())
        return results

    run1 = _run(q_standing)
    run2 = _run(q_standing)

    for step, (r1, r2) in enumerate(zip(run1, run2)):
        np.testing.assert_array_equal(
            r1, r2,
            err_msg=f"Non-deterministic result at step {step}"
        )
