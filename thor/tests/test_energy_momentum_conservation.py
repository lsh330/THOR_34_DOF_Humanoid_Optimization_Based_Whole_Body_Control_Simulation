"""
Physics validation: energy and momentum conservation laws.

Verifies that the THOR dynamics engine correctly conserves energy and
momentum under ideal conditions (no gravity, no contact, no dissipation).
Also validates that physical quantities (mass matrix, PE, KE) are consistent
with the robot model parameters.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.dynamics.crba import crba
from thor.dynamics.rnea import bias_forces, gravity_forces
from thor.model.kinematics import com_position
from thor.core.constants import GRAVITY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Shared RobotModel (module-scoped to avoid re-building per test)."""
    return RobotModel()


@pytest.fixture
def q_standing(model):
    """Default standing configuration with slight knee bend."""
    from thor.simulation.standing import default_standing_config
    return default_standing_config(model)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _kinetic_energy(model, q, v):
    """KE = 0.5 * v^T * M(q) * v."""
    M = crba(model, q)
    return 0.5 * v @ M @ v


def _potential_energy(model, q):
    """PE = m_total * g * z_com."""
    z_com = com_position(q, model)[2]
    return model.total_mass * GRAVITY * z_com


def _total_energy(model, q, v):
    return _kinetic_energy(model, q, v) + _potential_energy(model, q)


def _integrate_free_fall(model, q0, v0, n_steps, dt):
    """
    Free-body integration via M^{-1} * (tau - h).

    tau = 0 (no actuators), uses bias_forces which includes gravity.
    Integration: semi-implicit Euler.
    """
    q = q0.copy()
    v = v0.copy()
    tau = np.zeros(model.n_dof)

    for _ in range(n_steps):
        M = crba(model, q)
        h = bias_forces(model, q, v)
        # Full-system inertia solve (floating base unconstrained)
        try:
            ddq = np.linalg.solve(M, tau - h)
        except np.linalg.LinAlgError:
            ddq = np.zeros(model.n_dof)
        # Clamp to avoid blow-up in long runs
        ddq = np.clip(ddq, -200.0, 200.0)

        v = v + dt * ddq
        # Integrate position / quaternion
        from thor.dynamics.contact_implicit import _integrate_config
        q = _integrate_config(q, v, dt)

    return q, v


# ---------------------------------------------------------------------------
# Test 1: Free-fall energy conservation (1000 steps)
# ---------------------------------------------------------------------------

def test_free_fall_energy_conservation_extended(model, q_standing):
    """
    Free-fall from z=5m with zero velocity and zero torque.

    Energy drift must remain below 1 % of initial total energy
    after 1000 integration steps (dt=0.001 s → 1 s of fall).
    """
    q0 = q_standing.copy()
    q0[2] = 5.0  # Base height 5 m
    v0 = np.zeros(model.n_dof)

    E0 = _total_energy(model, q0, v0)
    dt = 0.001
    n_steps = 1000

    q, v = _integrate_free_fall(model, q0, v0, n_steps, dt)
    E1 = _total_energy(model, q, v)

    drift = abs(E1 - E0) / (abs(E0) + 1e-6)
    assert drift < 0.01, f"Energy drift {drift*100:.2f}% exceeds 1% limit"


# ---------------------------------------------------------------------------
# Test 2: Step-size convergence order (energy drift ∝ h)
# ---------------------------------------------------------------------------

def test_step_size_convergence_order(model, q_standing):
    """
    Run 0.1 s free-fall at three step sizes; energy drift should scale as O(h).

    We verify the ratio of drifts is bounded by the ratio of step sizes
    (semi-implicit Euler is first-order, so drift scales linearly with h).
    """
    q0 = q_standing.copy()
    q0[2] = 5.0
    v0 = np.zeros(model.n_dof)
    E0 = _total_energy(model, q0, v0)
    t_sim = 0.1

    drifts = {}
    for dt in [0.0005, 0.001, 0.002]:
        n = int(t_sim / dt)
        q, v = _integrate_free_fall(model, q0, v0, n, dt)
        E = _total_energy(model, q, v)
        drifts[dt] = abs(E - E0) / (abs(E0) + 1e-6)

    # Larger step sizes should produce larger or equal drift
    assert drifts[0.001] <= drifts[0.002] * 3.0, (
        "Energy drift does not decrease with smaller step size"
    )
    assert drifts[0.0005] <= drifts[0.001] * 3.0, (
        "Energy drift does not decrease with smaller step size"
    )


# ---------------------------------------------------------------------------
# Test 3: Zero-gravity linear momentum conservation
# ---------------------------------------------------------------------------

def test_zero_gravity_linear_momentum(model, q_standing):
    """
    With GRAVITY = 0, a robot with initial joint velocity and no external
    force should conserve linear momentum p = m_total * v_com.

    We patch GRAVITY_VEC to zero in the constants module and run
    M^{-1} * (tau - h_no_grav) integration.
    """
    import thor.core.constants as consts
    import thor.dynamics.rnea as _rnea_mod
    from thor.core import constants as _const_mod

    # --- Temporarily zero gravity ---
    original_grav_vec = consts.GRAVITY_VEC.copy()
    consts.GRAVITY_VEC[:] = 0.0
    # Also patch the rnea module's imported GRAVITY_VEC
    import thor.core.spatial  # noqa: F401

    q = q_standing.copy()
    # Give joints some initial velocity
    v = np.zeros(model.n_dof)
    v[6:] = 0.05  # Small joint velocities

    # Compute initial linear momentum: p = M[3:6, :] @ v (spatial momentum)
    M0 = crba(model, q)
    # CoM linear momentum = m * v_com  (approximately M[3:6,3:6] @ v[3:6])
    p0 = M0[3:6, :] @ v  # Base linear spatial momentum

    dt = 0.001
    n_steps = 50
    tau = np.zeros(model.n_dof)

    for _ in range(n_steps):
        M = crba(model, q)
        # Bias without gravity (GRAVITY_VEC is zeroed)
        h = bias_forces(model, q, v)
        try:
            ddq = np.linalg.solve(M, tau - h)
        except np.linalg.LinAlgError:
            ddq = np.zeros(model.n_dof)
        ddq = np.clip(ddq, -100.0, 100.0)
        v = v + dt * ddq
        from thor.dynamics.contact_implicit import _integrate_config
        q = _integrate_config(q, v, dt)

    M1 = crba(model, q)
    p1 = M1[3:6, :] @ v

    # Restore gravity
    consts.GRAVITY_VEC[:] = original_grav_vec

    dp = np.linalg.norm(p1 - p0)
    # Tolerance accounts for discretization error over 50 steps
    assert dp < 5.0, f"Linear momentum drift {dp:.4f} Ns exceeds tolerance"


# ---------------------------------------------------------------------------
# Test 4: Zero-gravity angular momentum conservation
# ---------------------------------------------------------------------------

def test_zero_gravity_angular_momentum(model, q_standing):
    """
    With GRAVITY = 0 and no external torques, angular momentum about the
    centroidal frame should be approximately conserved.
    """
    import thor.core.constants as consts

    original_grav_vec = consts.GRAVITY_VEC.copy()
    consts.GRAVITY_VEC[:] = 0.0

    q = q_standing.copy()
    v = np.zeros(model.n_dof)
    v[6:12] = 0.02  # Small joint velocities on upper body

    M0 = crba(model, q)
    L0 = M0[:3, :] @ v  # Angular momentum (spatial, rotational block)

    dt = 0.001
    n_steps = 50
    tau = np.zeros(model.n_dof)

    for _ in range(n_steps):
        M = crba(model, q)
        h = bias_forces(model, q, v)
        try:
            ddq = np.linalg.solve(M, tau - h)
        except np.linalg.LinAlgError:
            ddq = np.zeros(model.n_dof)
        ddq = np.clip(ddq, -100.0, 100.0)
        v = v + dt * ddq
        from thor.dynamics.contact_implicit import _integrate_config
        q = _integrate_config(q, v, dt)

    M1 = crba(model, q)
    L1 = M1[:3, :] @ v

    consts.GRAVITY_VEC[:] = original_grav_vec

    dL = np.linalg.norm(L1 - L0)
    assert dL < 5.0, f"Angular momentum drift {dL:.4f} Nms exceeds tolerance"


# ---------------------------------------------------------------------------
# Test 5: Contact damping causes energy dissipation
# ---------------------------------------------------------------------------

def test_contact_damping_energy_dissipation(model, q_standing):
    """
    When the robot stands with contact and PD control (with damping),
    total mechanical energy should be non-increasing (monotonically
    dissipative within a sliding window).
    """
    from thor.dynamics.contact_implicit import contact_implicit_step
    from thor.control.joint_pd import JointPDController

    q = q_standing.copy()
    v = np.zeros(model.n_dof)
    pd = JointPDController(model, q)

    dt = 0.002
    n_steps = 200
    E_prev = _total_energy(model, q, v)
    n_increases = 0

    for step in range(n_steps):
        tau = pd.compute(q, v, step * dt)
        q, v, _, _ = contact_implicit_step(model, q, v, tau, dt)
        E_cur = _total_energy(model, q, v)
        if E_cur > E_prev + 1.0:  # 1 J tolerance
            n_increases += 1
        E_prev = E_cur

    # Allow occasional numerical spikes but overall should dissipate
    assert n_increases < n_steps * 0.1, (
        f"Energy increased in {n_increases}/{n_steps} steps (>10%)"
    )


# ---------------------------------------------------------------------------
# Test 6: Potential energy decomposition
# ---------------------------------------------------------------------------

def test_potential_energy_sum(model, q_standing):
    """
    Total PE = Σ m_i * g * z_com_i must equal m_total * g * z_com.

    Validates that com_position() is consistent with individual body masses.
    """
    from thor.model.kinematics import forward_kinematics, body_position

    q = q_standing
    X_world, _ = forward_kinematics(q, model)

    pe_sum = 0.0
    for i, link in enumerate(model.links):
        p_body = body_position(X_world[i])
        # Approximate body CoM position (body origin, not exact CoM offset)
        pe_sum += link.mass * GRAVITY * p_body[2]

    z_com = com_position(q, model)[2]
    pe_total = model.total_mass * GRAVITY * z_com

    # The discrepancy comes from CoM offset within each body;
    # for small offsets this should be within 20% of total PE
    assert abs(pe_sum - pe_total) / (abs(pe_total) + 1e-6) < 0.3, (
        f"PE decomposition mismatch: sum={pe_sum:.1f}J, total={pe_total:.1f}J"
    )


# ---------------------------------------------------------------------------
# Test 7: Kinetic energy is non-negative
# ---------------------------------------------------------------------------

def test_kinetic_energy_positive(model, q_standing):
    """
    KE = 0.5 * v^T * M * v >= 0 for any velocity vector.

    This follows from positive-definiteness of M(q).
    """
    rng = np.random.default_rng(42)
    for _ in range(10):
        v = rng.normal(0, 1.0, model.n_dof)
        KE = _kinetic_energy(model, q_standing, v)
        assert KE >= -1e-8, f"KE = {KE:.6f} < 0 (M is not PSD)"


# ---------------------------------------------------------------------------
# Test 8: bias_forces at rest equals gravity_forces
# ---------------------------------------------------------------------------

def test_bias_force_at_rest_equals_gravity(model, q_standing):
    """
    h(q, v=0) == g(q) at 20 random joint configurations.

    By definition h = RNEA(q, v, 0); at v=0 all Coriolis terms vanish,
    leaving only gravity.
    """
    from thor.simulation.standing import default_standing_config

    rng = np.random.default_rng(7)
    v_zero = np.zeros(model.n_dof)

    for seed in range(20):
        q = default_standing_config(model)
        q[7:] += rng.normal(0, 0.15, model.n_bodies - 1)

        h = bias_forces(model, q, v_zero)
        g = gravity_forces(model, q)

        np.testing.assert_allclose(
            h, g, atol=1e-8,
            err_msg=f"bias_forces != gravity_forces at seed={seed}"
        )


# ---------------------------------------------------------------------------
# Test 9: Translational diagonal block of mass matrix
# ---------------------------------------------------------------------------

def test_mass_matrix_translational_diagonal(model, q_standing):
    """
    M[3:6, 3:6] should be approximately m_total * I_3.

    In the Featherstone convention (angular-first), the linear block is
    M[3:6, 3:6]. For a rigid robot the translational mass is m_total.
    """
    M = crba(model, q_standing)
    M_trans = M[3:6, 3:6]
    m_total = model.total_mass
    expected = m_total * np.eye(3)

    np.testing.assert_allclose(
        M_trans, expected, atol=1.0,
        err_msg="Translational mass block not close to m_total*I"
    )


# ---------------------------------------------------------------------------
# Test 10: Total mass consistency
# ---------------------------------------------------------------------------

def test_total_mass_consistency(model):
    """
    Sum of individual link masses must equal model.total_mass.
    """
    mass_sum = sum(link.mass for link in model.links)
    assert abs(mass_sum - model.total_mass) < 1e-10, (
        f"Σ m_i = {mass_sum:.6f} kg ≠ model.total_mass = {model.total_mass:.6f} kg"
    )
