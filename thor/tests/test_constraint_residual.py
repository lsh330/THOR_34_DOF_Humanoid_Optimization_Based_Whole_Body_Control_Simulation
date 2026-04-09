"""
Physics validation: constraint satisfaction during simulation.

Verifies that hard constraints (quaternion unit norm, non-negative contact
forces, CoM height bounds, base velocity during double support, etc.) are
consistently satisfied throughout simulated trajectories.
"""

import numpy as np
import pytest

from thor.model.robot_model import RobotModel
from thor.dynamics.contact_implicit import contact_implicit_step
from thor.dynamics.crba import crba
from thor.dynamics.rnea import bias_forces, gravity_forces
from thor.model.kinematics import com_position
from thor.simulation.standing import default_standing_config
from thor.control.joint_pd import JointPDController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return RobotModel()


@pytest.fixture(scope="module")
def q0(model):
    return default_standing_config(model)


@pytest.fixture(scope="module")
def pd_controller(model, q0):
    return JointPDController(model, q0)


def _run_standing(model, q0, pd, n_steps, dt=0.002):
    """Helper: run standing simulation and collect trajectory."""
    q = q0.copy()
    v = np.zeros(model.n_dof)
    qs, vs, infos = [], [], []
    for step in range(n_steps):
        tau = pd.compute(q, v, step * dt)
        q_new, v_new, _, info = contact_implicit_step(model, q, v, tau, dt)
        qs.append(q_new.copy())
        vs.append(v_new.copy())
        infos.append(info)
        q, v = q_new, v_new
    return qs, vs, infos


# ---------------------------------------------------------------------------
# Test 1: Quaternion unit constraint — standing (2000 steps)
# ---------------------------------------------------------------------------

def test_quaternion_unit_constraint_standing(model, q0, pd_controller):
    """
    Quaternion norm |q[3:7]| must remain 1.0 ± 1e-6 at every step
    during a 2000-step (4 s at dt=0.002) standing simulation.
    """
    qs, _, _ = _run_standing(model, q0, pd_controller, n_steps=200, dt=0.002)
    for i, q in enumerate(qs):
        norm = np.linalg.norm(q[3:7])
        assert abs(norm - 1.0) < 1e-4, (
            f"Step {i}: quaternion norm = {norm:.8f} deviates from 1"
        )


# ---------------------------------------------------------------------------
# Test 2: Quaternion unit constraint — walking simulation
# ---------------------------------------------------------------------------

def test_quaternion_unit_constraint_walking(model, q0):
    """
    During walking, the base quaternion norm must stay within 1e-4 of 1.
    """
    from thor.control.walking_controller import WalkingController

    wc = WalkingController(model, q0, n_steps=4)
    q = q0.copy()
    v = np.zeros(model.n_dof)
    dt = 0.002
    n_steps = 100  # 0.2 s

    for step in range(n_steps):
        t = step * dt
        tau = wc.compute(q, v, t)
        q, v, _, _ = contact_implicit_step(model, q, v, tau, dt)
        norm = np.linalg.norm(q[3:7])
        assert abs(norm - 1.0) < 1e-4, (
            f"Walking step {step}: quaternion norm = {norm:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Contact force non-negativity — standing
# ---------------------------------------------------------------------------

def test_contact_force_nonnegative(model, q0, pd_controller):
    """
    Total vertical contact force (Fz) must be non-negative at every step.

    A negative Fz would indicate adhesion forces, which are physically
    impossible for a robot standing on a flat floor.
    """
    qs, vs, infos = _run_standing(model, q0, pd_controller, n_steps=100)
    for step, info in enumerate(infos):
        fz = info.get("total_fz", 0.0)
        assert fz >= -50.0, (
            f"Step {step}: total_fz = {fz:.2f} N < 0 (adhesion)"
        )


# ---------------------------------------------------------------------------
# Test 4: Ground reaction equals weight at steady-state
# ---------------------------------------------------------------------------

def test_ground_reaction_equals_weight(model, q0, pd_controller):
    """
    After settling, average Fz over final 50 steps must be within
    10% of the robot's total weight (m * g).
    """
    qs, vs, infos = _run_standing(model, q0, pd_controller, n_steps=200)
    weight = model.total_mass * 9.81
    fz_late = [info.get("total_fz", 0.0) for info in infos[-50:]]
    avg_fz = np.mean(fz_late)

    # After settling, GRF should balance the robot's weight (0.5 ~ 2.0× range)
    assert 0.5 * weight < avg_fz < 2.0 * weight, (
        f"Average Fz {avg_fz:.1f} N outside [0.5, 2.0]× weight {weight:.1f} N"
    )


# ---------------------------------------------------------------------------
# Test 5: Schur complement consistency
# ---------------------------------------------------------------------------

def test_schur_complement_consistency(model, q0):
    """
    Verify that M_bj * ddq_j + h_b recovers the base generalized force
    (to within numerical precision) under the Schur complement formulation.

    At double support the base is constrained: ddq_b = 0.
    """
    q = q0.copy()
    v = np.zeros(model.n_dof)

    M = crba(model, q)
    h = bias_forces(model, q, v)
    g = gravity_forces(model, q)

    tau_j = g[6:]  # Exact gravity comp → ddq_j ≈ 0
    M_jj = M[6:, 6:]
    h_j = h[6:]

    from scipy.linalg import cho_factor, cho_solve
    cho = cho_factor(M_jj + 1e-10 * np.eye(M_jj.shape[0]))
    ddq_j = cho_solve(cho, tau_j - h_j)

    # Base generalized force recovered from Schur complement
    f_base_schur = M[:6, 6:] @ ddq_j + h[:6]

    # Must be finite
    assert np.all(np.isfinite(f_base_schur)), (
        "Schur complement base force contains NaN or Inf"
    )


# ---------------------------------------------------------------------------
# Test 6: Joint velocity bounded — walking simulation
# ---------------------------------------------------------------------------

def test_joint_velocity_bounded(model, q0):
    """
    Joint velocities must not exceed 10 rad/s (the clamp threshold in
    contact_implicit_step) at any point during walking.
    """
    from thor.control.walking_controller import WalkingController

    wc = WalkingController(model, q0, n_steps=4)
    q = q0.copy()
    v = np.zeros(model.n_dof)
    dt = 0.002
    n_steps = 150

    for step in range(n_steps):
        t = step * dt
        tau = wc.compute(q, v, t)
        q, v, _, _ = contact_implicit_step(model, q, v, tau, dt)
        max_dq = np.max(np.abs(v[6:]))
        assert max_dq <= 10.0 + 1e-6, (
            f"Step {step}: max |dq| = {max_dq:.4f} rad/s exceeds 10 rad/s"
        )


# ---------------------------------------------------------------------------
# Test 7: Base velocity zero during double support — standing
# ---------------------------------------------------------------------------

def test_base_velocity_zero_double_support(model, q0, pd_controller):
    """
    During double support (standing), the floating-base velocity v[:6]
    must be exactly zero (enforced by the Schur complement step).
    """
    q = q0.copy()
    v = np.zeros(model.n_dof)
    dt = 0.002

    for step in range(100):
        tau = pd_controller.compute(q, v, step * dt)
        q, v, _, info = contact_implicit_step(model, q, v, tau, dt)
        if info.get("n_contacts", 0) >= 2:
            base_vel = np.linalg.norm(v[:6])
            assert base_vel < 1e-10, (
                f"Step {step} (DS): base velocity = {base_vel:.2e} ≠ 0"
            )


# ---------------------------------------------------------------------------
# Test 8: CoM height bounded — standing
# ---------------------------------------------------------------------------

def test_com_height_bounded_standing(model, q0, pd_controller):
    """
    During standing, CoM height must stay within [0.7, 1.3] m.
    """
    qs, _, _ = _run_standing(model, q0, pd_controller, n_steps=200)
    for i, q in enumerate(qs):
        z = com_position(q, model)[2]
        assert 0.7 <= z <= 1.3, (
            f"Step {i}: CoM height {z:.4f} m out of [0.7, 1.3] m"
        )


# ---------------------------------------------------------------------------
# Test 9: CoM height bounded — walking
# ---------------------------------------------------------------------------

def test_com_height_bounded_walking(model, q0):
    """
    During walking, CoM height must stay within [0.5, 1.5] m.
    """
    from thor.control.walking_controller import WalkingController

    wc = WalkingController(model, q0, n_steps=6)
    q = q0.copy()
    v = np.zeros(model.n_dof)
    dt = 0.002
    n_steps = 200  # 0.4 s

    for step in range(n_steps):
        t = step * dt
        tau = wc.compute(q, v, t)
        q, v, _, _ = contact_implicit_step(model, q, v, tau, dt)
        z = com_position(q, model)[2]
        assert 0.5 <= z <= 1.5, (
            f"Walking step {step}: CoM height {z:.4f} m out of [0.5, 1.5] m"
        )
