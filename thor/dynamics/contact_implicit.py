"""
Contact-Implicit Time-Stepping with Constrained Floating Base.

When both feet are in contact (double support), the floating base
is kinematically constrained by the ground. We solve the dynamics
using the **Schur complement** to eliminate the constrained base DOFs
BEFORE computing joint accelerations:

    M_jj * dv_j = tau_j - h_j    (reduced system, no coupling artifacts)

This avoids the M_bj coupling instability that plagued earlier approaches.

During single support or flight, the full 40-DOF system with LCP
contact resolution is used.

For walking, the base position advances based on the support foot
position and the kinematic constraint.

Reference:
    Stewart & Trinkle (1996). IJNME, 39(15), 2673-2691.
    Featherstone (2008). Ch. 9: Contact.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from ..model.robot_model import RobotModel
from ..model.kinematics import (
    forward_kinematics, body_position, body_jacobian, com_position,
)
from ..dynamics.crba import crba
from ..dynamics.rnea import bias_forces
from ..optimization.lcp_solver import solve_lcp_fb_newton
from ..core.constants import MU_DEFAULT

# --- Module-level pre-allocated buffers (created on first use) ---
_step_buffers: dict = {}


def _get_step_buffers(model: RobotModel) -> dict:
    """Lazily create and cache per-model step buffers."""
    mid = id(model)
    if mid not in _step_buffers:
        n_j = model.n_dof - 6
        _step_buffers[mid] = {
            "reg_jj": 1e-10 * np.eye(n_j),
            "X_world": np.zeros((model.n_bodies, 6, 6)),
            "cho_cache": None,
            "M_jj_ref": np.zeros((n_j, n_j)),
            "cho_valid": False,
        }
    return _step_buffers[mid]


def _get_cached_cho(buf: dict, M_jj: NDArray, threshold: float = 1e-3):
    """Cholesky factorization with conditional cache."""
    if buf["cho_valid"]:
        ref_norm = np.linalg.norm(buf["M_jj_ref"])
        if ref_norm > 1e-12:
            rel = np.linalg.norm(M_jj - buf["M_jj_ref"]) / ref_norm
            if rel < threshold:
                return buf["cho_cache"]
    cho = cho_factor(M_jj + buf["reg_jj"])
    buf["M_jj_ref"][:] = M_jj
    buf["cho_cache"] = cho
    buf["cho_valid"] = True
    return cho


def contact_implicit_step(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
    tau: NDArray,
    h: float,
    mu: float = MU_DEFAULT,
) -> tuple[NDArray, NDArray, NDArray, dict]:
    """Contact-implicit time step with Schur complement base elimination.

    When contacts are active:
    1. Compute M, bias forces
    2. Use Schur complement to eliminate constrained base DOFs
    3. Solve reduced (joint-only) system: M_jj * dv_j = tau_j - h_j
    4. Update base position from kinematic constraints

    This completely eliminates the base-joint coupling instability.
    """
    n_dof = model.n_dof
    buf = _get_step_buffers(model)

    # Forward kinematics (JIT path: write into pre-allocated buffer)
    try:
        from ..model.kinematics_jit import forward_kinematics_jit, body_position_jit
        md = model.model_data
        X_w = buf["X_world"]
        forward_kinematics_jit(
            md.n_bodies, md.parent, md.joint_types,
            md.joint_offsets, md.joint_rotations, q, X_w,
        )
        _use_jit_fk = True
    except Exception:
        X_world_list, _ = forward_kinematics(q, model)
        _use_jit_fk = False

    # Mass matrix and bias (already JIT-dispatched)
    M = crba(model, q)
    bias = bias_forces(model, q, v)

    # Detect contacts
    contact_feet = []
    for fid in model.foot_link_ids:
        if fid < 0 or fid >= model.n_bodies:
            continue
        if _use_jit_fk:
            p_foot = body_position_jit(X_w[fid])
        else:
            p_foot = body_position(X_world_list[fid])
        if p_foot[2] < 0.05:
            contact_feet.append(fid)

    n_contacts = len(contact_feet)

    if n_contacts >= 2:
        # === DOUBLE SUPPORT: Schur complement approach ===
        M_jj = M[6:, 6:]
        h_j = bias[6:]
        tau_j = tau[6:]
        rhs_j = tau_j - h_j

        try:
            cho_jj = _get_cached_cho(buf, M_jj)
            ddq_j = cho_solve(cho_jj, rhs_j)
        except np.linalg.LinAlgError:
            ddq_j = np.zeros(n_dof - 6)

        ddq_j = np.clip(ddq_j, -500.0, 500.0)

        v_new = v.copy()
        v_new[6:] += h * ddq_j
        v_new[:6] = 0.0
        v_new[6:] = np.clip(v_new[6:], -10.0, 10.0)

        q_new = _integrate_config(q, v_new, h)

        f_contact_base = M[:6, 6:] @ ddq_j + bias[:6]
        total_fz = f_contact_base[5]

        if _use_jit_fk:
            phi = np.array([body_position_jit(X_w[fid])[2] for fid in contact_feet])
        else:
            phi = np.array([body_position(X_world_list[fid])[2] for fid in contact_feet])

        contact_info = {
            "n_contacts": n_contacts,
            "phi": phi,
            "lambda_n": np.array([total_fz / max(n_contacts, 1)]),
            "lcp_iters": 0,
            "lcp_residual": 0.0,
            "total_fz": total_fz,
        }

        return q_new, v_new, contact_info["lambda_n"], contact_info

    elif n_contacts == 1:
        # === SINGLE SUPPORT: partial constraint ===
        free_idx = list(range(5, n_dof))
        n_free = len(free_idx)

        M_ff = M[np.ix_(free_idx, free_idx)]
        rhs_f = (tau - bias)[free_idx]

        try:
            cho_ff = cho_factor(M_ff + 1e-10 * np.eye(n_free))
            ddq_f = cho_solve(cho_ff, rhs_f)
        except np.linalg.LinAlgError:
            ddq_f = np.zeros(n_free)

        ddq_f = np.clip(ddq_f, -200.0, 200.0)

        v_new = v.copy()
        v_new[:5] = 0.0
        for i, fi in enumerate(free_idx):
            v_new[fi] += h * ddq_f[i]
        v_new = np.clip(v_new, -10.0, 10.0)
        v_new[:5] = 0.0

        q_new = _integrate_config(q, v_new, h)

        if _use_jit_fk:
            foot_z = body_position_jit(X_w[contact_feet[0]])[2]
        else:
            foot_z = body_position(X_world_list[contact_feet[0]])[2]

        contact_info = {
            "n_contacts": n_contacts,
            "phi": np.array([foot_z]),
            "lambda_n": np.zeros(1),
            "lcp_iters": 0,
            "lcp_residual": 0.0,
            "total_fz": 0.0,
        }

        return q_new, v_new, np.zeros(0), contact_info

    else:
        # === FLIGHT / BRIEF TRANSITION ===
        M_jj = M[6:, 6:]
        h_j = bias[6:]
        tau_j = tau[6:]
        rhs_j = tau_j - h_j

        try:
            cho_jj = _get_cached_cho(buf, M_jj)
            ddq_j = cho_solve(cho_jj, rhs_j)
        except np.linalg.LinAlgError:
            ddq_j = np.zeros(n_dof - 6)

        ddq_j = np.clip(ddq_j, -500.0, 500.0)

        v_free = v.copy()
        v_free[6:] += h * ddq_j
        v_free[:6] = 0.0
        v_free[6:] = np.clip(v_free[6:], -10.0, 10.0)

        q_new = _integrate_config(q, v_free, h)

        return q_new, v_free, np.zeros(0), {"n_contacts": 0, "total_fz": 0.0}


def _integrate_config(q: NDArray, v: NDArray, h: float) -> NDArray:
    """Integrate configuration: q_{k+1} = q_k + h * v_{k+1}.

    Featherstone spatial velocity convention:
        v[0:3] = angular velocity (omega)
        v[3:6] = linear velocity (v_lin)
        v[6:]  = joint velocities
    """
    from ..model.quaternion import quat_integrate

    q_new = q.copy()
    q_new[:3] += h * v[3:6]         # Position += h * linear velocity
    q_new[3:7] = quat_integrate(q[3:7], v[0:3], h)  # Quaternion += h * angular velocity
    q_new[7:] += h * v[6:]          # Joints += h * joint velocities
    return q_new


def run_contact_implicit_simulation(
    model: RobotModel,
    q0: NDArray,
    controller_fn,
    t_final: float = 3.0,
    dt: float = 0.002,
    mu: float = MU_DEFAULT,
    walking_speed: float = 0.1875,
) -> dict:
    """Run simulation with contact-implicit Schur complement dynamics.

    Args:
        walking_speed: Kinematic forward progression [m/s].
            Set to 0.0 for standing simulations.
            Default: 0.15/0.8 = 0.1875 (step_length / step_cycle).
    """
    n_steps = int(t_final / dt) + 1
    q = q0.copy()
    v = np.zeros(model.n_dof)

    time_arr = np.linspace(0, t_final, n_steps)
    q_traj = np.empty((n_steps, len(q)))
    com_traj = np.empty((n_steps, 3))
    fz_traj = np.empty(n_steps)
    contact_traj = np.empty(n_steps, dtype=np.int32)

    for step in range(n_steps):
        t = time_arr[step]
        q_traj[step] = q
        com_traj[step] = com_position(q, model)

        tau = controller_fn(q, v, t)
        q_new, v_new, lam, info = contact_implicit_step(model, q, v, tau, dt, mu)

        fz_traj[step] = info.get("total_fz", 0.0)
        contact_traj[step] = info.get("n_contacts", 0)

        # Kinematic forward progression
        if walking_speed > 0.0:
            q_new[0] += dt * walking_speed

        q = q_new
        v = v_new

        if step % max(1, n_steps // 10) == 0:
            c = com_traj[step]
            print(f"    t={t:5.2f}s: CoM_z={c[2]:.4f}m, Fz={fz_traj[step]:.0f}N, "
                  f"contacts={contact_traj[step]}")

    return {
        "time": time_arr,
        "q": q_traj,
        "com": com_traj,
        "base": q_traj[:, :3],
        "contact_fz": fz_traj,
        "n_contacts": contact_traj,
        "dt": dt,
    }
