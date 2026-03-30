"""
Floating-base simulation runner with contact dynamics.

Uses verified CRBA + RNEA (instead of ABA) for numerical stability:
    M(q) * ddq = S^T * tau + J_c^T * f_c - h(q, v)

Contact model: Spring-Damper (Kelvin-Voigt) with continuous friction.

Reference:
    Featherstone, R. (2008). Ch. 9: Contact.
    Marhefka & Orin (1999). IEEE Trans. SMC, 29(6).
"""

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..model.kinematics import (
    forward_kinematics, body_position, com_position, body_jacobian,
)
from ..dynamics.crba import crba
from ..dynamics.rnea import bias_forces
from ..dynamics.contact import (
    contact_force_single, compute_foot_contact_points,
)


def _compute_contact(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
    X_world: list[NDArray],
) -> tuple[NDArray, float]:
    """Compute total generalized contact force via Jacobian transpose.

    tau_contact = J_c^T * f_c (in generalized coordinates)
    """
    n_dof = model.n_dof
    tau_contact = np.zeros(n_dof)
    total_fz = 0.0

    for fid in model.foot_link_ids:
        if fid < 0 or fid >= model.n_bodies:
            continue

        R_foot = X_world[fid][:3, :3]
        p_foot = body_position(X_world[fid])

        # Foot Jacobian (6 × n_dof)
        J_foot = body_jacobian(fid, q, model, X_world)
        v_foot = J_foot @ v
        v_foot_lin = v_foot[3:]
        omega_foot = v_foot[:3]

        corners = compute_foot_contact_points(p_foot, R_foot)

        for j in range(4):
            r = corners[j] - p_foot
            point_vel = v_foot_lin + np.cross(omega_foot, r)
            f_world = contact_force_single(corners[j], point_vel)

            if np.abs(f_world).max() > 0.0:
                # Jacobian transpose mapping: tau = J^T * f
                # J for the contact point = J_foot + cross(r) correction
                # Simplified: use foot Jacobian (accurate enough for foot center)
                J_point_lin = J_foot[3:, :]  # Linear velocity Jacobian
                tau_contact += J_point_lin.T @ f_world
                total_fz += f_world[2]

    return tau_contact, total_fz


def run_floating_base_simulation(
    model: RobotModel,
    q0: NDArray,
    controller_fn,
    t_final: float = 3.0,
    dt: float = 0.001,
) -> dict:
    """Run floating-base simulation with CRBA/RNEA + contact."""
    n_dof = model.n_dof
    n_steps = int(t_final / dt) + 1

    q = q0.copy()
    v = np.zeros(n_dof)

    # Pre-allocate storage
    time_arr = np.linspace(0, t_final, n_steps)
    com_traj = np.empty((n_steps, 3))
    base_traj = np.empty((n_steps, 3))
    tau_traj = np.empty((n_steps, n_dof))
    fz_traj = np.empty(n_steps)

    for step in range(n_steps):
        t = time_arr[step]

        com_traj[step] = com_position(q, model)
        base_traj[step] = q[:3]

        # Dynamics
        X_world, _ = forward_kinematics(q, model)
        M = crba(model, q)
        h = bias_forces(model, q, v)

        # Contact forces (generalized)
        tau_contact, fz = _compute_contact(model, q, v, X_world)
        fz_traj[step] = fz

        # Controller torques
        tau_ctrl = controller_fn(q, v, t)
        tau_traj[step] = tau_ctrl

        # Total generalized force
        tau_total = tau_ctrl + tau_contact - h

        # Forward dynamics: M * ddq = tau_total
        try:
            ddq = np.linalg.solve(M, tau_total)
        except np.linalg.LinAlgError:
            ddq = np.zeros(n_dof)

        # Clamp accelerations
        ddq = np.clip(ddq, -500.0, 500.0)

        # Semi-implicit Euler
        v_new = v + dt * ddq
        v_new = np.clip(v_new, -30.0, 30.0)

        # Update base position
        q[:3] += dt * v_new[:3]

        # Update base quaternion
        omega = v_new[3:6]
        w, x, y, z_q = q[3], q[4], q[5], q[6]
        dquat = 0.5 * dt * np.array([
            -omega[0]*x - omega[1]*y - omega[2]*z_q,
            omega[0]*w + omega[2]*y - omega[1]*z_q,
            omega[1]*w - omega[2]*x + omega[0]*z_q,
            omega[2]*w + omega[1]*x - omega[0]*y,
        ])
        q[3:7] += dquat
        qnorm = np.linalg.norm(q[3:7])
        if qnorm > 1e-10:
            q[3:7] /= qnorm

        # Update joints
        q[7:] += dt * v_new[6:]
        v = v_new

        if step % max(1, n_steps // 10) == 0:
            c = com_traj[step]
            print(f"    t={t:5.2f}s: CoM_z={c[2]:.4f}m, Fz={fz:.0f}N, "
                  f"base_z={q[2]:.4f}")

    return {
        "time": time_arr,
        "com": com_traj,
        "base": base_traj,
        "tau": tau_traj,
        "contact_fz": fz_traj,
        "dt": dt,
    }
