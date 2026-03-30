"""
Floating-base simulation with Jacobian-transpose contact model.

Contact force computed in Cartesian space, mapped to generalized
forces via Jacobian transpose:
    tau_contact = J_foot^T * f_contact

where f_contact is a spring-damper force in world frame:
    f_z = k * max(0, -phi) + d * max(0, -dphi)

This correctly maps contact forces through the kinematic chain
without KKT conditioning issues.

Reference:
    Khatib, O. (1987). "A unified approach for motion and force
    control of robot manipulators." IEEE J-RA, 3(1), 43-53.
"""

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..model.kinematics import (
    forward_kinematics, body_position, com_position, body_jacobian,
)
from ..dynamics.crba import crba
from ..dynamics.rnea import bias_forces


# Contact model parameters (tuned for stability with dt=0.001)
# Tuned for 65kg robot, 2 feet contact:
# Equilibrium penetration: mg/(2k) = 659/(2*30000) = 0.011m (11mm)
# Critical damping: d_cr = 2*sqrt(k*m/2) = 2*sqrt(30000*32.5) = 1975 N*s/m
CONTACT_K: float = 30000.0     # Normal stiffness [N/m]
CONTACT_D: float = 2000.0      # Normal damping [N*s/m] (~critical)
GROUND_Z: float = 0.0


def _compute_jacobian_contact(
    model: RobotModel,
    q: NDArray,
    v: NDArray,
    X_world: list[NDArray],
) -> tuple[NDArray, float]:
    """Compute contact generalized force via Jacobian transpose.

    For each foot in contact:
        1. Compute foot position and velocity via Jacobian
        2. Compute contact force in world frame (spring-damper on z)
        3. Map to generalized force: tau += J_foot^T * f

    Returns:
        tau_contact: (n_dof,) generalized contact force
        total_fz: Total vertical contact force [N]
    """
    n_dof = model.n_dof
    tau_contact = np.zeros(n_dof)
    total_fz = 0.0

    for fid in model.foot_link_ids:
        if fid < 0 or fid >= model.n_bodies:
            continue

        p_foot = body_position(X_world[fid])
        phi = p_foot[2] - GROUND_Z  # Penetration (negative = in contact)

        if phi < 0.05:  # Contact or near-contact
            J_foot = body_jacobian(fid, q, model, X_world)
            v_foot = J_foot @ v  # (6,) [omega(3), v_lin(3)]
            vz = v_foot[5]  # z-component of linear velocity

            # Normal force (z-direction)
            penetration = max(0.0, -phi)
            f_z = CONTACT_K * penetration + CONTACT_D * max(0.0, -vz)
            f_z = min(f_z, 3000.0)  # Clamp

            if f_z > 0.0:
                # World-frame force (only z for ground contact)
                f_world = np.array([0.0, 0.0, 0.0, 0.0, 0.0, f_z])

                # Map to generalized coordinates via Jacobian transpose
                tau_contact += J_foot.T @ f_world
                total_fz += f_z

    return tau_contact, total_fz


def run_floating_base_simulation(
    model: RobotModel,
    q0: NDArray,
    controller_fn,
    t_final: float = 3.0,
    dt: float = 0.001,
) -> dict:
    """Run floating-base simulation with Jacobian-transpose contact."""
    n_dof = model.n_dof
    n_steps = int(t_final / dt) + 1

    q = q0.copy()
    v = np.zeros(n_dof)

    time_arr = np.linspace(0, t_final, n_steps)
    com_traj = np.empty((n_steps, 3))
    base_traj = np.empty((n_steps, 3))
    tau_traj = np.empty((n_steps, n_dof))
    fz_traj = np.empty(n_steps)

    for step in range(n_steps):
        t = time_arr[step]
        com_traj[step] = com_position(q, model)
        base_traj[step] = q[:3]

        X_world, _ = forward_kinematics(q, model)
        M = crba(model, q)
        h = bias_forces(model, q, v)

        # Contact forces via Jacobian transpose
        tau_contact, fz = _compute_jacobian_contact(model, q, v, X_world)
        fz_traj[step] = fz

        # Controller
        tau_ctrl = controller_fn(q, v, t)
        tau_traj[step] = tau_ctrl

        # Forward dynamics with base rotation constraint
        # When feet are in contact, constrain base rotation to zero
        # and horizontal translation to zero (only vertical + joints free)
        tau_total = tau_ctrl + tau_contact - h

        if fz > 10.0:
            # Both feet in contact: solve reduced system
            # Free DOFs: [vz(1), joints(34)] = indices [5, 6:40]
            # Constrained: [wx,wy,wz,vx,vy] = indices [0:5] → ddq=0
            free_idx = [5] + list(range(6, n_dof))
            n_free = len(free_idx)
            M_red = M[np.ix_(free_idx, free_idx)]
            rhs_red = tau_total[free_idx]

            try:
                ddq_red = np.linalg.solve(M_red, rhs_red)
            except np.linalg.LinAlgError:
                ddq_red = np.zeros(n_free)

            ddq = np.zeros(n_dof)
            for i, fi in enumerate(free_idx):
                ddq[fi] = ddq_red[i]
        else:
            # Free flight: solve full system
            try:
                ddq = np.linalg.solve(M, tau_total)
            except np.linalg.LinAlgError:
                ddq = np.zeros(n_dof)

        # Clamp accelerations
        ddq = np.clip(ddq, -100.0, 100.0)

        # Semi-implicit Euler
        v += dt * ddq
        v = np.clip(v, -15.0, 15.0)

        q[:3] += dt * v[:3]

        # Quaternion integration
        omega = v[3:6]
        w, x, y, z_q = q[3], q[4], q[5], q[6]
        dquat = 0.5 * dt * np.array([
            -omega[0]*x - omega[1]*y - omega[2]*z_q,
            omega[0]*w + omega[2]*y - omega[1]*z_q,
            omega[1]*w - omega[2]*x + omega[0]*z_q,
            omega[2]*w + omega[1]*x - omega[0]*y,
        ])
        q[3:7] += dquat
        qn = np.linalg.norm(q[3:7])
        if qn > 1e-10:
            q[3:7] /= qn

        q[7:] += dt * v[6:]

        if step % max(1, n_steps // 10) == 0:
            c = com_traj[step]
            print(f"    t={t:5.2f}s: CoM_z={c[2]:.4f}m, Fz={fz:.0f}N, base_z={q[2]:.4f}")

    return {
        "time": time_arr,
        "com": com_traj,
        "base": base_traj,
        "tau": tau_traj,
        "contact_fz": fz_traj,
        "dt": dt,
    }
