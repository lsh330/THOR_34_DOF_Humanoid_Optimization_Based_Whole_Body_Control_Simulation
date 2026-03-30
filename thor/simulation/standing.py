"""
Static standing simulation scenario.

Uses gravity compensation + PD joint control as baseline,
with the option to upgrade to full WB-QP.

The standing controller computes:
    tau = g(q) + Kp*(q_des - q) + Kd*(0 - dq)

where g(q) is the gravity torque vector from RNEA(q, 0, 0).

Reference:
    Hopkins & Leonessa (2015). IJHR, Sec. IV-A.
"""

import numpy as np
from numpy.typing import NDArray

from ..model.robot_model import RobotModel
from ..model.kinematics import com_position
from ..dynamics.rnea import gravity_forces
from ..dynamics.crba import crba


def default_standing_config(model: RobotModel) -> NDArray:
    """Generate default standing configuration."""
    n_joints = model.n_bodies - 1
    q = np.zeros(7 + n_joints)
    q[2] = 0.85    # Base height
    q[3] = 1.0     # Quaternion w = 1

    # Slight knee bend for natural posture
    # l_leg: hip_p(20), kn_p(21), an_p(22)
    q[7 + 20] = -0.25
    q[7 + 21] = 0.5
    q[7 + 22] = -0.25

    # r_leg: hip_p(26), kn_p(27), an_p(28)
    q[7 + 26] = -0.25
    q[7 + 27] = 0.5
    q[7 + 28] = -0.25

    return q


def run_standing_simulation(
    model: RobotModel,
    t_final: float = 3.0,
    dt: float = 0.002,
) -> dict:
    """Run static standing simulation with gravity compensation + PD.

    The floating base is constrained (feet on ground), so we simulate
    only the joint dynamics with the base fixed at its initial pose.

    This simplification is valid for static standing and avoids
    the full floating-base contact dynamics complexity.
    """
    n_dof = model.n_dof
    n_joints = n_dof - 6
    n_steps = int(t_final / dt) + 1

    # Configuration
    q = default_standing_config(model)
    v = np.zeros(n_dof)
    q_des = q.copy()  # Target: maintain initial posture

    # PD gains
    Kp = np.ones(n_joints) * 200.0   # Proportional gain
    Kd = np.ones(n_joints) * 20.0    # Derivative gain

    # Higher gains for legs (load-bearing)
    for i in range(n_joints):
        link = model.links[i + 1]
        if "leg" in link.name:
            Kp[i] = 500.0
            Kd[i] = 50.0
        elif "arm" in link.name:
            Kp[i] = 100.0
            Kd[i] = 10.0

    # Storage
    time_arr = np.linspace(0, t_final, n_steps)
    com_traj = np.empty((n_steps, 3))
    tau_traj = np.empty((n_steps, n_joints))
    joint_err_traj = np.empty(n_steps)
    energy_traj = np.empty(n_steps)

    com_0 = com_position(q, model)
    print(f"  Standing simulation: {n_steps} steps, dt={dt}s")
    print(f"  Initial CoM: [{com_0[0]:.4f}, {com_0[1]:.4f}, {com_0[2]:.4f}]")

    for step in range(n_steps):
        # Current CoM
        com_cur = com_position(q, model)
        com_traj[step] = com_cur

        # Gravity compensation
        g = gravity_forces(model, q)
        g_joints = g[6:]  # Joint torques for gravity compensation

        # PD control
        q_err = q[7:] - q_des[7:]
        dq = v[6:]
        tau_pd = -Kp * q_err - Kd * dq

        # Total torque: gravity compensation + PD
        tau = g_joints + tau_pd

        # Clamp torques to actuator limits
        for i in range(n_joints):
            if i + 1 < model.n_bodies:
                tau_lim = model.links[i + 1].tau_max
                tau[i] = np.clip(tau[i], -tau_lim, tau_lim)

        tau_traj[step] = tau
        joint_err_traj[step] = np.linalg.norm(q_err)

        # Compute mass matrix for forward dynamics
        M = crba(model, q)

        # Use joint-joint block only (base is fixed/constrained)
        # M_jj * ddq_j = tau - h_j (with ddq_base = 0)
        M_jj = M[6:, 6:]
        h_j = g[6:]  # Gravity on joints (at low velocity, bias ≈ gravity)

        rhs_j = tau - h_j

        try:
            ddq_j = np.linalg.solve(M_jj, rhs_j)
        except np.linalg.LinAlgError:
            ddq_j = np.zeros(n_joints)

        # Full acceleration (base fixed)
        ddq = np.zeros(n_dof)
        ddq[6:] = ddq_j

        # Energy
        KE = 0.5 * v @ M @ v
        PE = model.total_mass * 9.81 * com_cur[2]
        energy_traj[step] = KE + PE

        # Semi-implicit Euler
        v += dt * ddq
        v[:6] = 0.0  # Base fixed

        # Update joint angles
        q[7:] += dt * v[6:]

        if step % (n_steps // 6) == 0:
            print(f"    t={time_arr[step]:5.2f}s: CoM_z={com_cur[2]:.4f}m, "
                  f"joint_err={joint_err_traj[step]:.4f}rad, "
                  f"tau_rms={np.sqrt(np.mean(tau**2)):.1f}Nm")

    return {
        "time": time_arr,
        "com_trajectory": com_traj,
        "torques": tau_traj,
        "joint_error": joint_err_traj,
        "energy": energy_traj,
        "com_initial": com_0,
        "n_dof": n_dof,
        "dt": dt,
    }
