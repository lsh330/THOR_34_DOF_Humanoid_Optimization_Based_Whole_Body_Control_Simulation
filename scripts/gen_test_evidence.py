"""Generate comprehensive test evidence plot (12 panels)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.crba import crba
from thor.dynamics.rnea import rnea, bias_forces
from thor.core.spatial import rot_x, rot_y, rot_z, skew, spatial_inertia
from thor.optimization.lcp_solver import solve_lcp_fb_newton
from thor.model.kinematics import com_position
from thor.control.gait.swing_trajectory import swing_leg_angles
from thor.dynamics.contact import contact_force_single
from thor.dynamics.contact_implicit import contact_implicit_step

model = RobotModel()
q0 = default_standing_config(model)
Path("output/plots").mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(4, 3, figsize=(18, 20))

# 1.1 Rotation det=1
ax = axes[0, 0]
angles = np.linspace(-np.pi, np.pi, 100)
ax.plot(np.degrees(angles), [np.linalg.det(rot_x(a)) for a in angles], label="rot_x")
ax.plot(np.degrees(angles), [np.linalg.det(rot_y(a)) for a in angles], label="rot_y", ls="--")
ax.plot(np.degrees(angles), [np.linalg.det(rot_z(a)) for a in angles], label="rot_z", ls=":")
ax.axhline(1.0, color="r", lw=0.5)
ax.set_xlabel("Angle [deg]"); ax.set_ylabel("det(R)")
ax.set_title("Rotation det=1"); ax.legend(fontsize=7); ax.grid(alpha=0.15)
ax.set_ylim(0.9999, 1.0001)

# 1.2 Skew antisymmetry
ax = axes[0, 1]
errs = [np.linalg.norm(skew(np.random.randn(3)) + skew(np.random.randn(3)).T) for _ in range(100)]
# actually test S+S^T for same vector
errs = []
for _ in range(100):
    v = np.random.randn(3); S = skew(v); errs.append(np.linalg.norm(S + S.T))
ax.hist(errs, bins=20, color="#4CAF50", alpha=0.7)
ax.set_xlabel("||S + S^T||"); ax.set_title(f"Skew Antisymmetry (max={max(errs):.1e})"); ax.grid(alpha=0.15)

# 1.3 Spatial inertia PD
ax = axes[0, 2]
eig_mins = []
for _ in range(50):
    m = np.random.uniform(0.1, 10); c = np.random.randn(3)*0.1
    I = np.diag(np.random.uniform(0.01, 1, 3))
    eigs = np.linalg.eigvalsh(spatial_inertia(m, c, I))
    eig_mins.append(eigs.min())
ax.bar(range(50), eig_mins, color="#1565C0", alpha=0.7)
ax.axhline(0, color="r", lw=1)
ax.set_xlabel("Config"); ax.set_title("Spatial Inertia min eigenvalue > 0"); ax.grid(alpha=0.15)

# 2.1 CRBA-RNEA scatter
ax = axes[1, 0]
cv, rv = [], []
for seed in range(10):
    rng = np.random.default_rng(seed)
    q = q0.copy(); q[7:] += rng.normal(0, 0.2, model.n_bodies-1)
    dq = rng.normal(0, 0.5, model.n_dof); ddq = rng.normal(0, 1.0, model.n_dof)
    M = crba(model, q); h = bias_forces(model, q, dq)
    cv.extend((M @ ddq + h).tolist()); rv.extend(rnea(model, q, dq, ddq).tolist())
ax.scatter(cv, rv, s=1, alpha=0.3, c="#1565C0")
lim = max(abs(min(cv+rv)), abs(max(cv+rv)))
ax.plot([-lim,lim],[-lim,lim],"r--",lw=0.8)
ax.set_xlabel("CRBA"); ax.set_ylabel("RNEA"); ax.set_aspect("equal")
ax.set_title(f"CRBA=RNEA (err={max(abs(np.array(cv)-np.array(rv))):.1e})"); ax.grid(alpha=0.15)

# 2.2 M symmetry
ax = axes[1, 1]
M = crba(model, q0)
ax.imshow(np.log10(np.abs(M - M.T)+1e-20), cmap="hot", aspect="equal")
ax.set_title(f"M Asymmetry (max={np.max(np.abs(M-M.T)):.1e})"); ax.set_xlabel("DOF")

# 2.3 Gravity = mg
ax = axes[1, 2]
gz = [bias_forces(model, q0, np.zeros(model.n_dof))[5]]
for _ in range(19):
    q = q0.copy(); q[7:] += np.random.randn(model.n_bodies-1)*0.1
    gz.append(bias_forces(model, q, np.zeros(model.n_dof))[5])
ax.bar(range(20), gz, color="#FF9800", alpha=0.7)
ax.axhline(model.total_mass*9.81, color="r", ls="--", label=f"mg={model.total_mass*9.81:.0f}N")
ax.set_xlabel("Config"); ax.set_title("Gravity=mg"); ax.legend(fontsize=7); ax.grid(alpha=0.15)

# 3.1 LCP z*w
ax = axes[2, 0]
z, _, _ = solve_lcp_fb_newton(np.array([[2.,1],[1,2]]), np.array([-1.,-1]))
w = np.array([[2.,1],[1,2]]) @ z + np.array([-1.,-1])
ax.bar(["z[0]","z[1]","w[0]","w[1]"], [z[0],z[1],w[0],w[1]], color=["#1565C0"]*2+["#2E7D32"]*2)
ax.set_title(f"LCP: z*w={np.dot(z,w):.1e}"); ax.grid(alpha=0.15)

# 3.2 Contact force
ax = axes[2, 1]
depths = np.linspace(0, 0.05, 50)
forces = [contact_force_single(np.array([0,0,-d]), np.zeros(3))[2] for d in depths]
ax.plot(depths*1000, forces, color="#E65100", lw=2)
ax.set_xlabel("Penetration [mm]"); ax.set_ylabel("Force [N]")
ax.set_title("Contact F vs depth"); ax.grid(alpha=0.15)

# 3.3 Swing trajectory
ax = axes[2, 2]
s = np.linspace(0, 1, 100)
hip = [np.degrees(swing_leg_angles(ss)[0]) for ss in s]
knee = [np.degrees(swing_leg_angles(ss)[1]) for ss in s]
ankle = [np.degrees(swing_leg_angles(ss)[2]) for ss in s]
ax.plot(s, hip, label="Hip", lw=2); ax.plot(s, knee, label="Knee", lw=2)
ax.plot(s, ankle, label="Ankle", lw=2)
ax.set_xlabel("Phase s"); ax.set_ylabel("[deg]")
ax.set_title("Biomechanical Ranges"); ax.legend(fontsize=7); ax.grid(alpha=0.15)

# 4.1 Performance
ax = axes[3, 0]
names = ["CRBA","RNEA","LCP"]
ts = []
for fn in [lambda: crba(model,q0), lambda: rnea(model,q0,np.zeros(model.n_dof),np.zeros(model.n_dof)),
           lambda: solve_lcp_fb_newton(np.array([[.05,.01],[.01,.05]]),np.array([-.5,-.3]))]:
    fn(); t0=time.perf_counter()
    for _ in range(20): fn()
    ts.append((time.perf_counter()-t0)/20*1000)
ax.bar(names, ts, color=["#1565C0","#2E7D32","#F57F17"], alpha=0.8)
for i,t in enumerate(ts): ax.text(i,t+0.02,f"{t:.2f}ms",ha="center",fontsize=8)
ax.set_ylabel("ms"); ax.set_title("Comp. Speed"); ax.grid(alpha=0.15, axis="y")

# 4.2 Energy conservation
ax = axes[3, 1]
q_t = q0.copy(); q_t[2] = 3.0; v_t = np.zeros(model.n_dof)
E = []
for _ in range(200):
    Mt = crba(model, q_t); com = com_position(q_t, model)
    E.append(0.5*v_t@Mt@v_t + model.total_mass*9.81*com[2])
    q_t, v_t, _, _ = contact_implicit_step(model, q_t, v_t, np.zeros(model.n_dof), 0.001)
E0 = E[0]; drift = [(e-E0)/abs(E0)*100 for e in E]
ax.plot(drift, color="#FF9800", lw=1.5)
ax.set_xlabel("Step"); ax.set_ylabel("Drift [%]")
ax.set_title(f"Energy Conservation ({max(abs(d) for d in drift):.2f}%)"); ax.grid(alpha=0.15)

# 4.3 Walking CoM
ax = axes[3, 2]
data = np.load("output/data/walking_traj.npz")
com_w = data["com"]; t_w = np.linspace(0, data["time"][-1], len(com_w))
ax.plot(t_w, com_w[:,2], color="#2E7D32", lw=1.5)
ax.set_xlabel("Time [s]"); ax.set_ylabel("CoM z [m]")
ax.set_title("Walking Stability"); ax.grid(alpha=0.15)

fig.suptitle("THOR 34-DOF: Comprehensive Test Evidence (104 tests)",
             fontsize=15, fontweight="bold", y=1.005)
plt.tight_layout()
fig.savefig("output/plots/test_evidence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Test evidence plot saved (12 panels)")
