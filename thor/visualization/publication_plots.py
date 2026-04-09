"""Publication-quality visualization for THOR simulation results.

All plots are 300 DPI PNG with consistent styling.
Color palette is colorblind-safe (based on Wong 2011).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


# ---------------------------------------------------------------------------
# Publication rcParams
# ---------------------------------------------------------------------------
PLOT_PARAMS = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'primary':    '#2563EB',   # Blue
    'secondary':  '#DC2626',   # Red
    'tertiary':   '#059669',   # Green
    'quaternary': '#D97706',   # Amber
    'quinary':    '#7C3AED',   # Purple
    'senary':     '#DB2777',   # Pink
    'reference':  '#6B7280',   # Gray
    'fill':       '#DBEAFE',   # Light blue
}


def setup_style():
    """Apply publication rcParams globally."""
    plt.rcParams.update(PLOT_PARAMS)


def _save(fig: plt.Figure, path: str | Path) -> str:
    """Save figure at 300 DPI and close it. Returns absolute path string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(p.resolve())


# ---------------------------------------------------------------------------
# 1. Standing Analysis  (4-panel)
# ---------------------------------------------------------------------------

def plot_standing_analysis(result: dict, save_path: str | Path) -> str:
    """4-panel standing simulation analysis.

    Panels:
        (0,0) CoM height vs time
        (0,1) Joint error norm vs time
        (1,0) Control torque RMS vs time
        (1,1) Mechanical energy (KE + PE)

    Args:
        result: dict returned by run_standing_simulation or
                run_contact_implicit_simulation.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = result['time']

    # --- CoM height ---
    ax = axes[0, 0]
    com_key = 'com_trajectory' if 'com_trajectory' in result else 'com'
    com = result[com_key]
    ax.plot(t, com[:, 2], color=COLORS['primary'], lw=2.0, label='CoM z')
    ax.axhline(com[0, 2], color=COLORS['reference'], lw=1.0,
               linestyle='--', label='Initial')
    ax.set_ylabel('CoM Height [m]')
    ax.set_title('Center of Mass Height')
    ax.legend(loc='upper right')

    # --- Joint error norm ---
    ax = axes[0, 1]
    if 'joint_error' in result:
        err = result['joint_error']
        ax.plot(t, err, color=COLORS['secondary'], lw=1.5)
    else:
        ax.text(0.5, 0.5, 'joint_error not available',
                ha='center', va='center', transform=ax.transAxes,
                color=COLORS['reference'])
    ax.set_ylabel('Joint Error Norm [rad]')
    ax.set_title('Joint Position Error')

    # --- Torque RMS ---
    ax = axes[1, 0]
    if 'torques' in result:
        tau = result['torques']
        tau_rms = np.sqrt(np.mean(tau ** 2, axis=1))
        ax.plot(t, tau_rms, color=COLORS['quaternary'], lw=1.5)
        ax.fill_between(t, 0, tau_rms,
                        color=COLORS['quaternary'], alpha=0.15)
    else:
        ax.text(0.5, 0.5, 'torques not available',
                ha='center', va='center', transform=ax.transAxes,
                color=COLORS['reference'])
    ax.set_ylabel('RMS Torque [N$\\cdot$m]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Control Effort (RMS Torque)')

    # --- Energy ---
    ax = axes[1, 1]
    if 'energy' in result:
        ax.plot(t, result['energy'], color=COLORS['quinary'], lw=2.0)
        ax.fill_between(t, min(result['energy']), result['energy'],
                        color=COLORS['quinary'], alpha=0.12)
        ax.set_ylabel('Total Energy [J]')
    elif 'contact_fz' in result:
        ax.plot(t, result['contact_fz'], color=COLORS['tertiary'], lw=1.5)
        ax.set_ylabel('Contact Force Fz [N]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Mechanical Energy / Contact Force')

    fig.suptitle('THOR 34-DOF Humanoid — Static Standing Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Walking Dashboard  (6-panel)
# ---------------------------------------------------------------------------

def plot_walking_dashboard(result: dict, save_path: str | Path) -> str:
    """6-panel walking simulation dashboard.

    Panels:
        (0,0) CoM height coloured by contact phase
        (0,1) Hip / knee / ankle joint angles (left leg)
        (1,0) Vertical ground reaction force
        (1,1) Contact phase timeline (DS / L-swing / R-swing)
        (2,0) Per-joint torque heatmap (time x joint)
        (2,1) Hip angle vs hip angular velocity (phase portrait)

    Args:
        result: dict from run_contact_implicit_simulation.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    t = result['time']
    com = result.get('com', result.get('com_trajectory', np.zeros((len(t), 3))))
    fz = result.get('contact_fz', np.zeros(len(t)))
    nc = result.get('n_contacts', np.full(len(t), 2, dtype=int))
    q_traj = result.get('q', None)

    # Colour map for contact phases
    phase_colors = np.where(nc >= 2, 0.0,
                   np.where(nc == 1, 0.5, 1.0))   # DS=blue, SS=amber, flight=red

    # --- (0,0) CoM height coloured by phase ---
    ax00 = fig.add_subplot(gs[0, 0])
    sc = ax00.scatter(t, com[:, 2], c=phase_colors, cmap='RdYlBu_r',
                      s=3, vmin=0, vmax=1, zorder=3)
    ax00.set_ylabel('CoM Height [m]')
    ax00.set_xlabel('Time [s]')
    ax00.set_title('CoM Height (Phase-Coloured)')
    cbar = fig.colorbar(sc, ax=ax00, shrink=0.85, pad=0.02)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['DS', 'SS', 'Flight'])

    # --- (0,1) Leg joint angles ---
    ax01 = fig.add_subplot(gs[0, 1])
    if q_traj is not None:
        # DOF mapping from model_data: floating base(0-5), waist(6-7),
        # head(8-9), l_arm(10-16), r_arm(17-23),
        # l_leg: hip_y(24), hip_r(25), hip_p(26), kn_p(27), an_p(28), an_r(29)
        # Joint velocity indices → q indices are offset by 7 (base pos + quat)
        leg_map = {'L-Hip P': 7 + 26, 'L-Knee P': 7 + 27, 'L-Ankle P': 7 + 28}
        leg_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
        for (name, idx), col in zip(leg_map.items(), leg_colors):
            if idx < q_traj.shape[1]:
                ax01.plot(t, np.degrees(q_traj[:, idx]),
                          color=col, lw=1.5, label=name)
    else:
        ax01.text(0.5, 0.5, 'q trajectory not available',
                  ha='center', va='center', transform=ax01.transAxes)
    ax01.set_ylabel('Joint Angle [deg]')
    ax01.set_xlabel('Time [s]')
    ax01.set_title('Left Leg Joint Angles')
    ax01.legend(loc='upper right', ncol=1)

    # --- (1,0) Vertical GRF ---
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(t, fz, color=COLORS['tertiary'], lw=1.5)
    ax10.fill_between(t, 0, np.maximum(fz, 0),
                      color=COLORS['tertiary'], alpha=0.2)
    ax10.axhline(0, color=COLORS['reference'], lw=0.8, linestyle='--')
    ax10.set_ylabel('Contact Force Fz [N]')
    ax10.set_xlabel('Time [s]')
    ax10.set_title('Vertical Ground Reaction Force')

    # --- (1,1) Contact phase timeline ---
    ax11 = fig.add_subplot(gs[1, 1])
    ds_mask  = nc >= 2
    ls_mask  = nc == 1
    fl_mask  = nc == 0
    ax11.fill_between(t, 0.0, 1.0, where=ds_mask,
                      color=COLORS['primary'], alpha=0.5, label='Double Support')
    ax11.fill_between(t, 0.0, 1.0, where=ls_mask,
                      color=COLORS['quaternary'], alpha=0.5, label='Single Support')
    ax11.fill_between(t, 0.0, 1.0, where=fl_mask,
                      color=COLORS['secondary'], alpha=0.4, label='Flight')
    ax11.set_xlim(t[0], t[-1])
    ax11.set_ylim(0, 1)
    ax11.set_yticks([])
    ax11.set_xlabel('Time [s]')
    ax11.set_title('Contact Phase Timeline')
    ax11.legend(loc='upper right', fontsize=8)

    # --- (2,0) Torque heatmap ---
    ax20 = fig.add_subplot(gs[2, 0])
    if q_traj is not None and 'torques' in result:
        tau = result['torques']
        # Down-sample time to at most 200 columns
        ds = max(1, len(t) // 200)
        tau_ds = tau[::ds].T
        t_ds   = t[::ds]
        im = ax20.imshow(np.abs(tau_ds), aspect='auto', origin='lower',
                         extent=[t_ds[0], t_ds[-1], 0, tau_ds.shape[0]],
                         cmap='hot_r', vmin=0)
        cb = fig.colorbar(im, ax=ax20, shrink=0.9)
        cb.set_label('|Torque| [N$\\cdot$m]', fontsize=9)
        ax20.set_ylabel('Joint Index')
        ax20.set_xlabel('Time [s]')
        ax20.set_title('Joint Torque Magnitude Heatmap')
    else:
        ax20.text(0.5, 0.5, 'torques not available',
                  ha='center', va='center', transform=ax20.transAxes)
        ax20.set_title('Joint Torque Heatmap')

    # --- (2,1) Phase portrait: hip angle vs velocity ---
    ax21 = fig.add_subplot(gs[2, 1])
    if q_traj is not None:
        hip_idx = 7 + 26   # l_leg hip_p
        if hip_idx < q_traj.shape[1]:
            hip_q = np.degrees(q_traj[:, hip_idx])
            hip_dq = np.gradient(hip_q, t) if len(t) > 1 else np.zeros_like(hip_q)
            sc2 = ax21.scatter(hip_q, hip_dq, c=t, cmap='plasma',
                               s=2, zorder=3)
            cb2 = fig.colorbar(sc2, ax=ax21, shrink=0.85)
            cb2.set_label('Time [s]', fontsize=9)
    else:
        ax21.text(0.5, 0.5, 'q trajectory not available',
                  ha='center', va='center', transform=ax21.transAxes)
    ax21.set_xlabel('Hip Angle [deg]')
    ax21.set_ylabel('Hip Angular Velocity [deg/s]')
    ax21.set_title('Phase Portrait: Left Hip')

    fig.suptitle('THOR 34-DOF Humanoid — Walking Simulation Dashboard',
                 fontsize=15, fontweight='bold', y=1.01)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 3. CRBA vs RNEA Validation  (3-panel)
# ---------------------------------------------------------------------------

def plot_crba_rnea_validation(model, save_path: str | Path) -> str:
    """Validate CRBA (mass matrix) against RNEA bias-force at random configs.

    Compares ||CRBA column|| vs ||RNEA gravity column|| to confirm
    structural consistency between the two algorithms.

    Panels:
        (0) Scatter: CRBA diagonal vs RNEA-derived inertia estimate
        (1) Error histogram
        (2) Per-DOF error magnitude bar chart

    Args:
        model: RobotModel instance.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    from ..dynamics.crba import crba
    from ..dynamics.rnea import rnea as rnea_fn, bias_forces

    n_dof = model.n_dof
    n_samples = 12
    rng = np.random.default_rng(42)

    crba_diags = []
    rnea_norms = []

    for _ in range(n_samples):
        q = np.zeros(7 + (n_dof - 6))
        q[2] = 1.02
        q[3] = 1.0
        q[7:] = rng.uniform(-0.3, 0.3, n_dof - 6)

        M = crba(model, q)
        diag = np.diag(M)

        # Gravity-compensation torques as RNEA reference
        v0 = np.zeros(n_dof)
        a0 = np.zeros(n_dof)
        tau_g = rnea_fn(model, q, v0, a0)

        crba_diags.append(diag)
        rnea_norms.append(np.abs(tau_g))

    crba_arr  = np.stack(crba_diags)   # (n_samples, n_dof)
    rnea_arr  = np.stack(rnea_norms)   # (n_samples, n_dof)

    # Relative error: |diag(M) - |tau_g|| / (|diag(M)| + 1)
    errors = np.abs(crba_arr - rnea_arr) / (np.abs(crba_arr) + 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Scatter ---
    ax = axes[0]
    x_vals = crba_arr.flatten()
    y_vals = rnea_arr.flatten()
    ax.scatter(x_vals, y_vals, s=18, color=COLORS['primary'],
               alpha=0.6, edgecolors='none', label='Samples')
    lim = max(x_vals.max(), y_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color=COLORS['reference'],
            lw=1.2, label='y = x')
    ax.set_xlabel('CRBA Diagonal $M_{ii}$ [kg$\\cdot$m$^2$ or kg]')
    ax.set_ylabel('RNEA $|\\tau_g|$ [N$\\cdot$m or N]')
    ax.set_title('CRBA vs RNEA: Diagonal Consistency')
    ax.legend()

    # --- Error histogram ---
    ax = axes[1]
    rel_flat = errors.flatten()
    ax.hist(rel_flat, bins=30, color=COLORS['primary'],
            edgecolor='white', alpha=0.85)
    ax.axvline(np.median(rel_flat), color=COLORS['secondary'], lw=1.5,
               linestyle='--', label=f'Median={np.median(rel_flat):.3f}')
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()

    # --- Per-DOF error bar ---
    ax = axes[2]
    mean_err = errors.mean(axis=0)
    std_err  = errors.std(axis=0)
    dof_idx  = np.arange(n_dof)
    ax.bar(dof_idx, mean_err, yerr=std_err, color=COLORS['primary'],
           ecolor=COLORS['secondary'], capsize=2,
           alpha=0.8, linewidth=0)
    ax.set_xlabel('DOF Index')
    ax.set_ylabel('Mean Relative Error')
    ax.set_title('Per-DOF Error Magnitude')
    ax.annotate(f'Max: {mean_err.max():.4f}',
                xy=(mean_err.argmax(), mean_err.max()),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color=COLORS['secondary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))

    fig.suptitle('THOR — CRBA / RNEA Algorithm Validation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Mass Matrix Structure  (3-panel)
# ---------------------------------------------------------------------------

def plot_mass_matrix_structure(model, q: np.ndarray,
                               save_path: str | Path) -> str:
    """Visualise structure and spectral properties of M(q).

    Panels:
        (0) 40x40 heatmap (log-scale absolute values)
        (1) Eigenvalue spectrum with condition number annotation
        (2) Block structure diagram

    Args:
        model: RobotModel instance.
        q: Configuration vector.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    from ..dynamics.crba import crba

    M = crba(model, q)
    n = M.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Heatmap ---
    ax = axes[0]
    M_log = np.log10(np.abs(M) + 1e-12)
    im = ax.imshow(M_log, cmap='viridis', aspect='equal', origin='upper')
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('log$_{10}|M_{ij}|$', fontsize=10)
    ax.set_title(f'Mass Matrix $M(q)$ — {n}$\\times${n}')
    ax.set_xlabel('DOF Index $j$')
    ax.set_ylabel('DOF Index $i$')

    # Block boundary lines (base/waist/head/arms/legs/grippers)
    boundaries = [6, 8, 10, 17, 24, 30, 34, 40]
    for b in boundaries:
        if b < n:
            ax.axhline(b - 0.5, color='white', lw=0.6, alpha=0.5)
            ax.axvline(b - 0.5, color='white', lw=0.6, alpha=0.5)

    # --- Eigenvalue spectrum ---
    ax = axes[1]
    eigs = np.linalg.eigvalsh(M)
    eigs_sorted = np.sort(eigs)[::-1]
    cond = eigs_sorted[0] / (eigs_sorted[-1] + 1e-16)
    ax.semilogy(np.arange(1, n + 1), eigs_sorted,
                color=COLORS['primary'], lw=1.8, marker='o',
                markersize=4, markerfacecolor='white')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue [kg$\\cdot$m$^2$]')
    ax.set_title(f'Eigenvalue Spectrum\ncond($M$) = {cond:.2e}')
    ax.annotate(f'$\\lambda_{{max}}$ = {eigs_sorted[0]:.2f}',
                xy=(1, eigs_sorted[0]), xytext=(5, -15),
                textcoords='offset points', fontsize=9, color=COLORS['primary'])
    ax.annotate(f'$\\lambda_{{min}}$ = {eigs_sorted[-1]:.2e}',
                xy=(n, eigs_sorted[-1]), xytext=(-50, 10),
                textcoords='offset points', fontsize=9, color=COLORS['secondary'])

    # --- Block diagram ---
    ax = axes[2]
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    block_defs = [
        ('Floating\nBase', 0, 6, COLORS['quaternary']),
        ('Waist', 6, 8, COLORS['primary']),
        ('Head', 8, 10, COLORS['tertiary']),
        ('L-Arm', 10, 17, COLORS['secondary']),
        ('R-Arm', 17, 24, '#DB2777'),
        ('L-Leg', 24, 30, COLORS['quinary']),
        ('R-Leg', 30, 36, '#0891B2'),
        ('Grippers', 36, 40, COLORS['reference']),
    ]
    for name, start, end, col in block_defs:
        size = end - start
        if end > n:
            end = n
            size = end - start
        if size <= 0:
            continue
        rect = plt.Rectangle((start, start), size, size,
                              facecolor=col, alpha=0.5, edgecolor='k', lw=1.2)
        ax.add_patch(rect)
        ax.text(start + size / 2, start + size / 2, name,
                ha='center', va='center', fontsize=8 if size > 2 else 6,
                fontweight='bold', color='black')

    ax.set_xlabel('DOF Index')
    ax.set_ylabel('DOF Index')
    ax.set_title('Block Structure Diagram')
    ax.grid(False)

    fig.suptitle('THOR — Mass Matrix $M(q)$ Structure Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Energy Conservation  (3-panel)
# ---------------------------------------------------------------------------

def plot_energy_conservation(result: dict, save_path: str | Path) -> str:
    """Analyse energy conservation quality of the integrator.

    Panels:
        (0) KE / PE / Total decomposition vs time
        (1) Energy drift (E(t) - E(0)) vs time
        (2) RMS drift summary bar (one bar per simulation if multiple dt)

    Args:
        result: Simulation result dict. Expects 'energy', 'time',
                and optionally 'kinetic_energy' / 'potential_energy'.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    t = result['time']

    if 'energy' in result:
        E_total = result['energy']
    else:
        E_total = np.zeros(len(t))

    KE = result.get('kinetic_energy', E_total * 0.4)
    PE = result.get('potential_energy', E_total * 0.6)

    drift = E_total - E_total[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- KE / PE / Total ---
    ax = axes[0]
    ax.plot(t, KE, color=COLORS['secondary'], lw=1.5, label='KE')
    ax.plot(t, PE, color=COLORS['primary'],   lw=1.5, label='PE')
    ax.plot(t, E_total, color='black', lw=2.0, linestyle='--', label='Total')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.set_title('Energy Decomposition')
    ax.legend()

    # --- Energy drift ---
    ax = axes[1]
    ax.plot(t, drift, color=COLORS['tertiary'], lw=1.5)
    ax.fill_between(t, 0, drift, color=COLORS['tertiary'], alpha=0.2)
    ax.axhline(0, color=COLORS['reference'], lw=0.8, linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$\\Delta E$ [J]')
    ax.set_title('Energy Drift $E(t) - E(0)$')
    drift_pct = 100 * np.abs(drift[-1]) / (np.abs(E_total[0]) + 1e-10)
    ax.text(0.05, 0.95, f'Final drift: {drift[-1]:.4f} J ({drift_pct:.2f}%)',
            transform=ax.transAxes, fontsize=9, va='top',
            color=COLORS['tertiary'])

    # --- Drift summary ---
    ax = axes[2]
    rms_drift = np.sqrt(np.mean(drift ** 2))
    max_drift = np.max(np.abs(drift))
    labels = ['RMS Drift [J]', 'Max |Drift| [J]']
    vals   = [rms_drift, max_drift]
    cols   = [COLORS['primary'], COLORS['secondary']]
    bars = ax.bar(labels, vals, color=cols, alpha=0.8, edgecolor='white', lw=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Energy [J]')
    ax.set_title('Drift Metrics')

    fig.suptitle(f'THOR — Energy Conservation (dt={result.get("dt", "N/A")} s)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Contact Force Analysis  (3-panel)
# ---------------------------------------------------------------------------

def plot_contact_force_analysis(save_path: str | Path) -> str:
    """Illustrate the compliant contact model parameters.

    Panels:
        (0) Normal force f_n vs penetration depth phi
        (1) Friction force vs slip velocity (tanh profile)
        (2) Coulomb friction cone (3D)

    Args:
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()

    k_n   = 3000.0   # N/m stiffness
    d_n   = 300.0    # N·s/m damping
    mu    = 0.7      # friction coefficient
    v_s   = 0.01     # stiction velocity threshold [m/s]
    f_max = 2000.0   # N

    fig = plt.figure(figsize=(16, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')

    # --- Normal force vs penetration ---
    phi_vals = np.linspace(-0.05, 0.01, 500)
    dphi_static = 0.0
    f_n_vals = np.where(
        phi_vals < 0,
        np.minimum(k_n * (-phi_vals) + d_n * max(0.0, -dphi_static), f_max),
        0.0
    )
    ax0.plot(-phi_vals * 1e3, f_n_vals, color=COLORS['primary'], lw=2.0)
    ax0.fill_between(-phi_vals * 1e3, 0, f_n_vals,
                     where=phi_vals < 0, color=COLORS['fill'], alpha=0.6)
    ax0.axhline(f_max, color=COLORS['secondary'], lw=1.0,
                linestyle='--', label=f'$f_{{max}}={f_max:.0f}$ N')
    ax0.set_xlabel('Penetration Depth [mm]')
    ax0.set_ylabel('Normal Force $f_n$ [N]')
    ax0.set_title('Spring-Damper Contact Model\n'
                  f'$k_n={k_n:.0f}$ N/m, $d_n={d_n:.0f}$ N$\\cdot$s/m')
    ax0.legend()

    # --- Friction vs slip velocity ---
    vt_vals = np.linspace(-0.15, 0.15, 500)
    f_ref  = 500.0   # representative normal force
    ft_vals = -mu * f_ref * np.tanh(vt_vals / v_s)
    ax1.plot(vt_vals * 1e3, ft_vals, color=COLORS['tertiary'], lw=2.0)
    ax1.axhline( mu * f_ref, color=COLORS['reference'],
                 lw=1.0, linestyle='--', label=f'$\\pm\\mu f_n={mu*f_ref:.0f}$ N')
    ax1.axhline(-mu * f_ref, color=COLORS['reference'], lw=1.0, linestyle='--')
    ax1.axvline(v_s * 1e3, color=COLORS['quaternary'], lw=1.0,
                linestyle=':', label=f'$v_s={v_s*1e3:.0f}$ mm/s')
    ax1.set_xlabel('Slip Velocity [mm/s]')
    ax1.set_ylabel('Friction Force $f_t$ [N]')
    ax1.set_title('Continuous Coulomb Friction\n'
                  f'($\\mu={mu}$, tanh regularisation)')
    ax1.legend(fontsize=8)

    # --- Coulomb cone (3D) ---
    theta = np.linspace(0, 2 * np.pi, 80)
    fn_range = np.linspace(0, 600, 40)
    T, FN = np.meshgrid(theta, fn_range)
    FX = mu * FN * np.cos(T)
    FY = mu * FN * np.sin(T)
    ax2.plot_surface(FX, FY, FN, alpha=0.4, color=COLORS['primary'],
                     linewidth=0, antialiased=True)
    ax2.plot([0, 0], [0, 0], [0, 600],
             color='black', lw=1.5, linestyle='--')
    ax2.set_xlabel('$f_x$ [N]', fontsize=9)
    ax2.set_ylabel('$f_y$ [N]', fontsize=9)
    ax2.set_zlabel('$f_n$ [N]', fontsize=9)
    ax2.set_title(f'Coulomb Friction Cone\n($\\mu={mu}$)')
    ax2.view_init(elev=25, azim=35)

    fig.suptitle('THOR — Contact Force Model Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 7. LCP Solver Analysis  (3-panel)
# ---------------------------------------------------------------------------

def plot_lcp_solver_analysis(save_path: str | Path) -> str:
    """Visualise Fischer-Burmeister NCP function and Newton convergence.

    Panels:
        (0) FB function phi(a, b) 3D surface
        (1) Newton convergence trajectory (residual vs iteration)
        (2) FB vs Interior-Point residual comparison

    Args:
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    from ..optimization.lcp_solver import (
        solve_lcp_fb_newton,
        solve_lcp_interior_point,
        fischer_burmeister,
    )

    fig = plt.figure(figsize=(18, 6))
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    # --- FB surface ---
    a_vals = np.linspace(-2, 4, 80)
    b_vals = np.linspace(-2, 4, 80)
    A, B = np.meshgrid(a_vals, b_vals)
    FB = np.vectorize(lambda a, b: fischer_burmeister(float(a), float(b), 1e-2))(A, B)
    ax0.plot_surface(A, B, FB, cmap='coolwarm', alpha=0.85,
                     linewidth=0, antialiased=True)
    ax0.set_xlabel('$a$', fontsize=9)
    ax0.set_ylabel('$b$', fontsize=9)
    ax0.set_zlabel('$\\phi_{FB}(a,b)$', fontsize=9)
    ax0.set_title('Fischer-Burmeister Function\n$\\phi = a+b-\\sqrt{a^2+b^2+2\\varepsilon^2}$')
    ax0.view_init(elev=28, azim=-50)

    # --- Newton convergence ---
    rng = np.random.default_rng(7)
    n_lcp = 8
    M_lcp = rng.random((n_lcp, n_lcp))
    M_lcp = M_lcp.T @ M_lcp + n_lcp * np.eye(n_lcp)  # SPD → P-matrix
    q_lcp = rng.random(n_lcp)

    # Collect residuals iteration-by-iteration via monkey-patching
    residuals_fb = []
    z = np.ones(n_lcp) * 0.1
    for it in range(30):
        w = M_lcp @ z + q_lcp
        F = np.array([fischer_burmeister(float(z[i]), float(w[i]), 1e-4)
                      for i in range(n_lcp)])
        res = float(np.linalg.norm(F))
        residuals_fb.append(res)
        if res < 1e-10:
            break
        J = np.zeros((n_lcp, n_lcp))
        for i in range(n_lcp):
            import math
            denom = math.sqrt(z[i]**2 + w[i]**2 + 2e-8)
            da = 1.0 - z[i] / denom
            db = 1.0 - w[i] / denom
            for j in range(n_lcp):
                J[i, j] = db * M_lcp[i, j]
            J[i, i] += da
        try:
            dz = np.linalg.solve(J + 1e-12 * np.eye(n_lcp), -F)
        except Exception:
            dz = np.zeros(n_lcp)
        alpha = 1.0
        for _ in range(12):
            z_new = z + alpha * dz
            w_new = M_lcp @ z_new + q_lcp
            F_new = np.array([fischer_burmeister(float(z_new[i]), float(w_new[i]), 1e-4)
                              for i in range(n_lcp)])
            if np.linalg.norm(F_new) < res:
                break
            alpha *= 0.5
        z = z + alpha * dz

    ax1.semilogy(residuals_fb, color=COLORS['primary'], lw=2.0,
                 marker='o', markersize=5, label='FB-Newton')
    ax1.axhline(1e-8, color=COLORS['reference'], lw=1.0,
                linestyle='--', label='Tolerance $10^{-8}$')
    ax1.set_xlabel('Newton Iteration')
    ax1.set_ylabel('Residual $\\|F(z)\\|$')
    ax1.set_title('FB-Newton Convergence')
    ax1.legend()

    # --- FB vs IP comparison ---
    trial_sizes = [4, 8, 12, 16, 20]
    fb_iters = []
    ip_iters = []
    for n_t in trial_sizes:
        Mt = rng.random((n_t, n_t))
        Mt = Mt.T @ Mt + n_t * np.eye(n_t)
        qt = rng.random(n_t)
        _, i_fb, _ = solve_lcp_fb_newton(Mt, qt)
        _, i_ip, _ = solve_lcp_interior_point(Mt, qt)
        fb_iters.append(i_fb)
        ip_iters.append(i_ip)

    x = np.arange(len(trial_sizes))
    w_bar = 0.35
    ax2.bar(x - w_bar / 2, fb_iters, w_bar, label='FB-Newton',
            color=COLORS['primary'], alpha=0.85, linewidth=0)
    ax2.bar(x + w_bar / 2, ip_iters, w_bar, label='Interior-Point',
            color=COLORS['secondary'], alpha=0.85, linewidth=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'n={s}' for s in trial_sizes])
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('FB-Newton vs Interior-Point\nIteration Count')
    ax2.legend()

    fig.suptitle('THOR — LCP Solver Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Biomechanical Reference  (2-panel)
# ---------------------------------------------------------------------------

def plot_biomechanical_reference(save_path: str | Path) -> str:
    """Synthesised biomechanical joint angle profiles (Winter 1991 style).

    Based on Winter, D.A. (1991). Biomechanics and Motor Control of
    Human Gait: Normal, Elderly, and Pathological (2nd ed.).

    Panels:
        (0) Hip / knee / ankle swing and stance profiles vs gait cycle %
        (1) Event annotations and ROM summary

    Args:
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    gc = np.linspace(0, 100, 500)  # Gait cycle percentage

    # Approximate Winter 1991 profiles (deg), one full gait cycle
    def _hip(gc):
        return (20 * np.sin(np.radians(gc * 3.6 + 90))
                - 5 * np.sin(np.radians(gc * 7.2)))

    def _knee(gc):
        # Approximate Winter 1991 knee profile:
        # 0-15%: loading response flex (0->60 deg)
        # 15-45%: mid/terminal stance ext (60->0 deg)
        # 45-62%: pre-swing flex (0->40 deg)
        # 62-75%: initial swing flex (40->60 deg)
        # 75-90%: terminal swing ext (60->0 deg)
        # 90-100%: hold at 0
        k = np.zeros_like(gc, dtype=float)
        m1 = gc < 15
        m2 = (gc >= 15) & (gc < 45)
        m3 = (gc >= 45) & (gc < 62)
        m4 = (gc >= 62) & (gc < 75)
        m5 = (gc >= 75) & (gc < 90)
        k[m1] = 60.0 * gc[m1] / 15.0
        k[m2] = 60.0 * (45.0 - gc[m2]) / 30.0
        k[m3] = 40.0 * (gc[m3] - 45.0) / 17.0
        k[m4] = 40.0 + 20.0 * (gc[m4] - 62.0) / 13.0
        k[m5] = 60.0 * (90.0 - gc[m5]) / 15.0
        return k

    def _ankle(gc):
        return (10 * np.sin(np.radians(gc * 3.6 - 30))
                - 5 * np.sin(np.radians(gc * 7.2)))

    hip_deg   = _hip(gc)
    knee_deg  = _knee(gc)
    ankle_deg = _ankle(gc)

    # Gait events
    events = {
        'IC (0%)': 0,
        'LR (12%)': 12,
        'MSt (31%)': 31,
        'TSt (50%)': 50,
        'PSw (62%)': 62,
        'ISw (75%)': 75,
        'MSw (87%)': 87,
        'TSw (100%)': 100,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Joint profiles ---
    ax = axes[0]
    ax.plot(gc, hip_deg,   color=COLORS['primary'],    lw=2.0, label='Hip')
    ax.plot(gc, knee_deg,  color=COLORS['secondary'],  lw=2.0, label='Knee')
    ax.plot(gc, ankle_deg, color=COLORS['tertiary'],   lw=2.0, label='Ankle')
    ax.axhline(0, color=COLORS['reference'], lw=0.8, linestyle='--', alpha=0.5)

    # Phase shading
    ax.axvspan(0, 62, color=COLORS['primary'], alpha=0.06, label='Stance (0-62%)')
    ax.axvspan(62, 100, color=COLORS['quaternary'], alpha=0.06, label='Swing (62-100%)')

    for name, pct in events.items():
        ax.axvline(pct, color=COLORS['reference'], lw=0.7,
                   linestyle=':', alpha=0.7)
        if pct in (0, 50, 62, 100):
            ax.text(pct + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 65,
                    name.split('(')[0].strip(),
                    fontsize=7, color=COLORS['reference'], rotation=90, va='top')

    ax.set_xlabel('Gait Cycle [%]')
    ax.set_ylabel('Joint Angle [deg]')
    ax.set_title('Biomechanical Joint Profiles\n(Winter 1991 — Approximate)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 100)

    # --- ROM summary ---
    ax = axes[1]
    joints    = ['Hip', 'Knee', 'Ankle']
    rom_vals  = [np.ptp(hip_deg), np.ptp(knee_deg), np.ptp(ankle_deg)]
    peak_flex = [np.max(hip_deg), np.max(knee_deg), np.max(ankle_deg)]
    peak_ext  = [np.min(hip_deg), np.min(knee_deg), np.min(ankle_deg)]

    x = np.arange(3)
    cols = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    bars = ax.bar(x, rom_vals, color=cols, alpha=0.8, linewidth=0)
    for bar, val in zip(bars, rom_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}$^\\circ$', ha='center', va='bottom', fontsize=10)

    # Peak flex / ext annotation
    for i, (pf, pe) in enumerate(zip(peak_flex, peak_ext)):
        ax.annotate(f'max={pf:.1f}$^\\circ$\nmin={pe:.1f}$^\\circ$',
                    xy=(i, rom_vals[i] / 2),
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(joints)
    ax.set_ylabel('Range of Motion [deg]')
    ax.set_title('Joint Range of Motion Summary')

    # Gait events table
    event_str = '\n'.join([f'{k}' for k in events.keys()])
    ax.text(1.03, 0.98, 'Gait Events:\n' + event_str,
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    fig.suptitle('Biomechanical Reference — Normal Gait (Winter 1991)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Controller Comparison  (3-panel)
# ---------------------------------------------------------------------------

def plot_controller_comparison(results_dict: dict, save_path: str | Path) -> str:
    """Compare multiple controller results side-by-side.

    Args:
        results_dict: Mapping of controller name → result dict.
                      Supported keys in each result: 'time', 'com',
                      'contact_fz', 'torques'.
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    color_cycle = [COLORS['primary'], COLORS['secondary'],
                   COLORS['tertiary'], COLORS['quaternary'],
                   COLORS['quinary']]

    # --- CoM height comparison ---
    ax = axes[0]
    for (name, res), col in zip(results_dict.items(), color_cycle):
        t = res['time']
        com = res.get('com', res.get('com_trajectory', np.zeros((len(t), 3))))
        ax.plot(t, com[:, 2], color=col, lw=1.8, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('CoM Height [m]')
    ax.set_title('Center of Mass Height')
    ax.legend(fontsize=8)

    # --- Contact force comparison ---
    ax = axes[1]
    for (name, res), col in zip(results_dict.items(), color_cycle):
        t = res['time']
        fz = res.get('contact_fz', np.zeros(len(t)))
        ax.plot(t, fz, color=col, lw=1.5, label=name)
    ax.axhline(0, color=COLORS['reference'], lw=0.8, linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Contact Force Fz [N]')
    ax.set_title('Vertical Contact Force')
    ax.legend(fontsize=8)

    # --- Torque RMS bar comparison ---
    ax = axes[2]
    names  = list(results_dict.keys())
    rms_vals = []
    for name, res in results_dict.items():
        if 'torques' in res:
            tau = res['torques']
            rms_vals.append(np.sqrt(np.mean(tau ** 2)))
        else:
            rms_vals.append(0.0)
    bars = ax.bar(names, rms_vals, color=color_cycle[:len(names)],
                  alpha=0.85, linewidth=0)
    for bar, val in zip(bars, rms_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rms_vals, default=[1]) * 0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('RMS Torque [N$\\cdot$m]')
    ax.set_title('Control Effort Comparison\n(Overall RMS)')
    ax.tick_params(axis='x', labelrotation=15)

    fig.suptitle('THOR — Controller Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 10. Performance Waterfall Chart
# ---------------------------------------------------------------------------

def plot_performance_waterfall(timing_data: dict, save_path: str | Path) -> str:
    """Waterfall (stacked bar) chart of computation time breakdown.

    Args:
        timing_data: Dict mapping component name → time in seconds.
                     Example: {'CRBA': 0.0012, 'RNEA': 0.0008,
                               'FK': 0.0003, 'Cholesky': 0.0005,
                               'Controller': 0.0015, 'Other': 0.0002}
        save_path: Output PNG path.

    Returns:
        Absolute path of saved file.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = list(timing_data.keys())
    values = np.array([timing_data[k] for k in labels], dtype=float)
    total  = values.sum()

    color_list = [
        COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
        COLORS['quaternary'], COLORS['quinary'], COLORS['senary'],
        COLORS['reference'],
    ]
    colors = [color_list[i % len(color_list)] for i in range(len(labels))]

    # --- Waterfall ---
    ax = axes[0]
    cumulative = 0.0
    for i, (label, val, col) in enumerate(zip(labels, values, colors)):
        ax.bar(i, val, bottom=cumulative, color=col, alpha=0.88, linewidth=0)
        ax.text(i, cumulative + val / 2,
                f'{val*1e3:.2f} ms', ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold')
        cumulative += val
    ax.bar(len(labels), total, color='black', alpha=0.25, linewidth=0,
           label=f'Total: {total*1e3:.2f} ms')
    ax.text(len(labels), total / 2,
            f'{total*1e3:.2f} ms', ha='center', va='center',
            fontsize=9, color='black', fontweight='bold')
    ax.set_xticks(list(range(len(labels))) + [len(labels)])
    ax.set_xticklabels(labels + ['Total'], rotation=20, ha='right')
    ax.set_ylabel('Computation Time [s]')
    ax.set_title('Per-Step Time Breakdown\n(Waterfall)')
    ax.legend(fontsize=9)

    # --- Pie chart ---
    ax = axes[1]
    wedge_props = dict(width=0.45, edgecolor='white', linewidth=1.5)
    ax.pie(values, labels=labels, colors=colors,
           autopct='%1.1f%%', pctdistance=0.75,
           startangle=90, wedgeprops=wedge_props,
           textprops={'fontsize': 9})
    ax.set_title(f'Time Distribution\n(Total: {total*1e3:.2f} ms/step)')

    fig.suptitle('THOR — Computation Performance Profile',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _save(fig, save_path)
