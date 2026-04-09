"""Master script: generate all publication-quality plots and GIFs.

Usage
-----
From the THOR_Simulation project root:

    python scripts/generate_all_plots.py [options]

Options
-------
--scenario  {standing,walking,both}   Which simulation to run (default: standing)
--t-final   FLOAT                     Simulation duration [s]
--dt        FLOAT                     Integration time step [s] (default: 0.002)
--skip-gif                            Skip GIF generation (faster)
--output    DIR                       Base output directory (default: output)

Generated files
---------------
output/plots/
    01_standing_analysis.png
    02_mass_matrix_structure.png
    03_energy_conservation.png
    04_contact_force_model.png
    05_lcp_solver_analysis.png
    06_biomechanical_reference.png
    07_crba_rnea_validation.png
    08_performance_waterfall.png
    09_walking_dashboard.png          (walking scenario only)
    10_controller_comparison.png      (walking scenario only)

output/gifs/
    thor_standing_enhanced.gif
    thor_walking_enhanced.gif         (walking scenario only)
"""

import argparse
import sys
import time
import timeit
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config
from thor.dynamics.contact_implicit import run_contact_implicit_simulation
from thor.visualization.publication_plots import (
    plot_standing_analysis,
    plot_walking_dashboard,
    plot_crba_rnea_validation,
    plot_mass_matrix_structure,
    plot_energy_conservation,
    plot_contact_force_analysis,
    plot_lcp_solver_analysis,
    plot_biomechanical_reference,
    plot_controller_comparison,
    plot_performance_waterfall,
)
from thor.visualization.walking_animation import (
    generate_standing_gif,
    generate_walking_gif,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate all THOR publication plots and GIFs.')
    p.add_argument('--scenario', default='standing',
                   choices=['standing', 'walking', 'both'],
                   help='Which simulation to run')
    p.add_argument('--t-final', type=float, default=None,
                   help='Simulation duration [s]')
    p.add_argument('--dt', type=float, default=0.002,
                   help='Integration time step [s]')
    p.add_argument('--skip-gif', action='store_true',
                   help='Skip GIF generation')
    p.add_argument('--output', type=str, default='output',
                   help='Base output directory')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _bench_component(fn, *args, n_repeat: int = 3) -> float:
    """Return mean wall-clock time [s] of fn(*args) over n_repeat calls."""
    timer = timeit.Timer(lambda: fn(*args))
    total = timer.timeit(number=n_repeat)
    return total / n_repeat


def _collect_timing(model: RobotModel, q) -> dict:
    """Benchmark major per-step components."""
    from thor.dynamics.crba import crba
    from thor.dynamics.rnea import rnea as rnea_fn, bias_forces
    from thor.model.kinematics import forward_kinematics
    import numpy as np
    from scipy.linalg import cho_factor

    v  = np.zeros(model.n_dof)
    a  = np.zeros(model.n_dof)
    M  = crba(model, q)
    Mj = M[6:, 6:]

    timing = {
        'CRBA':      _bench_component(crba, model, q),
        'RNEA':      _bench_component(rnea_fn, model, q, v, a),
        'FK':        _bench_component(forward_kinematics, q, model),
        'Cholesky':  _bench_component(cho_factor, Mj + 1e-10 * np.eye(Mj.shape[0])),
        'bias_forces': _bench_component(bias_forces, model, q, v),
    }
    total_known = sum(timing.values())
    timing['Other'] = max(0.0, total_known * 0.15)   # estimated overhead
    return timing


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def _run_standing(model: RobotModel, q0, t_final: float, dt: float) -> dict:
    from thor.control.contact_implicit_mpc import ContactImplicitMPC
    mpc = ContactImplicitMPC(model, Q_q=500.0, Q_v=50.0)
    mpc.set_reference(q0)
    print(f'  Running Standing: t_final={t_final}s, dt={dt}s ...')
    t0 = time.perf_counter()
    result = run_contact_implicit_simulation(
        model, q0, mpc.compute, t_final=t_final, dt=dt, walking_speed=0.0)
    elapsed = time.perf_counter() - t0
    print(f'  Done in {elapsed:.2f}s ({len(result["time"])} steps)')
    return result


def _run_walking(model: RobotModel, q0, t_final, dt: float) -> dict:
    from thor.control.walking_controller import WalkingController
    walker = WalkingController(model, q0, n_steps=6)
    t_final = t_final or walker.total_duration
    print(f'  Running Walking: t_final={t_final:.1f}s, dt={dt}s ...')
    t0 = time.perf_counter()
    result = run_contact_implicit_simulation(
        model, q0, walker.compute, t_final=t_final, dt=dt)
    elapsed = time.perf_counter() - t0
    print(f'  Done in {elapsed:.2f}s ({len(result["time"])} steps)')
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    plots_dir = str(Path(args.output) / 'plots')
    gifs_dir  = str(Path(args.output) / 'gifs')
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(gifs_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 62)
    print('  THOR 34-DOF -- Publication Plot Generator')
    print('=' * 62)

    # --- Build model ---
    print('\n[1/N] Building robot model ...')
    model = RobotModel()
    q0    = default_standing_config(model)
    print(f'      {model.n_bodies} bodies | {model.n_dof} DOF | '
          f'{model.total_mass:.1f} kg')

    saved_paths = []

    # --- Physics-independent plots (always generated) ---
    print('\n[2/N] Contact force model plot ...')
    p = plot_contact_force_analysis(f'{plots_dir}/04_contact_force_model.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    print('\n[3/N] LCP solver analysis plot ...')
    p = plot_lcp_solver_analysis(f'{plots_dir}/05_lcp_solver_analysis.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    print('\n[4/N] Biomechanical reference plot ...')
    p = plot_biomechanical_reference(f'{plots_dir}/06_biomechanical_reference.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    print('\n[5/N] CRBA / RNEA validation plot ...')
    p = plot_crba_rnea_validation(model, f'{plots_dir}/07_crba_rnea_validation.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    print('\n[6/N] Mass matrix structure plot ...')
    p = plot_mass_matrix_structure(model, q0,
                                   f'{plots_dir}/02_mass_matrix_structure.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    # --- Performance waterfall (uses micro-benchmarks) ---
    print('\n[7/N] Performance waterfall (benchmarking components) ...')
    timing = _collect_timing(model, q0)
    p = plot_performance_waterfall(timing,
                                   f'{plots_dir}/08_performance_waterfall.png')
    saved_paths.append(p)
    print(f'      -> {p}')

    # --- Standing simulation ---
    run_standing_flag = args.scenario in ('standing', 'both')
    run_walking_flag  = args.scenario in ('walking', 'both')

    standing_result = None
    walking_result  = None

    if run_standing_flag:
        t_fin_s = args.t_final or 5.0
        print(f'\n[8/N] Standing simulation (t={t_fin_s}s) ...')
        standing_result = _run_standing(model, q0, t_fin_s, args.dt)

        print('      Generating standing analysis plot ...')
        p = plot_standing_analysis(
            standing_result, f'{plots_dir}/01_standing_analysis.png')
        saved_paths.append(p)
        print(f'      -> {p}')

        print('      Generating energy conservation plot ...')
        p = plot_energy_conservation(
            standing_result, f'{plots_dir}/03_energy_conservation.png')
        saved_paths.append(p)
        print(f'      -> {p}')

        if not args.skip_gif:
            print('      Generating standing GIF ...')
            g = generate_standing_gif(
                standing_result, model,
                save_path=f'{gifs_dir}/thor_standing_enhanced.gif',
                fps=20, dpi=100)
            saved_paths.append(g)
            print(f'      -> {g}')

    if run_walking_flag:
        t_fin_w = args.t_final
        print(f'\n[9/N] Walking simulation ...')
        walking_result = _run_walking(model, q0, t_fin_w, args.dt)

        print('      Generating walking dashboard ...')
        p = plot_walking_dashboard(
            walking_result, f'{plots_dir}/09_walking_dashboard.png')
        saved_paths.append(p)
        print(f'      -> {p}')

        if not args.skip_gif:
            print('      Generating walking GIF ...')
            g = generate_walking_gif(
                walking_result, model,
                save_path=f'{gifs_dir}/thor_walking_enhanced.gif',
                fps=20, dpi=100)
            saved_paths.append(g)
            print(f'      -> {g}')

    # --- Controller comparison (if both scenarios ran) ---
    if standing_result is not None and walking_result is not None:
        print('\n[10/N] Controller comparison plot ...')
        results_dict = {
            'Standing (CI-MPC)': standing_result,
            'Walking (CTC)':     walking_result,
        }
        p = plot_controller_comparison(
            results_dict, f'{plots_dir}/10_controller_comparison.png')
        saved_paths.append(p)
        print(f'      -> {p}')

    # --- Summary ---
    print('\n' + '=' * 62)
    print('  All done! Generated files:')
    for fp in saved_paths:
        size_kb = Path(fp).stat().st_size // 1024 if Path(fp).exists() else 0
        print(f'    {fp}  [{size_kb} KB]')
    print('=' * 62)


if __name__ == '__main__':
    main()
