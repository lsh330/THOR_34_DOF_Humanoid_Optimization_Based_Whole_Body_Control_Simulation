"""
THOR 34-DOF Humanoid Whole-Body Control Simulation — Entry Point

Usage:
    python main.py                          # Standing (CI-MPC)
    python main.py --scenario walking       # Walking (CTC)
    python main.py --scenario walking --save-gif
"""

import argparse
import time
import sys

from thor.model.robot_model import RobotModel
from thor.simulation.standing import default_standing_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="THOR 34-DOF Whole-Body Control Simulation")
    parser.add_argument("--scenario", type=str, default="standing",
                        choices=["standing", "walking"],
                        help="Simulation scenario")
    parser.add_argument("--t-final", type=float, default=None,
                        help="Duration [s] (default: 5.0 standing, 5.1 walking)")
    parser.add_argument("--dt", type=float, default=0.002,
                        help="Integration time step [s]")
    parser.add_argument("--save-gif", action="store_true", default=False,
                        help="Save walking/standing animation GIF")
    parser.add_argument("--output-dir", type=str, default="output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  THOR 34-DOF Humanoid Whole-Body Control Simulation")
    print("=" * 60)

    model = RobotModel()
    q0 = default_standing_config(model)
    print(f"  Robot: {model.n_bodies} bodies, {model.n_dof} DOF, "
          f"{model.total_mass:.1f} kg")

    from thor.dynamics.contact_implicit import run_contact_implicit_simulation

    if args.scenario == "standing":
        from thor.control.contact_implicit_mpc import ContactImplicitMPC
        t_final = args.t_final or 5.0
        mpc = ContactImplicitMPC(model, Q_q=500.0, Q_v=50.0)
        mpc.set_reference(q0)
        print(f"  Scenario: Standing (CI-MPC + LCP)")
        print(f"  Duration: {t_final}s, dt={args.dt}s")

        t0 = time.perf_counter()
        result = run_contact_implicit_simulation(
            model, q0, mpc.compute, t_final=t_final, dt=args.dt,
            walking_speed=0.0)
        elapsed = time.perf_counter() - t0

    elif args.scenario == "walking":
        from thor.control.walking_controller import WalkingController
        walker = WalkingController(model, q0, n_steps=6)
        t_final = args.t_final or walker.total_duration
        print(f"  Scenario: Walking (Computed Torque Control)")
        print(f"  Duration: {t_final:.1f}s, {6} steps, dt={args.dt}s")

        t0 = time.perf_counter()
        result = run_contact_implicit_simulation(
            model, q0, walker.compute, t_final=t_final, dt=args.dt)
        elapsed = time.perf_counter() - t0

    # Results
    com = result["com"]
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Simulation time: {elapsed:.2f}s ({len(result['time'])} steps)")
    print(f"  CoM z: {com[0,2]:.4f} → {com[-1,2]:.4f} m")
    if args.scenario == "walking":
        forward = result["q"][-1, 0] - result["q"][0, 0]
        print(f"  Forward: {forward:.3f} m")
    print(f"  Contact Fz (final): {result['contact_fz'][-1]:.0f} N")
    print(f"{'=' * 60}")

    # Save GIF
    if args.save_gif and "q" in result:
        print("\nGenerating GIF...")
        from thor.visualization.stick_figure import generate_standing_gif
        path = generate_standing_gif(result, model,
                                      output_dir=f"{args.output_dir}/gifs",
                                      fps=20)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
