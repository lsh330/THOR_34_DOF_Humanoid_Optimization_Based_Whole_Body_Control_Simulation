"""
THOR 34-DOF Humanoid Whole-Body Control Simulation - Entry Point

Contact-Implicit MPC + Hierarchical QP Whole-Body Control
for the THOR humanoid robot (Virginia Tech RoMeLa).

Usage:
    python main.py                       # Default: static standing
    python main.py --scenario walking    # Forward walking
    python main.py --save-gif            # With animation
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="THOR 34-DOF Whole-Body Control Simulation")
    parser.add_argument("--scenario", type=str, default="standing",
                        choices=["standing", "stepping", "walking", "push_recovery"],
                        help="Simulation scenario")
    parser.add_argument("--t-final", type=float, default=5.0,
                        help="Simulation duration [s]")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Integration time step [s]")
    parser.add_argument("--save-plots", action="store_true", default=True)
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("THOR 34-DOF Humanoid Whole-Body Control Simulation")
    print(f"  Scenario: {args.scenario}")
    print(f"  Duration: {args.t_final}s, dt={args.dt}s")
    print("  (Implementation in progress)")


if __name__ == "__main__":
    main()
