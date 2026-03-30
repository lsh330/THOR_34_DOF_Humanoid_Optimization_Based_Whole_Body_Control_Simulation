"""
Physical and mathematical constants for the THOR simulation.

All values in SI units.
"""

import numpy as np

# Gravitational acceleration [m/s^2]
GRAVITY: float = 9.81
GRAVITY_VEC: np.ndarray = np.array([0.0, 0.0, -GRAVITY])

# Numerical tolerances
EPS: float = 1e-12
SINGULAR_TOL: float = 1e-8
CONTACT_TOL: float = 1e-4        # Contact detection threshold [m]
FRICTION_CONE_FACES: int = 8      # Linearized friction cone facets

# Default friction coefficient
MU_DEFAULT: float = 0.7

# Simulation defaults
DEFAULT_DT: float = 0.001         # 1 kHz dynamics
MPC_DT: float = 0.02              # 50 Hz MPC
CONTROL_DT: float = 0.001         # 1 kHz WBC

# THOR robot constants
THOR_NUM_JOINTS: int = 34
THOR_NUM_ACTUATED: int = 28       # Exclude 6 floating-base DOFs
THOR_TOTAL_DOF: int = 40          # 6 floating-base + 34 joints
THOR_MASS: float = 65.0           # Total mass [kg]
THOR_HEIGHT: float = 1.78         # Standing height [m]
