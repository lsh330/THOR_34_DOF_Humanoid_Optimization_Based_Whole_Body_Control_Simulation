"""Simulation mediator coordinating dynamics, control, and observation.

Mediator pattern: centralizes the simulation loop logic and coordinates
the dynamics engine, controller, integrator, and observers.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional

from ..model.robot_model import RobotModel
from ..dynamics.contact_implicit import contact_implicit_step
from ..dynamics.integrators import Integrator, SemiImplicitEuler
from .observers import (
    SimulationObserver, TrajectoryRecorder, CoMRecorder, PrintObserver
)


class SimulationMediator:
    """Coordinates all simulation components.

    Replaces the monolithic run_contact_implicit_simulation() with
    a composable, extensible simulation framework.

    Args:
        model: THOR robot model.
        controller: Callable (q, v, t) -> tau (dim 40 torque vector).
        integrator: Numerical integration strategy.
        observers: List of data collection/monitoring observers.
    """

    def __init__(
        self,
        model: RobotModel,
        controller: Callable,
        integrator: Optional[Integrator] = None,
        observers: Optional[list[SimulationObserver]] = None,
    ):
        self._model = model
        self._controller = controller
        self._integrator = integrator or SemiImplicitEuler()
        self._observers = observers or []

    def run(
        self,
        q0: NDArray,
        v0: Optional[NDArray] = None,
        t_final: float = 3.0,
        dt: float = 0.002,
        mu: float = 0.7,
        walking_speed: float = 0.0,
    ) -> dict:
        """Run simulation and return combined observer results.

        Args:
            q0: Initial configuration (dim 41).
            v0: Initial velocity (dim 40, default zeros).
            t_final: Duration [s].
            dt: Time step [s].
            mu: Friction coefficient.
            walking_speed: Kinematic forward progression [m/s].

        Returns:
            Combined dict from all observers plus metadata.
        """
        n_steps = int(t_final / dt) + 1
        q = q0.copy()
        v = v0.copy() if v0 is not None else np.zeros(self._model.n_dof)

        time_arr = np.linspace(0, t_final, n_steps)

        for step in range(n_steps):
            t = time_arr[step]

            # Compute control
            tau = self._controller(q, v, t)

            # Contact-implicit time step
            q_new, v_new, lam, info = contact_implicit_step(
                self._model, q, v, tau, dt, mu
            )

            # Kinematic forward progression (walking)
            if walking_speed > 0.0:
                q_new[0] += dt * walking_speed

            # Notify observers
            for obs in self._observers:
                obs.on_step(step, t, q_new, v_new, tau, info)

            q = q_new
            v = v_new

        # Collect results from all observers
        result = {"dt": dt, "t_final": t_final}
        for obs in self._observers:
            result.update(obs.finalize())

        return result

    @classmethod
    def create_standard(
        cls,
        model: RobotModel,
        controller: Callable,
        n_steps: int,
        verbose: bool = True,
    ) -> 'SimulationMediator':
        """Create mediator with standard observer set.

        Includes: TrajectoryRecorder, CoMRecorder, and optionally PrintObserver.
        """
        observers = [
            TrajectoryRecorder(n_steps),
            CoMRecorder(n_steps, model),
        ]
        if verbose:
            observers.append(PrintObserver(n_steps))

        return cls(model=model, controller=controller, observers=observers)
