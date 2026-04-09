"""Simulation observers for data collection and monitoring.

Observer pattern: decouple data recording from the simulation loop.
Multiple observers can be attached to a single simulation run.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional


class SimulationObserver(ABC):
    """Abstract base for simulation event observers."""

    @abstractmethod
    def on_step(self, step: int, t: float, q: NDArray, v: NDArray,
                tau: NDArray, info: dict) -> None:
        """Called after each simulation step.

        Args:
            step: Step index (0-based).
            t: Current time [s].
            q: Configuration after step (dim 41).
            v: Velocity after step (dim 40).
            tau: Applied torque (dim 40).
            info: Contact/dynamics info dict.
        """
        ...

    @abstractmethod
    def finalize(self) -> dict:
        """Called after simulation completes. Returns collected data."""
        ...


class TrajectoryRecorder(SimulationObserver):
    """Records full simulation trajectory into pre-allocated arrays.

    This is the primary data collector, replacing the inline recording
    in run_contact_implicit_simulation().
    """

    def __init__(self, n_steps: int, n_q: int = 41, n_v: int = 40):
        self.n_steps = n_steps
        self.time = np.empty(n_steps)
        self.q_traj = np.empty((n_steps, n_q))
        self.v_traj = np.empty((n_steps, n_v))
        self.tau_traj = np.empty((n_steps, n_v))
        self.fz_traj = np.empty(n_steps)
        self.contact_traj = np.empty(n_steps, dtype=np.int32)
        self._count = 0

    def on_step(self, step: int, t: float, q: NDArray, v: NDArray,
                tau: NDArray, info: dict) -> None:
        if step < self.n_steps:
            self.time[step] = t
            self.q_traj[step] = q
            self.v_traj[step] = v
            self.tau_traj[step] = tau
            self.fz_traj[step] = info.get("total_fz", 0.0)
            self.contact_traj[step] = info.get("n_contacts", 0)
            self._count = step + 1

    def finalize(self) -> dict:
        n = self._count
        return {
            "time": self.time[:n],
            "q": self.q_traj[:n],
            "v": self.v_traj[:n],
            "tau": self.tau_traj[:n],
            "contact_fz": self.fz_traj[:n],
            "n_contacts": self.contact_traj[:n],
        }


class CoMRecorder(SimulationObserver):
    """Records Center of Mass trajectory."""

    def __init__(self, n_steps: int, model):
        self.n_steps = n_steps
        self.com_traj = np.empty((n_steps, 3))
        self.base_traj = np.empty((n_steps, 3))
        self._model = model
        self._count = 0

    def on_step(self, step: int, t: float, q: NDArray, v: NDArray,
                tau: NDArray, info: dict) -> None:
        if step < self.n_steps:
            from ..model.kinematics import com_position
            self.com_traj[step] = com_position(q, self._model)
            self.base_traj[step] = q[:3]
            self._count = step + 1

    def finalize(self) -> dict:
        n = self._count
        return {
            "com": self.com_traj[:n],
            "base": self.base_traj[:n],
        }


class EnergyMonitor(SimulationObserver):
    """Monitors kinetic and potential energy for conservation checks."""

    def __init__(self, n_steps: int, model):
        self.n_steps = n_steps
        self.ke_traj = np.empty(n_steps)
        self.pe_traj = np.empty(n_steps)
        self.total_traj = np.empty(n_steps)
        self._model = model
        self._count = 0

    def on_step(self, step: int, t: float, q: NDArray, v: NDArray,
                tau: NDArray, info: dict) -> None:
        if step < self.n_steps:
            from ..dynamics.crba import crba
            from ..model.kinematics import com_position
            from ..core.constants import GRAVITY

            M = crba(self._model, q)
            ke = 0.5 * v @ M @ v
            com_z = com_position(q, self._model)[2]
            pe = self._model.total_mass * GRAVITY * com_z

            self.ke_traj[step] = ke
            self.pe_traj[step] = pe
            self.total_traj[step] = ke + pe
            self._count = step + 1

    def finalize(self) -> dict:
        n = self._count
        return {
            "kinetic_energy": self.ke_traj[:n],
            "potential_energy": self.pe_traj[:n],
            "total_energy": self.total_traj[:n],
        }


class PrintObserver(SimulationObserver):
    """Prints periodic status updates to console."""

    def __init__(self, n_steps: int, print_interval: int = 0):
        """
        Args:
            n_steps: Total number of steps.
            print_interval: Print every N steps. 0 = auto (10 prints total).
        """
        self.n_steps = n_steps
        self.print_interval = print_interval if print_interval > 0 else max(1, n_steps // 10)
        self._last_com_z = 0.0

    def on_step(self, step: int, t: float, q: NDArray, v: NDArray,
                tau: NDArray, info: dict) -> None:
        if step % self.print_interval == 0:
            fz = info.get("total_fz", 0.0)
            nc = info.get("n_contacts", 0)
            com_z = q[2]  # Approximate CoM height from base height
            print(f"    t={t:5.2f}s: base_z={com_z:.4f}m, Fz={fz:.0f}N, contacts={nc}")

    def finalize(self) -> dict:
        return {}
