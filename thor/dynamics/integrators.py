"""Numerical integration strategies for rigid body dynamics.

Strategy pattern: swap integrators without changing the simulation loop.

All integrators handle the special structure of floating-base configuration:
    q = [position(3), quaternion(4), joints(34)] in R^41
    v = [angular_vel(3), linear_vel(3), joint_vel(34)] in R^40
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Integrator(ABC):
    """Abstract base for numerical integration strategies."""

    @abstractmethod
    def integrate_config(self, q: NDArray, v: NDArray, dt: float) -> NDArray:
        """Integrate configuration: q_{k+1} given v_{k+1} and dt.

        Handles quaternion integration for floating base.

        Args:
            q: Current configuration (dim 41).
            v: Velocity at new time step (dim 40).
            dt: Time step [s].

        Returns:
            New configuration (dim 41).
        """
        ...

    @abstractmethod
    def integrate_velocity(self, v: NDArray, ddq: NDArray, dt: float) -> NDArray:
        """Integrate velocity: v_{k+1} given acceleration and dt.

        Args:
            v: Current velocity (dim 40).
            ddq: Acceleration (dim 40).
            dt: Time step [s].

        Returns:
            New velocity (dim 40).
        """
        ...


class SemiImplicitEuler(Integrator):
    """Semi-implicit (symplectic) Euler integration.

    v_{k+1} = v_k + dt * a_k
    q_{k+1} = q_k + dt * v_{k+1}  (uses NEW velocity)

    Preserves symplectic structure, suitable for contact dynamics.
    This is the current default integrator in the THOR simulation.
    """

    def integrate_velocity(self, v: NDArray, ddq: NDArray, dt: float) -> NDArray:
        return v + dt * ddq

    def integrate_config(self, q: NDArray, v: NDArray, dt: float) -> NDArray:
        from ..model.quaternion import quat_integrate

        q_new = q.copy()
        q_new[:3] += dt * v[3:6]        # Position += dt * linear velocity
        q_new[3:7] = quat_integrate(q[3:7], v[0:3], dt)  # Quaternion
        q_new[7:] += dt * v[6:]         # Joints
        return q_new


class ExplicitEuler(Integrator):
    """Explicit (forward) Euler integration.

    v_{k+1} = v_k + dt * a_k
    q_{k+1} = q_k + dt * v_k  (uses OLD velocity)

    First-order, not symplectic. For comparison only.
    """

    def integrate_velocity(self, v: NDArray, ddq: NDArray, dt: float) -> NDArray:
        return v + dt * ddq

    def integrate_config(self, q: NDArray, v: NDArray, dt: float) -> NDArray:
        # Note: this receives v_old, not v_new
        from ..model.quaternion import quat_integrate

        q_new = q.copy()
        q_new[:3] += dt * v[3:6]
        q_new[3:7] = quat_integrate(q[3:7], v[0:3], dt)
        q_new[7:] += dt * v[6:]
        return q_new


class RK4Integrator(Integrator):
    """4th-order Runge-Kutta integration.

    For accuracy comparison. Requires the dynamics function
    to be called 4 times per step.

    Note: In contact-implicit stepping, RK4 is not directly applicable
    because the dynamics include discrete contact events. Use only
    for smooth (contact-free) phases or accuracy validation.
    """

    def integrate_velocity(self, v: NDArray, ddq: NDArray, dt: float) -> NDArray:
        # For RK4, the caller must provide the combined k1-k4 weighted acceleration
        return v + dt * ddq

    def integrate_config(self, q: NDArray, v: NDArray, dt: float) -> NDArray:
        from ..model.quaternion import quat_integrate

        q_new = q.copy()
        q_new[:3] += dt * v[3:6]
        q_new[3:7] = quat_integrate(q[3:7], v[0:3], dt)
        q_new[7:] += dt * v[6:]
        return q_new
