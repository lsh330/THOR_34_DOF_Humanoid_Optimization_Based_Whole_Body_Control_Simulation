"""
Unified dynamics API (Facade pattern).

Centralizes JIT/Python dispatch, buffer management, and Cholesky caching
for all dynamics computations. This is the recommended high-level interface
for accessing THOR's dynamics engine.

Usage:
    model = RobotModel()
    dyn = DynamicsFacade(model)

    M = dyn.mass_matrix(q)
    h = dyn.bias_forces(q, v)
    X_world = dyn.forward_kinematics(q)
    com = dyn.com_position(q)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from ..model.robot_model import RobotModel
from .buffers import DynamicsBuffers


class DynamicsFacade:
    """Unified API for all dynamics computations.

    Manages:
    - JIT vs Python dispatch (transparent fallback)
    - Pre-allocated working buffers
    - Cholesky factorization caching
    - FK result caching within a timestep

    Args:
        model: THOR robot model.
        use_jit: Enable Numba JIT acceleration (default True).
    """

    __slots__ = (
        '_model', '_md', '_buffers', '_use_jit',
        '_fk_cache_q', '_fk_cache_X',
    )

    def __init__(self, model: RobotModel, use_jit: bool = True):
        self._model = model
        self._md = model.model_data
        self._buffers = DynamicsBuffers(model.n_bodies, model.n_dof)
        self._use_jit = use_jit
        # FK cache: avoid redundant FK within same timestep
        self._fk_cache_q = None
        self._fk_cache_X = None

    @property
    def model(self) -> RobotModel:
        return self._model

    @property
    def buffers(self) -> DynamicsBuffers:
        return self._buffers

    def mass_matrix(self, q: NDArray) -> NDArray:
        """Compute joint-space inertia matrix M(q).

        Dispatches to JIT CRBA when available.

        Args:
            q: Configuration (dim 41).

        Returns:
            M: (n_dof, n_dof) symmetric positive-definite mass matrix.
        """
        from .crba import crba
        return crba(self._model, q)

    def bias_forces(self, q: NDArray, v: NDArray) -> NDArray:
        """Compute bias forces h(q, v) = C(q,v)*v + g(q).

        Dispatches to JIT RNEA when available.

        Args:
            q: Configuration (dim 41).
            v: Velocity (dim 40).

        Returns:
            h: (n_dof,) bias force vector.
        """
        from .rnea import bias_forces as _bias
        return _bias(self._model, q, v)

    def gravity_forces(self, q: NDArray) -> NDArray:
        """Compute gravity forces g(q) = RNEA(q, 0, 0).

        Args:
            q: Configuration (dim 41).

        Returns:
            g: (n_dof,) gravity force vector.
        """
        from .rnea import gravity_forces as _grav
        return _grav(self._model, q)

    def forward_kinematics(self, q: NDArray) -> NDArray:
        """Compute FK for all bodies (JIT path, returns 3D array).

        Results are cached — calling twice with the same q is free.

        Args:
            q: Configuration (dim 41).

        Returns:
            X_world: (n_bodies, 6, 6) spatial transforms.
        """
        # Check cache
        if (self._fk_cache_q is not None
                and np.array_equal(q, self._fk_cache_q)):
            return self._fk_cache_X

        if self._use_jit:
            try:
                from ..model.kinematics_jit import forward_kinematics_jit
                X = self._buffers.X_world
                forward_kinematics_jit(
                    self._md.n_bodies, self._md.parent,
                    self._md.joint_types, self._md.joint_offsets,
                    self._md.joint_rotations, q, X,
                )
                self._fk_cache_q = q.copy()
                self._fk_cache_X = X
                return X
            except Exception:
                pass

        # Python fallback
        from ..model.kinematics import forward_kinematics
        X_list, _ = forward_kinematics(q, self._model)
        X = np.array(X_list)
        self._fk_cache_q = q.copy()
        self._fk_cache_X = X
        return X

    def body_position(self, X_world_i: NDArray) -> NDArray:
        """Extract 3D position from spatial transform.

        Args:
            X_world_i: (6, 6) spatial transform for one body.

        Returns:
            p: (3,) world position [m].
        """
        if self._use_jit:
            try:
                from ..model.kinematics_jit import body_position_jit
                return body_position_jit(X_world_i)
            except Exception:
                pass
        from ..model.kinematics import body_position
        return body_position(X_world_i)

    def com_position(self, q: NDArray) -> NDArray:
        """Compute center of mass position.

        Uses cached FK when available.

        Args:
            q: Configuration (dim 41).

        Returns:
            com: (3,) CoM position in world frame [m].
        """
        from ..model.kinematics import com_position as _com
        return _com(q, self._model)

    def solve_joints(self, M_jj: NDArray, rhs: NDArray,
                     use_cache: bool = True) -> NDArray:
        """Solve M_jj * ddq = rhs with optional Cholesky caching.

        Args:
            M_jj: (n_joints, n_joints) joint mass matrix.
            rhs: (n_joints,) right-hand side.
            use_cache: Use Cholesky cache (default True).

        Returns:
            ddq: (n_joints,) joint accelerations.
        """
        try:
            if use_cache:
                cho = self._buffers.get_cho_factor(M_jj)
            else:
                cho = cho_factor(M_jj + self._buffers._reg_jj)
            return cho_solve(cho, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(len(rhs))

    def invalidate_caches(self) -> None:
        """Invalidate all caches. Call when configuration changes significantly."""
        self._fk_cache_q = None
        self._fk_cache_X = None
        self._buffers.invalidate_cho_cache()
