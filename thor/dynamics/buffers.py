"""Pre-allocated working memory for dynamics computations.

Eliminates per-step heap allocation in the simulation inner loop.
All buffers are created once at initialization and reused in-place.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve


class DynamicsBuffers:
    """Pre-allocated buffers for Featherstone algorithm computations.

    Args:
        n_bodies: Number of rigid bodies (35 for THOR).
        n_dof: Total degrees of freedom (40 for THOR).
    """
    __slots__ = (
        'n_bodies', 'n_dof', 'n_joints',
        # RNEA buffers
        'vel', 'acc', 'f_body', 'X_up',
        # CRBA buffers
        'I_c', 'F_crba',
        # FK buffers
        'X_world', 'X_parent',
        # Mass matrix and bias
        'M', 'bias', 'tau', 'ddq',
        # Contact-implicit step
        'v_new', 'q_new',
        # Regularization (pre-allocated identity)
        '_reg_jj',
        # Cholesky cache
        '_cho_cache', '_M_jj_ref', '_cho_valid',
    )

    def __init__(self, n_bodies: int, n_dof: int):
        self.n_bodies = n_bodies
        self.n_dof = n_dof
        self.n_joints = n_dof - 6  # Joint DOFs (exclude floating base)

        # RNEA: velocity, acceleration, force per body
        self.vel = np.zeros((n_bodies, 6))
        self.acc = np.zeros((n_bodies, 6))
        self.f_body = np.zeros((n_bodies, 6))
        self.X_up = np.zeros((n_bodies, 6, 6))

        # CRBA: composite inertia, force propagation
        self.I_c = np.zeros((n_bodies, 6, 6))
        self.F_crba = np.zeros(6)  # Temporary for CRBA inner loop

        # FK: spatial transforms
        self.X_world = np.zeros((n_bodies, 6, 6))
        self.X_parent = np.zeros((n_bodies, 6, 6))

        # Mass matrix (full n_dof x n_dof)
        self.M = np.zeros((n_dof, n_dof))
        self.bias = np.zeros(n_dof)
        self.tau = np.zeros(n_dof)
        self.ddq = np.zeros(n_dof)

        # Integration buffers
        self.v_new = np.zeros(n_dof)
        self.q_new = np.zeros(n_dof + 1)  # q has dim 41 (3+4+34)

        # Pre-allocated regularization matrix: 1e-10 * I_{n_joints}
        self._reg_jj = 1e-10 * np.eye(self.n_joints)

        # Cholesky cache
        self._cho_cache = None
        self._M_jj_ref = np.zeros((self.n_joints, self.n_joints))
        self._cho_valid = False

    def reset_rnea(self) -> None:
        """Zero out RNEA working buffers."""
        self.vel[:] = 0.0
        self.acc[:] = 0.0
        self.f_body[:] = 0.0

    def reset_crba(self) -> None:
        """Zero out CRBA working buffers."""
        self.I_c[:] = 0.0
        self.M[:] = 0.0

    def get_cho_factor(self, M_jj: NDArray, threshold: float = 1e-3):
        """Get Cholesky factorization with conditional cache.

        Re-factorizes only when M_jj changes beyond threshold.

        Args:
            M_jj: Joint-space mass matrix (n_joints x n_joints).
            threshold: Relative Frobenius norm threshold for re-factorization.

        Returns:
            Cholesky factorization tuple (from scipy.linalg.cho_factor).
        """
        if self._cho_valid:
            ref_norm = np.linalg.norm(self._M_jj_ref)
            if ref_norm > 1e-12:
                rel_change = np.linalg.norm(M_jj - self._M_jj_ref) / ref_norm
                if rel_change < threshold:
                    return self._cho_cache

        # Re-factorize
        self._cho_cache = cho_factor(M_jj + self._reg_jj)
        self._M_jj_ref[:] = M_jj
        self._cho_valid = True
        return self._cho_cache

    def invalidate_cho_cache(self) -> None:
        """Invalidate Cholesky cache (call on configuration change)."""
        self._cho_valid = False
