"""
Linear Complementarity Problem (LCP) solver.

Solves: find z >= 0 such that w = M*z + q >= 0 and z^T*w = 0.

Provides two methods:
    1. Fischer-Burmeister + Newton (smooth, differentiable, Numba-friendly)
    2. Interior-Point (robust, guaranteed convergence for P-matrices)

The LCP is the core of contact-implicit dynamics:
    - Normal contact: 0 <= lambda_n  perp  phi/h + J_n*v >= 0
    - Friction: encoded via polyhedral approximation

Reference:
    Cottle, R.W., Pang, J.-S. & Stone, R.E. (1992). "The Linear
    Complementarity Problem." Academic Press.
    Fischer, A. (1992). "A special Newton-type optimization method."
    Optimization, 24(3-4), 269-284.
"""

import math

import numpy as np
from numpy.typing import NDArray


def fischer_burmeister(a: float, b: float, eps: float = 1e-10) -> float:
    """Fischer-Burmeister NCP function.

    phi(a,b) = a + b - sqrt(a^2 + b^2 + 2*eps^2)

    phi(a,b) = 0 iff a >= 0, b >= 0, a*b ≈ 0 (exact as eps→0).
    Smooth and differentiable everywhere (unlike min(a,b)).
    """
    return a + b - math.sqrt(a * a + b * b + 2.0 * eps * eps)


def solve_lcp_fb_newton(
    M: NDArray,
    q: NDArray,
    z0: NDArray | None = None,
    eps: float = 1e-4,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[NDArray, int, float]:
    """Solve LCP via Fischer-Burmeister + Newton's method.

    Reformulates the LCP as a system of smooth equations:
        F_i(z) = phi_FB(z_i, w_i) = 0, where w = M*z + q

    Solves F(z) = 0 using damped Newton iteration with backtracking.

    Args:
        M: (n, n) LCP matrix (must be a P-matrix for unique solution)
        q: (n,) LCP vector
        z0: Initial guess (default: ones*0.1)
        eps: FB smoothing parameter
        tol: Convergence tolerance on ||F||
        max_iter: Maximum Newton iterations

    Returns:
        z: (n,) solution
        iters: Number of iterations used
        residual: Final ||F||
    """
    n = len(q)
    z = z0.copy() if z0 is not None else np.ones(n) * 0.1

    for iteration in range(max_iter):
        w = M @ z + q

        # Evaluate FB residual
        F = np.empty(n)
        for i in range(n):
            F[i] = fischer_burmeister(z[i], w[i], eps)

        res = np.linalg.norm(F)
        if res < tol:
            return z, iteration, res

        # Jacobian: dF_i/dz_j
        J = np.empty((n, n))
        for i in range(n):
            denom = math.sqrt(z[i]**2 + w[i]**2 + 2.0 * eps**2)
            da = 1.0 - z[i] / denom   # d(phi)/d(a) where a=z_i
            db = 1.0 - w[i] / denom   # d(phi)/d(b) where b=w_i
            for j in range(n):
                J[i, j] = db * M[i, j]
            J[i, i] += da

        # Newton step with regularization
        try:
            dz = np.linalg.solve(J + 1e-12 * np.eye(n), -F)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]

        # Backtracking line search
        alpha = 1.0
        for _ in range(15):
            z_new = z + alpha * dz
            w_new = M @ z_new + q
            F_new = np.empty(n)
            for i in range(n):
                F_new[i] = fischer_burmeister(z_new[i], w_new[i], eps)
            if np.linalg.norm(F_new) < res:
                break
            alpha *= 0.5

        z = z + alpha * dz

    w = M @ z + q
    F = np.array([fischer_burmeister(z[i], w[i], eps) for i in range(n)])
    return z, max_iter, np.linalg.norm(F)


def solve_lcp_interior_point(
    M: NDArray,
    q: NDArray,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[NDArray, int, float]:
    """Solve LCP via interior-point method.

    Relaxes complementarity z*w = 0 → z*w = kappa,
    drives kappa → 0 via centering parameter sigma.

    More robust than Newton-FB for ill-conditioned problems.

    Args:
        M: (n, n) LCP matrix
        q: (n,) LCP vector

    Returns:
        z: (n,) solution
        iters: Iterations used
        residual: Final duality gap
    """
    n = len(q)
    z = np.ones(n)
    w = M @ z + q
    # Ensure initial feasibility
    w = np.maximum(w, 0.01)

    sigma = 0.1  # Centering parameter

    for iteration in range(max_iter):
        mu = np.dot(z, w) / n  # Duality measure
        if mu < tol:
            return z, iteration, mu

        kappa = sigma * mu

        # Residuals
        r_w = w - M @ z - q
        r_c = z * w - kappa

        # Solve Newton system via Schur complement:
        # (W + Z*M)*dz = -r_c + Z*r_w
        W = np.diag(w)
        Z = np.diag(z)
        lhs = W + Z @ M + 1e-10 * np.eye(n)
        rhs = -r_c + Z @ r_w

        try:
            dz = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        dw = M @ dz - r_w

        # Step size (maintain z > 0, w > 0)
        alpha = 0.99
        for i in range(n):
            if dz[i] < -1e-14:
                alpha = min(alpha, -0.95 * z[i] / dz[i])
            if dw[i] < -1e-14:
                alpha = min(alpha, -0.95 * w[i] / dw[i])

        z += alpha * dz
        w += alpha * dw
        z = np.maximum(z, 1e-14)
        w = np.maximum(w, 1e-14)

    return z, max_iter, np.dot(z, w) / n
