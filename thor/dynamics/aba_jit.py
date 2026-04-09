"""JIT-compiled Articulated Body Algorithm (ABA) — skeleton.

ABA (Featherstone 2008, Algorithm 7.1) is currently not on the critical
performance path for THOR_Simulation, so this file provides only a stub.
A full @njit implementation can be added here when benchmarking reveals
ABA becomes a bottleneck (expected after CRBA and FK optimisations).

Three-pass structure (for reference):
    Pass 1 (base → tips)  : propagate velocities, compute bias terms
    Pass 2 (tips → base)  : accumulate articulated-body inertias
    Pass 3 (base → tips)  : back-solve for accelerations

When implemented the function signature will be:

    @njit(cache=True)
    def aba_jit(
        n_bodies, n_dof,
        parent, joint_types, joint_axes,
        spatial_inertias, joint_offsets, joint_rotations,
        q, v, tau,
        f_ext,          # (n_bodies, 6) or None
    ) -> np.ndarray:    # (n_dof,) accelerations

Notes:
    - np.linalg.solve is available in Numba (LAPACK binding).
    - Requires careful handling of the 6×6 base-body inertia inversion.
    - Coriolis/centrifugal bias uses spatial_cross_force (@njit compatible).
"""
