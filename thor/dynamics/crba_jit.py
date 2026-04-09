"""JIT-compiled Composite Rigid Body Algorithm.

Monolithic @njit function for computing the mass matrix M(q).
Avoids Python object overhead by operating on flat NumPy arrays
from ModelData (see thor.model.model_data).

Joint type encoding (must match JointType enum):
    0 = FIXED
    1 = REVOLUTE_X  (axis 0)
    2 = REVOLUTE_Y  (axis 1)
    3 = REVOLUTE_Z  (axis 2)
    4 = FLOATING

Spatial transform convention (Featherstone 2008, §2.8):
    X = [[R,   0   ],
         [-R*skew(p), R]]

Inertia propagation:
    I_c[parent] += X_up[i]^T @ I_c[i] @ X_up[i]
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def crba_jit(
    n_bodies: int,
    n_dof: int,
    parent: np.ndarray,          # (n_bodies,)  int32
    joint_types: np.ndarray,     # (n_bodies,)  int32
    joint_axes: np.ndarray,      # (n_bodies,)  int32
    spatial_inertias: np.ndarray,# (n_bodies, 6, 6) float64
    joint_offsets: np.ndarray,   # (n_bodies, 3) float64
    joint_rotations: np.ndarray, # (n_bodies, 3, 3) float64
    q: np.ndarray,               # (n_q,) float64  [p(3), quat(4), joints(N-1)]
) -> np.ndarray:
    """JIT CRBA: compute mass matrix M(q).

    Args:
        n_bodies: Number of rigid bodies (35 for THOR).
        n_dof: Total DOF (40 for THOR: 6 floating + 34 revolute).
        parent: Parent body index array; -1 for root.
        joint_types: Joint type codes per body.
        joint_axes: Primary axis index (0/1/2) for revolute joints.
        spatial_inertias: Spatial inertia matrices in body frames [kg, kg*m^2].
        joint_offsets: Translation from parent origin to child joint origin [m].
        joint_rotations: Fixed rotation from parent frame to child joint frame.
        q: Configuration [p_base(3), quat_base(4), q_joints(N-1)].

    Returns:
        M: (n_dof, n_dof) symmetric positive-definite joint-space mass matrix.
    """
    # ------------------------------------------------------------------ #
    # Parse configuration                                                  #
    # ------------------------------------------------------------------ #
    quat = q[3:7]        # [w, x, y, z]
    q_joints = q[7:]     # revolute joint angles [rad]

    # ------------------------------------------------------------------ #
    # Allocate working arrays                                              #
    # ------------------------------------------------------------------ #
    X_up = np.zeros((n_bodies, 6, 6))   # parent-to-body spatial transforms
    I_c  = np.zeros((n_bodies, 6, 6))   # composite inertias (mutated in Pass 1)
    M    = np.zeros((n_dof, n_dof))

    # Copy spatial inertias — composite accumulation mutates these in-place
    for i in range(n_bodies):
        for r in range(6):
            for c in range(6):
                I_c[i, r, c] = spatial_inertias[i, r, c]

    # ------------------------------------------------------------------ #
    # Compute spatial transforms X_up[i]                                  #
    # ------------------------------------------------------------------ #

    # --- Body 0: floating base ---
    # Quaternion [w, x, y, z] → rotation matrix R_base
    w = quat[0]; x = quat[1]; y = quat[2]; z = quat[3]

    R_base = np.zeros((3, 3))
    R_base[0, 0] = 1.0 - 2.0 * (y*y + z*z)
    R_base[0, 1] = 2.0 * (x*y - w*z)
    R_base[0, 2] = 2.0 * (x*z + w*y)
    R_base[1, 0] = 2.0 * (x*y + w*z)
    R_base[1, 1] = 1.0 - 2.0 * (x*x + z*z)
    R_base[1, 2] = 2.0 * (y*z - w*x)
    R_base[2, 0] = 2.0 * (x*z - w*y)
    R_base[2, 1] = 2.0 * (y*z + w*x)
    R_base[2, 2] = 1.0 - 2.0 * (x*x + y*y)

    # Position of base [m]
    px = q[0]; py = q[1]; pz = q[2]

    # skew(p) = [[0, -pz, py], [pz, 0, -px], [-py, px, 0]]
    # X_up[0, 3:, :3] = -R_base @ skew(p)
    # Pre-multiply analytically to avoid allocating skew matrix:
    #   -R @ skew(p) → row i of (-R @ skew(p)) = [-R_row · col_j_of_skew]
    #   col0 of skew(p) = [0, pz, -py]
    #   col1 of skew(p) = [-pz, 0, px]
    #   col2 of skew(p) = [py, -px, 0]
    X_up[0, :3, :3] = R_base
    X_up[0, 3:, 3:] = R_base
    for r in range(3):
        X_up[0, 3+r, 0] = -(R_base[r, 1]*pz - R_base[r, 2]*py)
        X_up[0, 3+r, 1] = -(R_base[r, 2]*px - R_base[r, 0]*pz)
        X_up[0, 3+r, 2] = -(R_base[r, 0]*py - R_base[r, 1]*px)

    # --- Bodies 1 .. n_bodies-1: revolute joints ---
    for i in range(1, n_bodies):
        qi = 0.0
        idx = i - 1
        if idx < len(q_joints):
            qi = q_joints[idx]

        jtype = joint_types[i]
        c = math.cos(qi)
        s = math.sin(qi)

        # Build elementary rotation R_joint
        R_joint = np.zeros((3, 3))
        if jtype == 1:          # REVOLUTE_X: rot around X
            R_joint[0, 0] = 1.0
            R_joint[1, 1] =  c;  R_joint[1, 2] = -s
            R_joint[2, 1] =  s;  R_joint[2, 2] =  c
        elif jtype == 2:        # REVOLUTE_Y: rot around Y
            R_joint[0, 0] =  c;  R_joint[0, 2] =  s
            R_joint[1, 1] = 1.0
            R_joint[2, 0] = -s;  R_joint[2, 2] =  c
        elif jtype == 3:        # REVOLUTE_Z: rot around Z
            R_joint[0, 0] =  c;  R_joint[0, 1] = -s
            R_joint[1, 0] =  s;  R_joint[1, 1] =  c
            R_joint[2, 2] = 1.0
        else:                   # FIXED or other
            R_joint[0, 0] = 1.0
            R_joint[1, 1] = 1.0
            R_joint[2, 2] = 1.0

        # R_total = joint_rotations[i] @ R_joint
        R_total = np.zeros((3, 3))
        for r in range(3):
            for cc in range(3):
                for k in range(3):
                    R_total[r, cc] += joint_rotations[i, r, k] * R_joint[k, cc]

        # Offset vector
        ox = joint_offsets[i, 0]
        oy = joint_offsets[i, 1]
        oz = joint_offsets[i, 2]

        # X_up[i] = spatial_transform(R_total, offset)
        X_up[i, :3, :3] = R_total
        X_up[i, 3:, 3:] = R_total
        for r in range(3):
            X_up[i, 3+r, 0] = -(R_total[r, 1]*oz - R_total[r, 2]*oy)
            X_up[i, 3+r, 1] = -(R_total[r, 2]*ox - R_total[r, 0]*oz)
            X_up[i, 3+r, 2] = -(R_total[r, 0]*oy - R_total[r, 1]*ox)

    # ------------------------------------------------------------------ #
    # Pass 1: Composite inertia accumulation (tips → root)                #
    # I_c[parent] += X_up[i]^T @ I_c[i] @ X_up[i]                       #
    # ------------------------------------------------------------------ #
    for i in range(n_bodies - 1, 0, -1):
        par = parent[i]
        # tmp = I_c[i] @ X_up[i]   (6x6)
        tmp = np.zeros((6, 6))
        for r in range(6):
            for cc in range(6):
                for k in range(6):
                    tmp[r, cc] += I_c[i, r, k] * X_up[i, k, cc]
        # I_c[par] += X_up[i]^T @ tmp
        for r in range(6):
            for cc in range(6):
                s_acc = 0.0
                for k in range(6):
                    s_acc += X_up[i, k, r] * tmp[k, cc]   # X^T[r,k] = X[k,r]
                I_c[par, r, cc] += s_acc

    # ------------------------------------------------------------------ #
    # Pass 2: Mass matrix extraction                                       #
    # ------------------------------------------------------------------ #

    # Floating base block (6×6)
    for r in range(6):
        for cc in range(6):
            M[r, cc] = I_c[0, r, cc]

    # Revolute joint columns
    for i in range(1, n_bodies):
        jtype = joint_types[i]
        if jtype < 1 or jtype > 3:
            continue                # Skip FIXED and FLOATING

        jaxis = joint_axes[i]       # 0, 1, or 2
        dof_i = 5 + i
        if dof_i >= n_dof:
            continue

        # F_i = I_c[i] @ S_i  where S_i is unit vector at jaxis
        # → F_i = I_c[i][:, jaxis]  (column selection)
        # .copy() is mandatory: F_up below mutates F_i
        F_i = I_c[i, :, jaxis].copy()

        # Diagonal: S_i^T @ F_i = F_i[jaxis]
        M[dof_i, dof_i] = F_i[jaxis]

        # Off-diagonal: walk up the tree toward root
        F_up = F_i.copy()
        j = i
        while True:
            # F_up = X_up[j]^T @ F_up
            tmp_f = np.zeros(6)
            for r in range(6):
                s_acc = 0.0
                for k in range(6):
                    s_acc += X_up[j, k, r] * F_up[k]   # X^T[r,k] = X[k,r]
                tmp_f[r] = s_acc
            F_up = tmp_f

            j = parent[j]

            if j == 0:
                # Floating base coupling block
                for r in range(6):
                    M[r, dof_i]    = F_up[r]
                    M[dof_i, r]    = F_up[r]
                break
            elif j < 0:
                break

            jtype_j = joint_types[j]
            if 1 <= jtype_j <= 3:
                jaxis_j = joint_axes[j]
                dof_j = 5 + j
                if dof_j < n_dof:
                    val = F_up[jaxis_j]    # S_j^T @ F_up (single element)
                    M[dof_i, dof_j] = val
                    M[dof_j, dof_i] = val

    return M
