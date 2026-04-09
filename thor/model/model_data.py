"""Numba-compatible flattened model data for JIT dynamics algorithms.

RobotModel stores link properties as Python objects (LinkData instances),
which Numba's @njit mode cannot traverse.  This module extracts every
numeric field into contiguous NumPy arrays that Numba can accept directly.

Usage
-----
    model = RobotModel()
    md = model.model_data          # lazy-cached via property
    # or explicitly:
    from thor.model.model_data import flatten_model
    md = flatten_model(model)

The resulting ModelData is a plain dataclass (slots=True) with no Python
objects, making it safe to pass into @njit-compiled dynamics functions.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(slots=True)
class ModelData:
    """Flattened robot model data for Numba @njit functions.

    All fields are contiguous C-order NumPy arrays, avoiding Python
    objects that Numba cannot handle in nopython mode.

    Attributes
    ----------
    n_bodies : int
        Number of rigid bodies (35 for THOR).
    n_dof : int
        Total degrees of freedom (40 for THOR: 6 floating + 34 revolute).
    parent : NDArray[int32], shape (n_bodies,)
        Parent body index; -1 for the root (pelvis).
    joint_types : NDArray[int32], shape (n_bodies,)
        JointType enum values: 0=FIXED, 1=REV_X, 2=REV_Y, 3=REV_Z, 4=FLOATING.
    joint_axes : NDArray[int32], shape (n_bodies,)
        Primary axis index (0=X, 1=Y, 2=Z) for revolute joints.
    spatial_inertias : NDArray[float64], shape (n_bodies, 6, 6)
        6x6 spatial inertia matrices in body frame.
    joint_offsets : NDArray[float64], shape (n_bodies, 3)
        Translation from parent joint origin to child joint origin [m].
    joint_rotations : NDArray[float64], shape (n_bodies, 3, 3)
        Fixed rotation from parent frame to child joint frame.
    masses : NDArray[float64], shape (n_bodies,)
        Link masses [kg].
    coms : NDArray[float64], shape (n_bodies, 3)
        Centre-of-mass position in body frame [m].
    tau_max : NDArray[float64], shape (n_bodies,)
        Maximum joint torque [Nm].
    q_min : NDArray[float64], shape (n_bodies,)
        Lower joint position limit [rad].
    q_max : NDArray[float64], shape (n_bodies,)
        Upper joint position limit [rad].
    """
    n_bodies: int
    n_dof: int
    parent: NDArray              # (n_bodies,) int32
    joint_types: NDArray         # (n_bodies,) int32
    joint_axes: NDArray          # (n_bodies,) int32
    spatial_inertias: NDArray    # (n_bodies, 6, 6) float64
    joint_offsets: NDArray       # (n_bodies, 3) float64
    joint_rotations: NDArray     # (n_bodies, 3, 3) float64
    masses: NDArray              # (n_bodies,) float64
    coms: NDArray                # (n_bodies, 3) float64
    tau_max: NDArray             # (n_bodies,) float64
    q_min: NDArray               # (n_bodies,) float64
    q_max: NDArray               # (n_bodies,) float64


def flatten_model(model) -> ModelData:
    """Extract RobotModel data into Numba-compatible flat arrays.

    Iterates over ``model.links`` once and packs every numeric field
    into pre-allocated contiguous arrays.  The resulting ModelData can
    be passed directly to @njit-compiled dynamics kernels without
    triggering Numba's Python-object reflection path.

    Args:
        model: RobotModel instance (thor.model.robot_model.RobotModel).

    Returns:
        ModelData with all data in contiguous C-order float64/int32 arrays.
    """
    n = model.n_bodies

    spatial_inertias = np.zeros((n, 6, 6), dtype=np.float64)
    joint_offsets = np.zeros((n, 3), dtype=np.float64)
    joint_rotations = np.zeros((n, 3, 3), dtype=np.float64)
    masses = np.zeros(n, dtype=np.float64)
    coms = np.zeros((n, 3), dtype=np.float64)
    tau_max = np.zeros(n, dtype=np.float64)
    q_min_arr = np.full(n, -np.pi, dtype=np.float64)
    q_max_arr = np.full(n, np.pi, dtype=np.float64)

    for i, link in enumerate(model.links):
        spatial_inertias[i] = model.spatial_inertias[i]
        joint_offsets[i] = link.joint_offset
        joint_rotations[i] = link.joint_rotation
        masses[i] = link.mass
        coms[i] = link.com
        tau_max[i] = link.tau_max
        q_min_arr[i] = link.q_min
        q_max_arr[i] = link.q_max

    # Ensure arrays are C-contiguous for Numba
    return ModelData(
        n_bodies=n,
        n_dof=model.n_dof,
        parent=np.ascontiguousarray(model.parent.copy(), dtype=np.int32),
        joint_types=np.ascontiguousarray(model.joint_types.copy(), dtype=np.int32),
        joint_axes=np.ascontiguousarray(model.joint_axes.copy(), dtype=np.int32),
        spatial_inertias=np.ascontiguousarray(spatial_inertias),
        joint_offsets=np.ascontiguousarray(joint_offsets),
        joint_rotations=np.ascontiguousarray(joint_rotations),
        masses=np.ascontiguousarray(masses),
        coms=np.ascontiguousarray(coms),
        tau_max=np.ascontiguousarray(tau_max),
        q_min=np.ascontiguousarray(q_min_arr),
        q_max=np.ascontiguousarray(q_max_arr),
    )
