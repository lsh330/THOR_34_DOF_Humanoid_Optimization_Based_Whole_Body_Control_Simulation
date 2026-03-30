"""
Spatial vector algebra package (Featherstone 2008).

Re-exports all spatial algebra functions for backward compatibility.
"""

from .rotation import skew, skew_vec, rot_x, rot_y, rot_z
from .transform import spatial_transform, spatial_transform_inv
from .inertia import spatial_inertia
from .cross_product import spatial_cross_motion, spatial_cross_force
from .motion_subspace import motion_subspace_revolute, motion_subspace_prismatic

__all__ = [
    "skew", "skew_vec", "rot_x", "rot_y", "rot_z",
    "spatial_transform", "spatial_transform_inv",
    "spatial_inertia",
    "spatial_cross_motion", "spatial_cross_force",
    "motion_subspace_revolute", "motion_subspace_prismatic",
]
