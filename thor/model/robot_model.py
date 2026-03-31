"""
THOR 34-DOF humanoid robot model definition.

Defines the complete kinematic tree with:
- 35 bodies (pelvis + 34 joint-linked bodies)
- 34 revolute joints + 6-DOF floating base
- Mass/inertia properties based on THORMANG3 URDF (scaled to THOR specs)
- Link geometry for visualization

The kinematic tree structure:
    pelvis (floating base)
    ├── waist_yaw → waist_pitch → chest
    │   ├── head_yaw → head_pitch → head
    │   ├── l_arm: sh_p1 → sh_r → sh_p2 → el_y → wr_r → wr_y → wr_p
    │   └── r_arm: sh_p1 → sh_r → sh_p2 → el_y → wr_r → wr_y → wr_p
    ├── l_leg: hip_y → hip_r → hip_p → kn_p → an_p → an_r
    └── r_leg: hip_y → hip_r → hip_p → kn_p → an_p → an_r

Reference:
    Hopkins, M.A. & Leonessa, A. (2015). IJHR 12(3).
    THORMANG3 URDF: github.com/ROBOTIS-GIT/ROBOTIS-THORMANG-Common
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .joint_types import JointType
from .link import LinkData


def _inertia_box(m: float, dx: float, dy: float, dz: float) -> NDArray:
    """Inertia tensor of a uniform box (principal axes)."""
    return np.diag([
        m / 12.0 * (dy**2 + dz**2),
        m / 12.0 * (dx**2 + dz**2),
        m / 12.0 * (dx**2 + dy**2),
    ])


def _inertia_cylinder(m: float, r: float, h: float) -> NDArray:
    """Inertia tensor of a uniform cylinder along z-axis."""
    Ixx = m / 12.0 * (3 * r**2 + h**2)
    Izz = m / 2.0 * r**2
    return np.diag([Ixx, Ixx, Izz])


def build_thor_model() -> list[LinkData]:
    """Construct the THOR 34-DOF kinematic tree.

    Returns list of LinkData, index 0 = pelvis (floating base).
    Joint indices correspond to body indices (body i is connected
    to its parent by joint i).

    Physical properties scaled from THORMANG3 URDF to match
    THOR specifications (65 kg total, 1.78 m height).

    DOF mapping (velocity vector indices):
        0-5:   Floating base [vx, vy, vz, wx, wy, wz]
        6-7:   Waist [yaw, pitch]
        8-9:   Head [yaw, pitch]
        10-16: Left arm [sh_p1, sh_r, sh_p2, el_y, wr_r, wr_y, wr_p]
        17-23: Right arm [sh_p1, sh_r, sh_p2, el_y, wr_r, wr_y, wr_p]
        24-29: Left leg [hip_y, hip_r, hip_p, kn_p, an_p, an_r]
        30-35: Right leg [hip_y, hip_r, hip_p, kn_p, an_p, an_r]
        36-37: Left gripper [grip1, grip2]
        38-39: Right gripper [grip1, grip2]
    """
    # Scale factor: THORMANG3 (42kg, 1.375m) → THOR (65kg, 1.78m)
    mass_scale = 65.0 / 42.0     # ~1.548
    length_scale = 1.78 / 1.375  # ~1.295

    links: list[LinkData] = []

    # === Body 0: Pelvis (floating base) ===
    links.append(LinkData(
        name="pelvis",
        mass=6.869 * mass_scale,
        com=np.array([0.0, 0.0, 0.0]),
        inertia=_inertia_box(6.869 * mass_scale, 0.20 * length_scale,
                             0.25 * length_scale, 0.17 * length_scale),
        parent_id=-1,
        joint_type=JointType.FLOATING,
        is_actuated=False,
    ))

    # === Bodies 1-2: Waist (yaw, pitch) ===
    links.append(LinkData(
        name="waist_yaw",
        mass=0.5 * mass_scale,
        com=np.array([0.0, 0.0, 0.02 * length_scale]),
        inertia=_inertia_cylinder(0.5 * mass_scale, 0.05, 0.04) * length_scale**2,
        parent_id=0,
        joint_type=JointType.REVOLUTE_Z,
        joint_offset=np.array([0.0, 0.0, 0.17 * length_scale]),
        q_min=-1.2, q_max=1.2, tau_max=150.0,
    ))
    links.append(LinkData(
        name="waist_pitch",
        mass=5.383 * mass_scale,
        com=np.array([0.0, 0.0, 0.12 * length_scale]),
        inertia=_inertia_box(5.383 * mass_scale, 0.30 * length_scale,
                             0.20 * length_scale, 0.25 * length_scale),
        parent_id=1,
        joint_type=JointType.REVOLUTE_Y,
        joint_offset=np.array([0.0, 0.0, 0.05 * length_scale]),
        q_min=-0.6, q_max=0.6, tau_max=200.0,
    ))

    # === Bodies 3-4: Head (yaw, pitch) ===
    links.append(LinkData(
        name="head_yaw",
        mass=0.3 * mass_scale,
        com=np.array([0.0, 0.0, 0.02]),
        inertia=_inertia_cylinder(0.3 * mass_scale, 0.04, 0.03),
        parent_id=2,
        joint_type=JointType.REVOLUTE_Z,
        joint_offset=np.array([0.0, 0.0, 0.229 * length_scale]),
        q_min=-2.0, q_max=2.0, tau_max=20.0,
    ))
    links.append(LinkData(
        name="head_pitch",
        mass=1.0 * mass_scale,
        com=np.array([0.0, 0.0, 0.05]),
        inertia=_inertia_box(1.0 * mass_scale, 0.15, 0.15, 0.12),
        parent_id=3,
        joint_type=JointType.REVOLUTE_Y,
        joint_offset=np.array([0.0, 0.0, 0.05]),
        q_min=-0.7, q_max=0.7, tau_max=20.0,
    ))

    # === Helper for arm chain (7 DOF per arm) ===
    def _add_arm(side: str, parent: int, y_sign: float):
        """Add 7-DOF arm chain. Returns index of last link."""
        base_idx = len(links)
        shoulder_offset = np.array([0.0, y_sign * 0.186 * length_scale,
                                     0.20 * length_scale])

        arm_joints = [
            ("sh_p1", JointType.REVOLUTE_Y, np.array([0, 0, 0]),
             0.194, (-3.14, 3.14), 60.0),
            ("sh_r", JointType.REVOLUTE_X,
             np.array([0, y_sign * 0.03 * length_scale, 0]),
             0.875, (-1.6, 1.6), 60.0),
            ("sh_p2", JointType.REVOLUTE_Y,
             np.array([0, y_sign * 0.03 * length_scale, -0.06 * length_scale]),
             1.122, (-3.14, 3.14), 60.0),
            ("el_y", JointType.REVOLUTE_Z,
             np.array([0, 0, -0.25 * length_scale]),
             1.357, (-2.6, 2.6), 40.0),
            ("wr_r", JointType.REVOLUTE_X,
             np.array([0, 0, -0.20 * length_scale]),
             0.087, (-1.6, 1.6), 20.0),
            ("wr_y", JointType.REVOLUTE_Z,
             np.array([0, 0, -0.03 * length_scale]),
             0.768, (-1.6, 1.6), 20.0),
            ("wr_p", JointType.REVOLUTE_Y,
             np.array([0, 0, -0.03 * length_scale]),
             0.565, (-1.6, 1.6), 20.0),
        ]

        for i, (name, jtype, offset, mass_raw, qlim, tmax) in enumerate(arm_joints):
            p = parent if i == 0 else len(links) - 1
            off = shoulder_offset if i == 0 else offset
            links.append(LinkData(
                name=f"{side}_{name}",
                mass=mass_raw * mass_scale,
                com=np.array([0, 0, -0.03 * length_scale]),
                inertia=_inertia_cylinder(mass_raw * mass_scale,
                                          0.03 * length_scale,
                                          0.10 * length_scale),
                parent_id=p,
                joint_type=jtype,
                joint_offset=off,
                q_min=qlim[0], q_max=qlim[1], tau_max=tmax,
            ))

    # === Helper for leg chain (6 DOF per leg) ===
    def _add_leg(side: str, parent: int, y_sign: float):
        """Add 6-DOF leg chain."""
        hip_offset = np.array([0.0, y_sign * 0.093 * length_scale,
                                -0.09 * length_scale])

        leg_joints = [
            ("hip_y", JointType.REVOLUTE_Z, np.array([0, 0, 0]),
             0.243, (-1.0, 1.0), 200.0),
            ("hip_r", JointType.REVOLUTE_X,
             np.array([0, y_sign * 0.01 * length_scale, -0.04 * length_scale]),
             1.045, (-0.5, 0.5), 200.0),
            ("hip_p", JointType.REVOLUTE_Y,
             np.array([0, 0, -0.04 * length_scale]),
             3.095, (-1.7, 0.5), 289.0),  # THOR SEA: 289 Nm
            ("kn_p", JointType.REVOLUTE_Y,
             np.array([0, 0, -0.30 * length_scale]),
             2.401, (-0.1, 2.6), 289.0),
            ("an_p", JointType.REVOLUTE_Y,
             np.array([0, 0, -0.30 * length_scale]),
             1.045, (-1.2, 0.7), 115.0),
            ("an_r", JointType.REVOLUTE_X,
             np.array([0, 0, -0.05 * length_scale]),
             1.689, (-0.5, 0.5), 115.0),
        ]

        for i, (name, jtype, offset, mass_raw, qlim, tmax) in enumerate(leg_joints):
            p = parent if i == 0 else len(links) - 1
            off = hip_offset if i == 0 else offset
            links.append(LinkData(
                name=f"{side}_{name}",
                mass=mass_raw * mass_scale,
                com=np.array([0, 0, -0.05 * length_scale]) if "hip" in name or "kn" in name
                     else np.array([0, 0, -0.02 * length_scale]),
                inertia=_inertia_cylinder(mass_raw * mass_scale,
                                          0.04 * length_scale,
                                          0.15 * length_scale) if "hip_p" in name or "kn" in name
                         else _inertia_cylinder(mass_raw * mass_scale,
                                                0.03 * length_scale,
                                                0.06 * length_scale),
                parent_id=p,
                joint_type=jtype,
                joint_offset=off,
                q_min=qlim[0], q_max=qlim[1], tau_max=tmax,
            ))

    # === Helper for gripper (2 DOF per hand) ===
    def _add_gripper(side: str, parent: int):
        for j in range(2):
            links.append(LinkData(
                name=f"{side}_grip{j+1}",
                mass=0.1 * mass_scale,
                com=np.array([0, 0, -0.02]),
                inertia=_inertia_box(0.1 * mass_scale, 0.02, 0.06, 0.02),
                parent_id=parent if j == 0 else len(links) - 1,
                joint_type=JointType.REVOLUTE_Y,
                joint_offset=np.array([0, 0, -0.05 * length_scale]),
                q_min=-0.5, q_max=1.2, tau_max=5.0,
            ))

    # Build arms (parent = chest = body 2)
    _add_arm("l_arm", 2, 1.0)    # Bodies 5-11 (7 links)
    _add_arm("r_arm", 2, -1.0)   # Bodies 12-18 (7 links)

    # Build legs (parent = pelvis = body 0)
    _add_leg("l_leg", 0, 1.0)    # Bodies 19-24 (6 links)
    _add_leg("r_leg", 0, -1.0)   # Bodies 25-30 (6 links)

    # Build grippers
    _add_gripper("l", 11)         # Bodies 31-32 (2 links, parent = l_wr_p)
    _add_gripper("r", 18)         # Bodies 33-34 (2 links, parent = r_wr_p)

    return links


class RobotModel:
    """THOR 34-DOF robot model with precomputed kinematic data.

    Provides efficient access to link properties, parent indices,
    and joint information for the dynamics algorithms.
    """

    __slots__ = (
        "links", "n_bodies", "n_joints", "n_dof",
        "parent", "joint_types", "joint_axes",
        "_joint_name_to_idx", "total_mass",
        "foot_link_ids", "spatial_inertias", "motion_subspaces",
    )

    def __init__(self) -> None:
        self.links = build_thor_model()
        self.n_bodies = len(self.links)
        self.n_joints = self.n_bodies  # Each body has one joint to parent
        self.n_dof = 6 + (self.n_bodies - 1)  # floating base + revolute joints

        # Parent array for O(1) traversal
        self.parent = np.array([l.parent_id for l in self.links], dtype=np.int32)

        # Joint type/axis arrays
        self.joint_types = np.array([l.joint_type for l in self.links], dtype=np.int32)
        self.joint_axes = np.array([l.joint_axis for l in self.links], dtype=np.int32)

        # Name-to-index mapping
        self._joint_name_to_idx = {l.name: i for i, l in enumerate(self.links)}

        # Total mass
        self.total_mass = sum(l.mass for l in self.links)

        # Foot link IDs for contact
        self.foot_link_ids = (
            self._joint_name_to_idx.get("l_leg_an_r", -1),
            self._joint_name_to_idx.get("r_leg_an_r", -1),
        )

        # Cache spatial inertias (constant, computed once at model load)
        from ..core.spatial import spatial_inertia, motion_subspace_revolute
        self.spatial_inertias = [
            spatial_inertia(l.mass, l.com, l.inertia) for l in self.links
        ]

        # Note: motion_subspace_revolute() is fast enough inline
        # (benchmarked: caching was 2.3% slower due to list lookup overhead)
        self.motion_subspaces = None  # Unused, kept for slot compatibility

    def joint_index(self, name: str) -> int:
        """Get joint/body index by name."""
        return self._joint_name_to_idx[name]

    def get_link(self, idx: int) -> LinkData:
        """Get link data by index."""
        return self.links[idx]
