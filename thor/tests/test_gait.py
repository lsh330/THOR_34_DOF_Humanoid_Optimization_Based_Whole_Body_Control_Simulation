"""
Tests for gait generation components.

Validates phase detection timing and trajectory continuity.
"""

import math
import numpy as np
import pytest

from thor.control.gait.phase_detector import detect_phase
from thor.control.gait.swing_trajectory import (
    swing_leg_angles, stance_leg_angles,
    HIP_SWING_FLEX, HIP_STANCE_EXT,
)


DS = 0.25
SW = 0.55


class TestPhaseDetector:
    def test_initial_ds(self):
        p = detect_phase(0.0, DS, SW, 4)
        assert p["phase"] == "ds"

    def test_first_swing(self):
        p = detect_phase(0.35, DS, SW, 4)
        assert p["phase"] == "swing_left"

    def test_second_swing(self):
        p = detect_phase(1.15, DS, SW, 4)
        assert p["phase"] == "swing_right"

    def test_ds_transition(self):
        p = detect_phase(0.85, DS, SW, 4)
        assert p["phase"] == "ds"

    def test_phase_s_range(self):
        """Phase progress s must be in [0, 1]."""
        for t in np.linspace(0, 4.0, 100):
            p = detect_phase(t, DS, SW, 4)
            assert 0 <= p["s"] <= 1.0 + 1e-10

    def test_all_phases_visited(self):
        """Over a full gait cycle, all phases should appear."""
        phases_seen = set()
        for t in np.linspace(0, 3.5, 200):
            p = detect_phase(t, DS, SW, 4)
            phases_seen.add(p["phase"])
        assert "ds" in phases_seen
        assert "swing_left" in phases_seen
        assert "swing_right" in phases_seen


class TestTrajectorySmoothnessm:
    def test_swing_continuous(self):
        """Swing trajectory should be C0 continuous."""
        prev_hip, prev_kn, prev_an = swing_leg_angles(0.0)
        for s in np.linspace(0.01, 1.0, 100):
            hip, kn, an = swing_leg_angles(s)
            # Difference between consecutive samples should be small
            assert abs(hip - prev_hip) < 0.1  # < 5.7 deg per 1% phase
            assert abs(kn - prev_kn) < 0.2
            assert abs(an - prev_an) < 0.05
            prev_hip, prev_kn, prev_an = hip, kn, an

    def test_stance_continuous(self):
        prev_hip, prev_kn, prev_an = stance_leg_angles(0.0)
        for s in np.linspace(0.01, 1.0, 100):
            hip, kn, an = stance_leg_angles(s)
            assert abs(hip - prev_hip) < 0.1
            assert abs(kn - prev_kn) < 0.1
            assert abs(an - prev_an) < 0.1
            prev_hip, prev_kn, prev_an = hip, kn, an

    def test_swing_stance_boundary_smooth(self):
        """At s=0, swing start should be close to stance end."""
        swing_start = swing_leg_angles(0.0)
        stance_end = stance_leg_angles(1.0)
        for a, b in zip(swing_start, stance_end):
            assert abs(a - b) < math.radians(30)  # Within 30 deg
