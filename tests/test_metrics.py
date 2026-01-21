"""Tests for the metrics module."""

import numpy as np
import pytest

from opensight.analysis.metrics import (
    calculate_angle_between_vectors,
    angles_to_direction,
)


class TestAngleCalculations:
    """Tests for angle calculation utilities."""

    def test_same_direction_zero_angle(self):
        """Verify same direction gives zero angle."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        angle = calculate_angle_between_vectors(v1, v2)
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_opposite_direction_180_angle(self):
        """Verify opposite direction gives 180 degrees."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        angle = calculate_angle_between_vectors(v1, v2)
        assert angle == pytest.approx(180.0, abs=0.01)

    def test_perpendicular_90_angle(self):
        """Verify perpendicular vectors give 90 degrees."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle_between_vectors(v1, v2)
        assert angle == pytest.approx(90.0, abs=0.01)

    def test_45_degree_angle(self):
        """Verify 45 degree angle calculation."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        angle = calculate_angle_between_vectors(v1, v2)
        assert angle == pytest.approx(45.0, abs=0.01)


class TestAnglesToDirection:
    """Tests for pitch/yaw to direction conversion."""

    def test_zero_angles_forward(self):
        """Verify zero pitch/yaw points forward (+x)."""
        direction = angles_to_direction(0, 0)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(direction, expected, decimal=5)

    def test_yaw_90_points_left(self):
        """Verify yaw 90 points in +y direction."""
        direction = angles_to_direction(0, 90)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(direction, expected, decimal=5)

    def test_pitch_negative_looks_up(self):
        """Verify negative pitch looks up (+z)."""
        direction = angles_to_direction(-90, 0)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(direction, expected, decimal=5)

    def test_pitch_positive_looks_down(self):
        """Verify positive pitch looks down (-z)."""
        direction = angles_to_direction(90, 0)
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(direction, expected, decimal=5)

    def test_direction_is_unit_vector(self):
        """Verify returned direction is unit vector."""
        direction = angles_to_direction(30, 45)
        magnitude = np.linalg.norm(direction)
        assert magnitude == pytest.approx(1.0, abs=0.0001)
