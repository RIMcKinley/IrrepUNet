"""Tests for irrepunet.data.spacing module."""

import numpy as np
import pytest

from irrepunet.data.spacing import (
    SPACING_GRID,
    round_to_grid,
    round_spacing_to_tolerance,
    group_cases_by_spacing,
    get_canonical_permutation,
    apply_axis_permutation,
)


# ---------------------------------------------------------------------------
# SPACING_GRID
# ---------------------------------------------------------------------------

class TestSpacingGrid:
    def test_grid_is_sorted(self):
        assert SPACING_GRID == tuple(sorted(SPACING_GRID))

    def test_grid_has_expected_length(self):
        assert len(SPACING_GRID) == 16

    def test_grid_starts_at_025(self):
        assert SPACING_GRID[0] == 0.25

    def test_grid_ends_at_11(self):
        assert SPACING_GRID[-1] == 11.0

    def test_grid_values_positive(self):
        assert all(v > 0 for v in SPACING_GRID)


# ---------------------------------------------------------------------------
# round_to_grid
# ---------------------------------------------------------------------------

class TestRoundToGrid:
    @pytest.mark.parametrize("value, expected", [
        (1.0, 1.0),    # exact match
        (0.25, 0.25),  # exact min
        (11.0, 11.0),  # exact max
    ])
    def test_exact_grid_values(self, value, expected):
        assert round_to_grid(value) == expected

    @pytest.mark.parametrize("value, expected", [
        (0.3, 0.25),   # rounds down to 0.25
        (0.6, 0.5),    # rounds to 0.5
        (0.9, 1.0),    # rounds up to 1.0
        (1.2, 1.0),    # rounds to 1.0
        (1.3, 1.5),    # rounds to 1.5
        (4.4, 4.0),    # rounds to 4.0
        (4.6, 5.0),    # rounds to 5.0
    ])
    def test_rounding_to_nearest(self, value, expected):
        assert round_to_grid(value) == expected

    def test_very_small_value(self):
        """Values below min grid should round to min."""
        assert round_to_grid(0.1) == 0.25

    def test_very_large_value(self):
        """Values above max grid should round to max."""
        assert round_to_grid(100.0) == 11.0

    def test_custom_grid(self):
        grid = (1.0, 2.0, 5.0)
        assert round_to_grid(3.0, grid=grid) == 2.0
        assert round_to_grid(4.0, grid=grid) == 5.0


# ---------------------------------------------------------------------------
# round_spacing_to_tolerance
# ---------------------------------------------------------------------------

class TestRoundSpacingToTolerance:
    def test_exact_values(self):
        assert round_spacing_to_tolerance((1.0, 1.0, 1.0)) == (1.0, 1.0, 1.0)

    def test_rounds_each_axis(self):
        result = round_spacing_to_tolerance((0.9, 1.2, 4.6))
        assert result == (1.0, 1.0, 5.0)

    def test_anisotropic(self):
        result = round_spacing_to_tolerance((0.5, 0.5, 5.0))
        assert result == (0.5, 0.5, 5.0)


# ---------------------------------------------------------------------------
# group_cases_by_spacing — Permutation invariance
# ---------------------------------------------------------------------------

class TestGroupCasesBySpacing:
    def _make_properties(self, spacings_dict):
        """Make a properties dict from case->spacing mapping."""
        return {
            case: {"spacing": sp}
            for case, sp in spacings_dict.items()
        }

    def test_basic_grouping(self):
        props = self._make_properties({
            "case_a": (1.0, 1.0, 1.0),
            "case_b": (1.0, 1.0, 1.0),
        })
        groups = group_cases_by_spacing(props)
        assert len(groups) == 1
        key = list(groups.keys())[0]
        assert set(groups[key]) == {"case_a", "case_b"}

    def test_permutation_invariance(self):
        """Axis permutations should end up in the same group."""
        props = self._make_properties({
            "case_a": (0.5, 0.5, 5.0),
            "case_b": (5.0, 0.5, 0.5),
            "case_c": (0.5, 5.0, 0.5),
        })
        groups = group_cases_by_spacing(props)
        assert len(groups) == 1
        key = list(groups.keys())[0]
        assert len(groups[key]) == 3

    def test_canonical_form_sorted_ascending(self):
        """Group keys should be sorted ascending (fine to coarse)."""
        props = self._make_properties({
            "case_a": (5.0, 0.5, 1.0),
        })
        groups = group_cases_by_spacing(props)
        key = list(groups.keys())[0]
        assert key == tuple(sorted(key)), f"Key {key} not sorted ascending"

    def test_different_spacings_separate_groups(self):
        props = self._make_properties({
            "case_a": (1.0, 1.0, 1.0),
            "case_b": (1.0, 1.0, 5.0),
        })
        groups = group_cases_by_spacing(props)
        assert len(groups) == 2

    def test_filter_min_spacing(self):
        props = self._make_properties({
            "case_a": (0.3, 0.3, 1.0),
            "case_b": (1.0, 1.0, 1.0),
        })
        groups = group_cases_by_spacing(props, min_spacing=0.5)
        # case_a has in-plane 0.3 < 0.5, should be excluded
        total_cases = sum(len(v) for v in groups.values())
        assert total_cases == 1

    def test_filter_max_slice_thickness(self):
        props = self._make_properties({
            "case_a": (1.0, 1.0, 10.0),
            "case_b": (1.0, 1.0, 3.0),
        })
        groups = group_cases_by_spacing(props, max_slice_thickness=5.0)
        total_cases = sum(len(v) for v in groups.values())
        assert total_cases == 1

    def test_empty_properties(self):
        groups = group_cases_by_spacing({})
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# get_canonical_permutation
# ---------------------------------------------------------------------------

class TestGetCanonicalPermutation:
    def test_identity(self):
        """Already sorted spacing returns identity permutation."""
        perm = get_canonical_permutation((0.5, 1.0, 5.0))
        assert tuple(perm) == (0, 1, 2)

    def test_reversed(self):
        perm = get_canonical_permutation((5.0, 1.0, 0.5))
        assert tuple(perm) == (2, 1, 0)

    def test_applying_permutation_sorts(self):
        spacing = (3.0, 1.0, 5.0)
        perm = get_canonical_permutation(spacing)
        sorted_sp = tuple(np.array(spacing)[list(perm)])
        assert sorted_sp == tuple(sorted(spacing))


# ---------------------------------------------------------------------------
# apply_axis_permutation
# ---------------------------------------------------------------------------

class TestApplyAxisPermutation:
    def test_identity_perm(self):
        arr = np.random.randn(1, 4, 8, 16)
        result = apply_axis_permutation(arr, (0, 1, 2), has_channel=True)
        np.testing.assert_array_equal(result, arr)

    def test_swap_axes(self):
        arr = np.zeros((1, 4, 8, 16))
        result = apply_axis_permutation(arr, (2, 1, 0), has_channel=True)
        assert result.shape == (1, 16, 8, 4)

    def test_no_channel(self):
        arr = np.zeros((4, 8, 16))
        result = apply_axis_permutation(arr, (2, 1, 0), has_channel=False)
        assert result.shape == (16, 8, 4)


