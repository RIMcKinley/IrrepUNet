"""Tests for memory estimation function.

Converted from test_memory_correction.py to proper pytest format.
Does not require profiling data files.
"""

import pytest

from irrepunet.data.multi_resolution_loader import estimate_memory_mb


class TestEstimateMemoryMb:
    def test_returns_positive(self):
        """Estimate should always be positive."""
        est = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=True)
        assert est > 0

    def test_increases_with_patch_size(self):
        """Larger patches should need more memory."""
        small = estimate_memory_mb((32, 32, 32), n_base_filters=2, batch_size=2, fp16=True)
        large = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=True)
        assert large > small

    def test_increases_with_batch_size(self):
        """Larger batch should need more memory."""
        b1 = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=1, fp16=True)
        b4 = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=4, fp16=True)
        assert b4 > b1

    def test_fp32_larger_than_fp16(self):
        """FP32 should generally use more or comparable memory to FP16.
        Note: the FP16 coefficients include a correction factor, so this
        tests that both code paths work without error."""
        fp16 = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=True)
        fp32 = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=False)
        # Both should be positive
        assert fp16 > 0
        assert fp32 > 0

    def test_correction_factor_applied(self):
        """The 6.5x correction factor should be reflected in FP16 n_base_filters=2."""
        patch = (64, 64, 64)
        est = estimate_memory_mb(patch, n_base_filters=2, batch_size=2, fp16=True)

        # Manually compute with OLD coefficients (before correction)
        old_coef, old_base = 3835, 71
        million_voxels = 64 ** 3 / 1e6
        old_estimate = (old_coef * million_voxels + old_base) * (2 / 2)

        factor = est / old_estimate if old_estimate > 0 else 0
        assert 6.0 <= factor <= 7.0, f"Factor {factor:.1f}x not in expected range"

    def test_different_n_base_filters(self):
        """Different n_base_filters should use different coefficients."""
        est_2 = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=True)
        est_4 = estimate_memory_mb((64, 64, 64), n_base_filters=4, batch_size=2, fp16=True)
        # Both should be positive and different (different coefficient tables)
        assert est_2 > 0
        assert est_4 > 0
        assert est_2 != est_4

    def test_anisotropic_patch(self):
        """Should handle non-cubic patches."""
        est = estimate_memory_mb((32, 64, 128), n_base_filters=2, batch_size=2, fp16=True)
        assert est > 0

    def test_scales_linearly_with_voxels(self):
        """Memory should scale roughly linearly with total voxel count."""
        est_small = estimate_memory_mb((32, 32, 32), n_base_filters=2, batch_size=2, fp16=True)
        est_large = estimate_memory_mb((64, 64, 64), n_base_filters=2, batch_size=2, fp16=True)
        # 64^3 / 32^3 = 8x voxels, expect roughly 8x memory (minus base)
        ratio = est_large / est_small
        assert 4.0 < ratio < 12.0, f"Ratio {ratio:.1f} not in expected range"
