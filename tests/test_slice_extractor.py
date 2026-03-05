"""Tests for slice extraction and sampling strategy."""

import numpy as np

from nifti_phi_scan.slice_extractor import SliceSpec, compute_slice_plan, extract_slice


def test_slice_plan_typical_volume():
    """A typical 256x256x180 volume should produce ~27 unique slices."""
    plan = compute_slice_plan((256, 256, 180))
    assert len(plan) > 20
    assert len(plan) <= 30

    # Should cover all three axes
    axes = {s.axis for s in plan}
    assert axes == {0, 1, 2}


def test_slice_plan_includes_boundaries():
    """First and last slices should be included for each axis."""
    plan = compute_slice_plan((100, 100, 100))
    for axis in range(3):
        axis_specs = [s for s in plan if s.axis == axis]
        indices = {s.index for s in axis_specs}
        assert 0 in indices, f"First slice missing for axis {axis}"
        assert 99 in indices, f"Last slice missing for axis {axis}"


def test_slice_plan_includes_quartiles():
    """25%, 50%, 75% slices should be present."""
    plan = compute_slice_plan((100, 100, 100))
    for axis in range(3):
        indices = {s.index for s in plan if s.axis == axis}
        assert 25 in indices
        assert 50 in indices
        assert 75 in indices


def test_slice_plan_small_volume():
    """Small volumes should not crash, indices should be valid."""
    plan = compute_slice_plan((4, 4, 4))
    for spec in plan:
        assert 0 <= spec.index < spec.total


def test_slice_plan_deduplication():
    """Very small axis should deduplicate boundary/quartile overlaps."""
    plan = compute_slice_plan((2, 2, 2))
    for axis in range(3):
        axis_indices = [s.index for s in plan if s.axis == axis]
        assert len(axis_indices) == len(set(axis_indices)), "Duplicate indices found"


def test_extract_slice_normalization():
    """Extracted slices should be uint8 in range [0, 255]."""
    data = np.random.default_rng(42).uniform(0, 4000, size=(32, 32, 32)).astype(np.float32)
    spec = SliceSpec(axis=2, index=16, total=32)
    result = extract_slice(data, spec)

    assert result.dtype == np.uint8
    assert result.shape == (32, 32)
    assert result.min() >= 0
    assert result.max() <= 255


def test_extract_slice_constant_volume():
    """Constant volume should produce all-zero slices (no division error)."""
    data = np.full((10, 10, 10), 42.0, dtype=np.float32)
    spec = SliceSpec(axis=0, index=5, total=10)
    result = extract_slice(data, spec)

    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_slice_spec_axis_name():
    assert SliceSpec(axis=0, index=0, total=10).axis_name == "sagittal"
    assert SliceSpec(axis=1, index=0, total=10).axis_name == "coronal"
    assert SliceSpec(axis=2, index=0, total=10).axis_name == "axial"
