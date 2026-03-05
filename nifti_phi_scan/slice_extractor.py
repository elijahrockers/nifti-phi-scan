"""NIfTI volume loading and 2D slice sampling.

Implements a "Boundary + Quartile" sampling strategy that targets
annotation-prone regions (edges of the volume) while also sampling
representative interior slices. ~27 slices total per volume.
"""

from dataclasses import dataclass

import nibabel as nib
import numpy as np


AXIS_NAMES = ["sagittal", "coronal", "axial"]


@dataclass(frozen=True)
class SliceSpec:
    """Specification for extracting a single 2D slice from a 3D volume."""

    axis: int  # 0=sagittal, 1=coronal, 2=axial
    index: int
    total: int  # total slices along this axis

    @property
    def axis_name(self) -> str:
        return AXIS_NAMES[self.axis]


def compute_slice_plan(shape: tuple[int, ...]) -> list[SliceSpec]:
    """Compute which slices to extract from a 3D volume.

    Strategy per axis: first 3 + last 3 boundary slices, plus 25%/50%/75%.
    Deduplicates indices (small volumes may overlap).

    Args:
        shape: 3D volume shape (X, Y, Z).

    Returns:
        List of SliceSpec, ~27 for typical volumes.
    """
    spatial_shape = shape[:3]
    specs: list[SliceSpec] = []

    for axis in range(3):
        n = spatial_shape[axis]
        if n == 0:
            continue

        indices: set[int] = set()

        # Boundary slices: first 3 and last 3
        for i in range(min(3, n)):
            indices.add(i)
            indices.add(n - 1 - i)

        # Quartile slices: 25%, 50%, 75%
        for frac in (0.25, 0.50, 0.75):
            indices.add(int(n * frac))

        for idx in sorted(indices):
            specs.append(SliceSpec(axis=axis, index=idx, total=n))

    return specs


def load_volume(filepath: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load a NIfTI file and return 3D float32 voxel data.

    4D volumes are reduced to the first timepoint.

    Args:
        filepath: Path to .nii or .nii.gz file.

    Returns:
        Tuple of (3D float32 array, original shape before any 4D reduction).
    """
    img = nib.load(filepath)
    data = np.asarray(img.dataobj, dtype=np.float32)
    original_shape = data.shape

    # 4D → 3D: take first timepoint
    if data.ndim == 4:
        data = data[..., 0]
    elif data.ndim < 3:
        raise ValueError(f"Expected 3D+ volume, got {data.ndim}D shape {data.shape}")

    return data, original_shape


def extract_slice(data: np.ndarray, spec: SliceSpec) -> np.ndarray:
    """Extract a single 2D slice and normalize to uint8 (0-255).

    Args:
        data: 3D float32 volume.
        spec: Which slice to extract.

    Returns:
        2D uint8 array suitable for OCR.
    """
    slicing = [slice(None)] * 3
    slicing[spec.axis] = spec.index
    plane = data[tuple(slicing)]

    # Normalize to 0-255 uint8
    pmin, pmax = plane.min(), plane.max()
    if pmax == pmin:
        return np.zeros(plane.shape, dtype=np.uint8)
    normalized = ((plane - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    return normalized
