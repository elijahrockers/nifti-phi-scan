"""Pydantic models for NIfTI PHI detection results."""

from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BoundingBox(BaseModel):
    """Pixel coordinates for detected text region within a 2D slice."""

    x: int
    y: int
    width: int
    height: int


class SliceLocation(BaseModel):
    """Locates a 2D slice within the 3D volume."""

    axis: str  # "sagittal", "coronal", or "axial"
    index: int
    total: int


class HeaderPHIFinding(BaseModel):
    """A PHI finding from NIfTI header field inspection."""

    field_name: str
    value: str
    severity: Severity
    category: str  # "free_text", "identifier", "auxiliary"


class PixelPHIFinding(BaseModel):
    """A PHI finding from burned-in pixel text detected via OCR."""

    text: str
    bbox: BoundingBox
    confidence: float
    severity: Severity
    slice_location: SliceLocation


class ScanReport(BaseModel):
    """Complete PHI scan report for a NIfTI file."""

    filepath: str
    shape: list[int]
    n_dimensions: int
    slices_scanned: int
    header_findings: list[HeaderPHIFinding]
    pixel_findings: list[PixelPHIFinding]
    total_phi_count: int
    risk_level: Severity
    recommendations: list[str]

    @property
    def has_phi(self) -> bool:
        return self.total_phi_count > 0


class FileError(BaseModel):
    """A per-file error encountered during batch scanning."""

    filepath: str
    error: str
