"""NIfTI PHI scanning pipeline.

Two-layer scan: header field inspection followed by OCR pixel scanning
on sampled slices from the volume.
"""

import gc
import logging

import nibabel as nib

from .header_scanner import scan_header
from .models import Severity, ScanReport
from .pixel_scanner import scan_slice
from .slice_extractor import compute_slice_plan, extract_slice, load_volume

logger = logging.getLogger(__name__)


def scan_file(filepath: str) -> ScanReport:
    """Scan a NIfTI file for PHI in headers and pixel data.

    Args:
        filepath: Path to .nii or .nii.gz file.

    Returns:
        ScanReport with all findings and recommendations.
    """
    img = nib.load(filepath)
    header_findings = scan_header(img)

    data, original_shape = load_volume(filepath)
    plan = compute_slice_plan(data.shape)

    pixel_findings = []
    for spec in plan:
        slice_data = extract_slice(data, spec)
        findings = scan_slice(slice_data, spec)
        pixel_findings.extend(findings)
        del slice_data

    del data
    gc.collect()

    total = len(header_findings) + len(pixel_findings)
    high_count = sum(1 for f in header_findings if f.severity == Severity.HIGH) + sum(
        1 for f in pixel_findings if f.severity == Severity.HIGH
    )

    recommendations = []
    if header_findings:
        recommendations.append(
            "Remove or redact PHI from NIfTI header fields before sharing"
        )
    if pixel_findings:
        recommendations.append(
            "Redact burned-in PHI text from voxel data at identified slice locations"
        )
    if not header_findings and not pixel_findings:
        recommendations.append("No PHI detected — file appears safe for sharing")

    if high_count > 0:
        risk_level = Severity.HIGH
    elif total > 0:
        risk_level = Severity.MEDIUM
    else:
        risk_level = Severity.LOW

    return ScanReport(
        filepath=filepath,
        shape=list(original_shape),
        n_dimensions=len(original_shape),
        slices_scanned=len(plan),
        header_findings=header_findings,
        pixel_findings=pixel_findings,
        total_phi_count=total,
        risk_level=risk_level,
        recommendations=recommendations,
    )
