"""Tests for the header scanner and full scan pipeline.

These tests use nibabel to create in-memory NIfTI images — no OCR dependency.
"""

import numpy as np
import nibabel as nib

from nifti_phi_scan.header_scanner import scan_header
from nifti_phi_scan.models import Severity


def _make_image(descrip: bytes = b"", db_name: bytes = b"", aux_file: bytes = b"") -> nib.Nifti1Image:
    """Helper: create a minimal NIfTI image with specified header fields."""
    data = np.zeros((8, 8, 8), dtype=np.int16)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    if descrip:
        img.header["descrip"] = descrip
    if db_name:
        img.header["db_name"] = db_name
    if aux_file:
        img.header["aux_file"] = aux_file
    return img


def test_clean_header_fsl():
    """FSL tool signature should not be flagged."""
    img = _make_image(descrip=b"FSL5.0 processed")
    findings = scan_header(img)
    assert len(findings) == 0


def test_clean_header_spm():
    """SPM tool signature should not be flagged."""
    img = _make_image(descrip=b"SPM12 realigned")
    findings = scan_header(img)
    assert len(findings) == 0


def test_clean_header_empty():
    """Empty header fields should not be flagged."""
    img = _make_image()
    findings = scan_header(img)
    assert len(findings) == 0


def test_phi_in_descrip():
    """Patient info in descrip should be flagged as HIGH."""
    img = _make_image(descrip=b"Patient: DOE^JANE 2024-01-15")
    findings = scan_header(img)
    assert len(findings) == 1
    assert findings[0].field_name == "descrip"
    assert findings[0].severity == Severity.HIGH
    assert findings[0].category == "free_text"


def test_phi_in_db_name():
    """MRN in db_name should be flagged as HIGH."""
    img = _make_image(db_name=b"MRN-12345678")
    findings = scan_header(img)
    assert len(findings) == 1
    assert findings[0].field_name == "db_name"
    assert findings[0].severity == Severity.HIGH


def test_phi_in_aux_file():
    """Patient name in aux_file should be flagged as MEDIUM."""
    img = _make_image(aux_file=b"doe_jane_scan1.txt")
    findings = scan_header(img)
    assert len(findings) == 1
    assert findings[0].field_name == "aux_file"
    assert findings[0].severity == Severity.MEDIUM


def test_safe_filename_in_aux_file():
    """A bare .nii filename in aux_file should NOT be flagged."""
    img = _make_image(aux_file=b"brain_mask.nii.gz")
    findings = scan_header(img)
    assert len(findings) == 0


def test_multiple_phi_fields():
    """Multiple PHI fields should all be reported."""
    img = _make_image(
        descrip=b"John Doe scan session 2024",
        db_name=b"PATIENT_42",
    )
    findings = scan_header(img)
    assert len(findings) == 2
    field_names = {f.field_name for f in findings}
    assert "descrip" in field_names
    assert "db_name" in field_names
