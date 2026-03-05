"""Tests for Pydantic models."""

import json

from nifti_phi_scan.models import (
    BoundingBox,
    FileError,
    HeaderPHIFinding,
    PixelPHIFinding,
    ScanReport,
    Severity,
    SliceLocation,
)


def test_severity_values():
    assert Severity.HIGH == "high"
    assert Severity.MEDIUM == "medium"
    assert Severity.LOW == "low"


def test_header_finding_serialization():
    finding = HeaderPHIFinding(
        field_name="descrip",
        value="Patient: DOE^JANE",
        severity=Severity.HIGH,
        category="free_text",
    )
    data = json.loads(finding.model_dump_json())
    assert data["field_name"] == "descrip"
    assert data["severity"] == "high"


def test_pixel_finding_serialization():
    finding = PixelPHIFinding(
        text="SMITH, JOHN",
        bbox=BoundingBox(x=10, y=20, width=100, height=15),
        confidence=0.92,
        severity=Severity.HIGH,
        slice_location=SliceLocation(axis="axial", index=0, total=64),
    )
    data = json.loads(finding.model_dump_json())
    assert data["text"] == "SMITH, JOHN"
    assert data["slice_location"]["axis"] == "axial"
    assert data["bbox"]["width"] == 100


def test_scan_report_has_phi():
    report = ScanReport(
        filepath="test.nii.gz",
        shape=[64, 64, 64],
        n_dimensions=3,
        slices_scanned=27,
        header_findings=[],
        pixel_findings=[],
        total_phi_count=0,
        risk_level=Severity.LOW,
        recommendations=["No PHI detected"],
    )
    assert not report.has_phi

    report_with_phi = report.model_copy(update={"total_phi_count": 3})
    assert report_with_phi.has_phi


def test_file_error_serialization():
    error = FileError(filepath="bad.nii", error="corrupted file")
    data = json.loads(error.model_dump_json())
    assert data["filepath"] == "bad.nii"
    assert "corrupted" in data["error"]
