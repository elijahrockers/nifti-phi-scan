"""Layer 1: NIfTI header field PHI scanner.

Inspects NIfTI header fields that can contain free-text strings for
PHI-like content. Uses a safelist approach — known tool signatures are
excluded, and any remaining non-empty content is flagged conservatively.
"""

import re

import nibabel as nib
import numpy as np

from .models import HeaderPHIFinding, Severity

# Known safe patterns: common neuroimaging tool signatures and standard labels.
# Content matching these is not flagged.
SAFE_PATTERNS = re.compile(
    r"^("
    r"FSL.*|SPM.*|FreeSurfer.*|AFNI.*|ANTs.*|MRtrix.*|"
    r"BrainSuite.*|Connectome.*|BIDS.*|dcm2niix.*|mri_convert.*|"
    r"NIfTI-\d.*|nifti\b.*|analyze\b.*|"
    r"\+\+orig.*|\+\+tlrc.*|"  # AFNI conventions
    r"MNI\b.*|Talairach\b.*|"  # standard space labels
    r"TR=[\d.]+.*|TE=[\d.]+.*|"  # acquisition parameter summaries
    r"[a-z_]+\.(nii|img|hdr)(\.gz)?|"  # bare filenames without path separators
    r"[\d.]+\s*(mm|ms|s|Hz|deg|T)\b.*|"  # numeric values with units
    r"0*\.?0*"  # zeros / empty-ish
    r")$",
    re.IGNORECASE,
)

# Header fields to inspect: (field_name, severity, category)
HEADER_FIELDS: list[tuple[str, Severity, str]] = [
    ("descrip", Severity.HIGH, "free_text"),
    ("db_name", Severity.HIGH, "identifier"),
    ("aux_file", Severity.MEDIUM, "auxiliary"),
    ("intent_name", Severity.LOW, "identifier"),
]


def scan_header(img: nib.Nifti1Image) -> list[HeaderPHIFinding]:
    """Scan NIfTI header fields for potential PHI.

    Args:
        img: A loaded nibabel NIfTI image.

    Returns:
        List of HeaderPHIFinding for fields with suspicious content.
    """
    header = img.header
    findings: list[HeaderPHIFinding] = []

    for field_name, severity, category in HEADER_FIELDS:
        try:
            raw = header[field_name]
        except KeyError:
            continue

        # Decode to string — nibabel returns numpy ndarray with bytes dtype
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, (bytes, np.bytes_)):
            value = bytes(raw).decode("latin-1", errors="replace").strip().rstrip("\x00")
        elif isinstance(raw, str):
            value = raw.strip()
        else:
            value = str(raw).strip()

        if not value:
            continue

        if SAFE_PATTERNS.match(value):
            continue

        findings.append(
            HeaderPHIFinding(
                field_name=field_name,
                value=value,
                severity=severity,
                category=category,
            )
        )

    return findings
