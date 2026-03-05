"""Layer 2: OCR-based pixel PHI detection on NIfTI slices.

Runs EasyOCR on extracted 2D slices and flags all detected text as
potential PHI — burned-in text in medical images is inherently suspicious.
"""

import numpy as np

from .models import BoundingBox, PixelPHIFinding, Severity, SliceLocation
from .ocr_reader import MIN_OCR_CONFIDENCE, get_reader
from .slice_extractor import SliceSpec


def scan_slice(slice_data: np.ndarray, spec: SliceSpec) -> list[PixelPHIFinding]:
    """Run OCR on a single 2D slice and return PHI findings.

    Args:
        slice_data: uint8 2D array (already normalized).
        spec: Slice location metadata.

    Returns:
        List of PixelPHIFinding for detected text.
    """
    reader = get_reader()
    ocr_results = reader.readtext(slice_data)

    findings: list[PixelPHIFinding] = []
    for bbox_pts, text, conf in ocr_results:
        text = text.strip()
        if not text or conf < MIN_OCR_CONFIDENCE:
            continue

        # bbox_pts is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [pt[0] for pt in bbox_pts]
        ys = [pt[1] for pt in bbox_pts]
        x = int(min(xs))
        y = int(min(ys))
        width = int(max(xs) - x)
        height = int(max(ys) - y)

        findings.append(
            PixelPHIFinding(
                text=text,
                bbox=BoundingBox(x=x, y=y, width=width, height=height),
                confidence=round(conf, 4),
                severity=Severity.HIGH,
                slice_location=SliceLocation(
                    axis=spec.axis_name,
                    index=spec.index,
                    total=spec.total,
                ),
            )
        )

    return findings
