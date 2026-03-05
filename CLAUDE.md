# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NIfTI PHI Scanner — a two-layer pipeline that detects protected health information in NIfTI neuroimaging files (.nii / .nii.gz). Layer 1 inspects header fields; Layer 2 runs EasyOCR on sampled pixel slices.

## Commands

```bash
pip install -e ".[dev]"                    # Install with dev dependencies
python fixtures/create_test_fixtures.py    # Generate synthetic test NIfTI files (requires Pillow for text fixture)
pytest                                     # Run all tests
pytest tests/test_scanner.py::test_phi_in_descrip  # Run a single test
ruff check .                               # Lint
nifti-phi-scan file.nii.gz -o report.json  # Scan single file
nifti-phi-scan --dir ./dataset -o out.jsonl  # Batch scan directory
nifti-phi-scan --manifest files.txt --chunk-size 100 --chunk-index 0  # SLURM-style chunked batch
```

## Architecture

Entry point: `scanner.scan_file()` orchestrates the two-layer pipeline:

1. **Header scanner** (`header_scanner.py`) — Inspects 4 NIfTI header fields (`descrip`, `db_name`, `aux_file`, `intent_name`) for non-empty content that doesn't match a safelist of known tool signatures (FSL, SPM, FreeSurfer, etc.). Any unrecognized content is conservatively flagged.

2. **Pixel scanner** (`pixel_scanner.py`) — Runs EasyOCR on 2D slices extracted from the volume. All detected text above 0.30 confidence is flagged as PHI (burned-in text in medical images is inherently suspicious).

Supporting modules:
- `slice_extractor.py` — "Boundary + Quartile" sampling: first/last 3 slices + 25/50/75% per axis (~27 slices total). Handles 4D→3D reduction and uint8 normalization.
- `ocr_reader.py` — Lazy singleton EasyOCR reader with GPU auto-detection (torch CUDA). Use `--cpu` CLI flag to force CPU.
- `models.py` — Pydantic models: `ScanReport`, `HeaderPHIFinding`, `PixelPHIFinding`, `FileError`. Reports serialize to JSON (single file) or JSONL (batch).
- `cli.py` — Three input modes (single file, `--dir` recursive, `--manifest` file list). Exit code 1 = PHI found, 0 = clean.

## Key Design Decisions

- Header scanning uses a **safelist** approach: known neuroimaging tool signatures are excluded, everything else is flagged. Add new safe patterns to `SAFE_PATTERNS` in `header_scanner.py`.
- Pixel scanning is **conservative** — any OCR-detected text is treated as potential PHI.
- Volume data is explicitly freed with `del` + `gc.collect()` after scanning to manage memory for large volumes.
- Tests for header scanning and slice extraction avoid the EasyOCR dependency by testing layers independently.

## Conventions

- Python 3.10+, ruff (line-length=100, target py310), pytest
- All test/fixture data is synthetic — never use real patient data
- PHI and patient privacy are primary concerns
