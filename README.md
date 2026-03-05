# NIfTI PHI Scanner

Two-layer pipeline that detects protected health information (PHI) in NIfTI neuroimaging files (`.nii` / `.nii.gz`).

- **Layer 1 — Header inspection:** Checks `descrip`, `db_name`, `aux_file`, and `intent_name` fields against a safelist of known tool signatures (FSL, SPM, FreeSurfer, etc.). Unrecognized content is conservatively flagged.
- **Layer 2 — Pixel OCR:** Runs [EasyOCR](https://github.com/JaidedAI/EasyOCR) on sampled 2D slices extracted from the volume. All detected text above 0.30 confidence is flagged as potential burned-in PHI.

## Installation

Requires Python 3.10+.

```bash
pip install -e .

# With dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

GPU acceleration (CUDA) is used automatically when available. Pass `--cpu` to force CPU-only OCR.

## Usage

### Single file

```bash
nifti-phi-scan brain.nii.gz
nifti-phi-scan brain.nii.gz -o report.json
```

Prints a human-readable summary and optionally writes a JSON report. Exit code 1 if PHI is found, 0 if clean.

### Batch — directory scan

```bash
nifti-phi-scan --dir ./dataset -o results.jsonl
```

Recursively finds all `.nii` / `.nii.gz` files, prints per-file findings as each is scanned, then prints an aggregate summary with risk breakdown, top header fields, and top pixel text detections. Streams results to JSONL.

### Batch — manifest file

```bash
nifti-phi-scan --manifest files.txt -o results.jsonl
```

Reads one filepath per line from the manifest. Supports SLURM-style chunking:

```bash
nifti-phi-scan --manifest files.txt --chunk-size 100 --chunk-index 0 -o chunk0.jsonl
```

### Query JSONL output

```bash
jq 'select(.risk_level == "high") | .filepath' results.jsonl
```

### CLI flags

| Flag | Description |
|---|---|
| `-o`, `--output` | Write report to file (JSON for single, JSONL for batch) |
| `--dir` | Recursively scan a directory |
| `--manifest` | Read file paths from a text file |
| `--chunk-size` / `--chunk-index` | Chunked processing for SLURM array jobs |
| `--limit` | Max files to scan |
| `-L`, `--follow-symlinks` | Follow symlinks during directory scan |
| `--cpu` | Force CPU for OCR |
| `-v`, `--verbose` | Verbose logging |

## Architecture

```
scanner.scan_file()          # Orchestration entry point
├── header_scanner.py        # Layer 1: header field safelist matching
├── pixel_scanner.py         # Layer 2: EasyOCR on 2D slices
├── slice_extractor.py       # "Boundary + Quartile" sampling (~27 slices)
├── ocr_reader.py            # Lazy singleton EasyOCR reader w/ GPU auto-detect
├── models.py                # Pydantic: ScanReport, HeaderPHIFinding, PixelPHIFinding
└── cli.py                   # Three input modes, rich batch reporting
```

**Slice sampling strategy:** First/last 3 slices + 25/50/75% quartiles per axis (~27 slices total). Handles 4D→3D reduction and uint8 normalization.

## Development

```bash
# Generate synthetic test fixtures
python fixtures/create_test_fixtures.py

# Run tests
pytest

# Lint
ruff check .
```

All test and fixture data is synthetic — never use real patient data.

## License

MIT
