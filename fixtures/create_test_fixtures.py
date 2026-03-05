"""Create synthetic NIfTI test fixtures with intentionally planted fake PHI.

All data is entirely synthetic — no real patient information is used.

Usage:
    python fixtures/create_test_fixtures.py
"""

import os

import nibabel as nib
import numpy as np

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def create_clean_fixture():
    """Create a clean NIfTI file with no PHI — just a synthetic brain-like volume."""
    filepath = os.path.join(FIXTURES_DIR, "test_clean.nii.gz")

    # 64^3 volume with a sphere in the center
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.int16)
    center = np.array(shape) / 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt(sum((np.array([x, y, z]) - center) ** 2))
                if dist < 25:
                    data[x, y, z] = int(1000 * (1 - dist / 25))

    img = nib.Nifti1Image(data, affine=np.eye(4))
    # Set header fields to safe tool signatures
    img.header["descrip"] = b"FSL5.0 processed"
    img.header["db_name"] = b""
    img.header["aux_file"] = b""
    nib.save(img, filepath)
    print(f"Created: {filepath}")
    return filepath


def create_phi_header_fixture():
    """Create a NIfTI file with fake PHI in header fields."""
    filepath = os.path.join(FIXTURES_DIR, "test_phi_header.nii.gz")

    data = np.zeros((32, 32, 32), dtype=np.int16)
    img = nib.Nifti1Image(data, affine=np.eye(4))

    # Plant fake PHI in header fields
    img.header["descrip"] = b"Patient: DOE^JANE 2024-01-15 Houston Methodist"
    img.header["db_name"] = b"MRN-12345678"
    img.header["aux_file"] = b"doe_jane_brain.nii"

    nib.save(img, filepath)
    print(f"Created: {filepath}")
    return filepath


def create_phi_text_fixture():
    """Create a NIfTI file with fake PHI burned into pixel data.

    Renders text onto edge slices where annotations typically appear.
    """
    filepath = os.path.join(FIXTURES_DIR, "test_phi_text.nii.gz")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow required for text fixture: pip install Pillow")
        return None

    shape = (128, 128, 64)
    data = np.zeros(shape, dtype=np.int16)

    # Add a sphere for "brain" content
    center = np.array(shape) / 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt(sum((np.array([x, y, z]) - center) ** 2))
                if dist < 30:
                    data[x, y, z] = int(500 * (1 - dist / 30))

    # Burn text into the first few axial slices (z=0,1,2)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    texts = [
        ("SMITH, JOHN A", (5, 5)),
        ("MRN: 9876543", (5, 25)),
        ("DOB: 1985-03-15", (5, 45)),
        ("Houston Methodist", (5, 65)),
    ]

    for z in range(3):
        img = Image.new("L", (shape[0], shape[1]), 0)
        draw = ImageDraw.Draw(img)
        for text, pos in texts:
            draw.text(pos, text, fill=255, font=font)
        text_array = np.array(img, dtype=np.int16)
        # Overlay text onto the slice
        data[:, :, z] = np.maximum(data[:, :, z], text_array)

    # Also burn text into last few axial slices
    for z in range(shape[2] - 3, shape[2]):
        img = Image.new("L", (shape[0], shape[1]), 0)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), "FACILITY: TMC IMAGING", fill=200, font=font)
        draw.text((5, 25), "DATE: 2024-01-15", fill=200, font=font)
        text_array = np.array(img, dtype=np.int16)
        data[:, :, z] = np.maximum(data[:, :, z], text_array)

    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.header["descrip"] = b"FSL5.0"  # safe header, PHI is in pixels only
    nib.save(img, filepath)
    print(f"Created: {filepath}")
    return filepath


def create_4d_fixture():
    """Create a 4D NIfTI file (e.g. fMRI timeseries) — clean, no PHI."""
    filepath = os.path.join(FIXTURES_DIR, "test_4d_clean.nii.gz")

    # 32^3 x 5 timepoints
    data = np.random.default_rng(42).integers(0, 500, size=(32, 32, 32, 5), dtype=np.int16)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.header["descrip"] = b"SPM12 realigned"
    nib.save(img, filepath)
    print(f"Created: {filepath}")
    return filepath


if __name__ == "__main__":
    create_clean_fixture()
    create_phi_header_fixture()
    create_phi_text_fixture()
    create_4d_fixture()
    print("All fixtures created.")
