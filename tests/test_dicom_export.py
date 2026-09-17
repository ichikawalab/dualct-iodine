# -*- coding: utf-8 -*-
"""Derived DICOM export keeps the source series' slice numbering.

The volume is stacked in ascending slice-position order, while many acquisitions
number their slices the other way (InstanceNumber 1 at the highest z). The exported
series must stamp each derived slice with its *source* slice's InstanceNumber and
name the file after it, so viewers that stack by InstanceNumber or file name (ImageJ)
show the prediction in the same order as the source. Pixel content must still follow
the slice geometry.
"""
from __future__ import annotations

import numpy as np
import pydicom

from dualct_iodine.inference import save_prediction_as_dicom
from tests._dicom_fixtures import _write_fake_dicom


def _write_descending_series(series_dir, n_slices: int, rows: int = 8, cols: int = 8):
    """InstanceNumber k sits at z = n_slices - k (numbering runs opposite to z)."""
    series_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, n_slices + 1):
        path = series_dir / f"src_{k:04d}.dcm"
        _write_fake_dicom(path, "CT", np.zeros((rows, cols), dtype=np.uint16), instance_number=k)
        ds = pydicom.dcmread(str(path))
        ds.ImagePositionPatient = [0.0, 0.0, float(n_slices - k)]
        ds.SeriesDescription = "SRC"
        pydicom.dcmwrite(str(path), ds, enforce_file_format=True)


def test_exported_slices_keep_source_instance_numbers_and_geometry(tmp_path):
    n = 5
    src = tmp_path / "src"
    _write_descending_series(src, n)
    # Position-sorted volume: slice index z holds the constant value 100 * z.
    vol = np.stack([np.full((8, 8), 100.0 * z, dtype=np.float32) for z in range(n)])

    out = tmp_path / "out"
    save_prediction_as_dicom(vol, src, out, series_desc_suffix="Synth")

    files = sorted(out.glob("*.dcm"))
    assert [f.name for f in files] == [f"IMG_{k:04d}.dcm" for k in range(1, n + 1)]
    for f in files:
        ds = pydicom.dcmread(str(f))
        k = int(ds.InstanceNumber)
        assert f.name == f"IMG_{k:04d}.dcm"
        z_mm = float(ds.ImagePositionPatient[2])
        # Source numbering: InstanceNumber k is the slice at z = n - k ...
        assert z_mm == float(n - k)
        # ... and the pixel content is the volume slice at that position.
        value = float(ds.pixel_array[0, 0]) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert value == 100.0 * (n - k)
        assert ds.SeriesDescription == "SRC_Synth"


def test_export_falls_back_to_volume_index_when_instance_numbers_collide(tmp_path):
    n = 3
    src = tmp_path / "src"
    _write_descending_series(src, n)
    # Give two source slices the same InstanceNumber: numbering is no longer usable.
    for path in sorted(src.glob("*.dcm"))[:2]:
        ds = pydicom.dcmread(str(path))
        ds.InstanceNumber = 7
        pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    vol = np.zeros((n, 8, 8), dtype=np.float32)

    out = tmp_path / "out"
    save_prediction_as_dicom(vol, src, out)

    numbers = sorted(int(pydicom.dcmread(str(f)).InstanceNumber) for f in out.glob("*.dcm"))
    assert numbers == [1, 2, 3]
