# -*- coding: utf-8 -*-
"""Synthetic DICOM generation used by the test suite.

Builds tiny, extension-less DICOM series mirroring the real dataset layout
(120kV_Iodinemap/120 kVp, 120kV_Iodinemap/iodinemaps, MASK) so that tests do
not depend on the real ~3.8 GB dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def _write_fake_dicom(
    path: Path,
    modality: str,
    arr_uint16: np.ndarray,
    instance_number: int,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Modality = modality
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number - 1)]
    ds.SliceThickness = 1.0
    ds.Rows, ds.Columns = arr_uint16.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelData = arr_uint16.astype(np.uint16).tobytes()
    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)


def make_synthetic_dataset(
    root: Path,
    patient_ids: List[str],
    n_slices: int = 8,
    rows: int = 16,
    cols: int = 16,
    seed: int = 0,
) -> Path:
    """Create a synthetic dataset mimicking the real directory layout under `root`."""
    rng = np.random.RandomState(seed)
    input_dir = Path(root) / "120kV_Iodinemap" / "120 kVp"
    target_dir = Path(root) / "120kV_Iodinemap" / "iodinemaps"
    mask_dir = Path(root) / "MASK"

    for pid in patient_ids:
        pdir_in = input_dir / pid
        pdir_tg = target_dir / pid
        pdir_mk = mask_dir / pid
        for d in (pdir_in, pdir_tg, pdir_mk):
            d.mkdir(parents=True, exist_ok=True)

        for z in range(n_slices):
            inst = z + 1
            # Input CT: stored 0..2000, slope=1/intercept=-1024 -> HU in [-1024, 976]
            stored_ct = rng.randint(0, 2000, size=(rows, cols)).astype(np.uint16)
            _write_fake_dicom(
                pdir_in / f"{inst:04d}", "CT", stored_ct, inst, rescale_slope=1.0, rescale_intercept=-1024.0
            )

            # Target iodine map: stored so rescaled value falls in roughly [-10, 60]
            stored_io = (rng.randint(-10, 60, size=(rows, cols)).astype(np.int32) + 1024).astype(np.uint16)
            _write_fake_dicom(
                pdir_tg / f"{inst:04d}", "CT", stored_io, inst, rescale_slope=1.0, rescale_intercept=-1024.0
            )

            # Mask: binary 0/1, Modality=OT (Secondary Capture), no rescale offset
            mask_arr = (rng.rand(rows, cols) > 0.5).astype(np.uint16)
            _write_fake_dicom(pdir_mk / f"{inst:04d}", "OT", mask_arr, inst, rescale_slope=1.0, rescale_intercept=0.0)

    return Path(root)


def make_synthetic_kvp_dataset(
    root: Path,
    patient_ids: List[str],
    n_slices: int = 8,
    rows: int = 16,
    cols: int = 16,
    input_dirname: str = "120kV",
    target_dirname: str = "80kV",
    seed: int = 0,
) -> Path:
    """Create a synthetic kVp-task dataset: flat <root>/<kv>/<patient>/ layout, no mask.

    Both series are CT in HU (same domain). A central high-HU block guarantees a
    non-empty body mask when BodyMaskd thresholds at -600 HU.
    """
    rng = np.random.RandomState(seed)
    input_dir = Path(root) / input_dirname
    target_dir = Path(root) / target_dirname

    for pid in patient_ids:
        pdir_in = input_dir / pid
        pdir_tg = target_dir / pid
        for d in (pdir_in, pdir_tg):
            d.mkdir(parents=True, exist_ok=True)

        for z in range(n_slices):
            inst = z + 1
            # Background air (HU ~ -1000) with a central soft-tissue block (HU ~ +30)
            # so the body-threshold mask (>-600 HU) is a well-defined connected region.
            def _ct_slice(offset):
                stored = np.full((rows, cols), 24, dtype=np.int32)  # -1000 HU
                lo_r, hi_r = rows // 4, rows - rows // 4
                lo_c, hi_c = cols // 4, cols - cols // 4
                stored[lo_r:hi_r, lo_c:hi_c] = 1054 + offset  # ~ +30 HU (+offset)
                return stored.astype(np.uint16)

            _write_fake_dicom(pdir_in / f"{inst:04d}", "CT", _ct_slice(0), inst,
                              rescale_slope=1.0, rescale_intercept=-1024.0)
            _write_fake_dicom(pdir_tg / f"{inst:04d}", "CT", _ct_slice(int(rng.randint(-20, 20))), inst,
                              rescale_slope=1.0, rescale_intercept=-1024.0)

    return Path(root)


def make_synthetic_nested_kvp_dataset(
    root: Path,
    groups: List[str],
    patients_per_group: int = 2,
    n_slices: int = 8,
    rows: int = 16,
    cols: int = 16,
    kv_dirs=("120kV", "80kV", "140kV"),
    seed: int = 0,
) -> Path:
    """Create a synthetic nested DE-group kVp dataset:
    <root>/<group>/<kv>/<patient>/  (patient numbers repeat across groups)."""
    for gi, g in enumerate(groups):
        for kv in kv_dirs:
            for p in range(1, patients_per_group + 1):
                make_synthetic_kvp_dataset(  # reuse: writes <root>/<input>/<pid> for a single kv
                    root / g,
                    [str(p)],
                    n_slices=n_slices,
                    rows=rows,
                    cols=cols,
                    input_dirname=kv,
                    target_dirname=kv,  # same folder; only the input_dirname series is used per call
                    seed=seed + gi * 100 + p,
                )
    return Path(root)
