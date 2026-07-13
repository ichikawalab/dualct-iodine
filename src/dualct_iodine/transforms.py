# -*- coding: utf-8 -*-
"""DICOM loading, normalization, and MONAI Compose pipelines.

Input and target are normalized independently (`NormalizeInputd` / `NormalizeTargetd`)
since the iodine task's target lives in a different physical range than the input
CT; the kVp task simply configures both to the same HU range.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    MapTransform,
    RandCropByPosNegLabeld,
    SpatialPadd,
)

from .config import Config

EXCLUDED_IMAGE_TYPES = ("LOCALIZER", "SCOUT")


# ----------------------------------------------------------------------------
# Shared DICOM-series helpers
# ----------------------------------------------------------------------------
def _list_all_files_recursively(pathlike) -> List[str]:
    p = Path(pathlike)
    if p.is_file():
        return [str(p)]
    return [str(fp) for fp in p.rglob("*") if fp.is_file()]


def _read_dicom_safe(path: str, require_modality: Optional[str]):
    import pydicom

    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:
        # The series directory may contain non-DICOM files; anything unreadable is
        # skipped by design rather than aborting the whole series load.
        return None
    if not hasattr(ds, "PixelData"):
        return None
    if require_modality and getattr(ds, "Modality", None) != require_modality:
        return None
    imgtype = [str(s).upper() for s in getattr(ds, "ImageType", [])]
    if any(t in imgtype for t in EXCLUDED_IMAGE_TYPES):
        return None
    try:
        _ = ds.pixel_array
    except Exception:
        return None
    return ds


def _slice_position(ds) -> Optional[float]:
    """Project ImagePositionPatient onto the slice normal when available."""
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is not None and iop is not None and len(ipp) >= 3 and len(iop) >= 6:
            row = np.asarray(iop[:3], dtype=np.float64)
            col = np.asarray(iop[3:6], dtype=np.float64)
            return float(np.dot(np.asarray(ipp[:3], dtype=np.float64), np.cross(row, col)))
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
    except Exception:
        pass
    return None


def _sort_key_from_ds(ds) -> Tuple[int, float]:
    position = _slice_position(ds)
    if position is not None:
        return (0, position)
    try:
        if hasattr(ds, "InstanceNumber"):
            return (1, float(ds.InstanceNumber))
    except Exception:
        pass
    try:
        if hasattr(ds, "AcquisitionNumber"):
            return (2, float(ds.AcquisitionNumber))
    except Exception:
        pass
    return (3, 0.0)


def _load_series_records(src, require_modality: Optional[str]):
    cand = _list_all_files_recursively(src)
    records = []
    for p in cand:
        ds = _read_dicom_safe(p, require_modality)
        if ds is None:
            continue
        records.append((p, ds))
    if not records:
        raise FileNotFoundError(
            f"No readable DICOM (modality={require_modality}) with PixelData under: {src}"
        )
    records.sort(key=lambda t: _sort_key_from_ds(t[1]))
    return records


def _most_frequent_shape(records) -> Tuple[int, int]:
    shapes: Dict[Tuple[int, int], int] = {}
    for _, ds in records:
        try:
            rr, cc = int(ds.Rows), int(ds.Columns)
            shapes[(rr, cc)] = shapes.get((rr, cc), 0) + 1
        except Exception:
            pass
    if not shapes:
        raise FileNotFoundError("No valid Rows/Columns found in DICOM series")
    return max(shapes.items(), key=lambda kv: kv[1])[0]


def _stack_rescaled_volume(records, h: int, w: int) -> np.ndarray:
    vol = []
    for _, ds in records:
        try:
            arr = ds.pixel_array.astype(np.float32)
            if arr.shape != (h, w):
                continue
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            vol.append(arr * slope + intercept)
        except Exception:
            continue
    if not vol:
        raise FileNotFoundError("Found DICOMs but none usable (shape mismatch / decode errors)")
    return np.stack(vol, axis=0).astype(np.float32)  # [D, H, W]


def _series_geometry(records) -> dict[str, np.ndarray]:
    first = records[0][1]
    required = ("PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient")
    missing = [name for name in required if not hasattr(first, name)]
    if missing:
        raise ValueError(f"DICOM geometry metadata missing: {', '.join(missing)}")
    positions = []
    for _, ds in records:
        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError("DICOM ImagePositionPatient is missing on one or more slices")
        positions.append(np.asarray(ds.ImagePositionPatient, dtype=np.float64))
    return {
        "pixel_spacing": np.asarray(first.PixelSpacing, dtype=np.float64),
        "orientation": np.asarray(first.ImageOrientationPatient, dtype=np.float64),
        "positions": np.stack(positions),
        "shape": np.asarray([len(records), int(first.Rows), int(first.Columns)]),
    }


def _assert_same_geometry(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], atol: float) -> None:
    for key in reference:
        if reference[key].shape != candidate[key].shape or not np.allclose(
            reference[key], candidate[key], rtol=0.0, atol=atol
        ):
            raise ValueError(f"DICOM geometry mismatch between co-registered series in {key}")


# ----------------------------------------------------------------------------
# MapTransforms
# ----------------------------------------------------------------------------
def list_sorted_dicom_paths(src_dir, require_modality: Optional[str] = "CT") -> List[str]:
    """Return DICOM file paths for a series, sorted the same way as LoadCTSeriesd.

    Used by inference.py to obtain a template series (for header/geometry reuse)
    when writing out predicted volumes as new DICOM series.
    """
    records = _load_series_records(src_dir, require_modality)
    return [p for p, _ in records]


class LoadCTSeriesd(MapTransform):
    """Load a CT-modality DICOM series (extension-less files supported).

    Returns the *physical* value volume (HU, or iodine-equivalent HU for the
    target series) with shape [1, D, H, W]. No normalization is applied here
    because input and target use different normalization ranges.
    """

    def __init__(
        self,
        keys,
        require_modality: Optional[str] = "CT",
        *,
        strict_geometry: bool = False,
        geometry_atol: float = 1e-4,
        export_geometry_key: Optional[str] = None,
    ):
        super().__init__(keys)
        self.require_modality = require_modality
        self.strict_geometry = strict_geometry
        self.geometry_atol = geometry_atol
        # When set (and strict_geometry is on), the first key's DICOM geometry is
        # written to this data key so a later transform (LoadMaskSeriesd) can verify
        # the mask series is spatially co-registered. The consumer must pop it, so it
        # never reaches the DataLoader collate.
        self.export_geometry_key = export_geometry_key

    def __call__(self, data):
        d = dict(data)
        reference_geometry = None
        for k in self.keys:
            records = _load_series_records(d[k], self.require_modality)
            if self.strict_geometry:
                geometry = _series_geometry(records)
                if reference_geometry is None:
                    reference_geometry = geometry
                else:
                    _assert_same_geometry(reference_geometry, geometry, self.geometry_atol)
            h, w = _most_frequent_shape(records)
            vol = _stack_rescaled_volume(records, h, w)  # [D,H,W] physical value
            d[k] = vol[None]  # [1,D,H,W]
        if self.export_geometry_key is not None and reference_geometry is not None:
            d[self.export_geometry_key] = reference_geometry
        return d


class LoadMaskSeriesd(MapTransform):
    """Load the externally-provided binary lung mask series (Modality=OT).

    Validates that the mask series has the same slice count and in-plane shape as the
    reference series (the target/iodine series) that must already be loaded into
    `data[ref_key]` (shape [1,D,H,W]). When `strict_geometry` is on and the reference
    geometry has been exported to `data[ref_geometry_key]` by LoadCTSeriesd, this also
    verifies the mask shares the reference's PixelSpacing, orientation, and per-slice
    positions -- catching a mask that is spatially misaligned (e.g. z-flipped) while
    still having a matching slice count.
    """

    def __init__(
        self,
        keys=("mask",),
        ref_key: str = "target",
        require_modality: Optional[str] = "OT",
        *,
        strict_geometry: bool = False,
        geometry_atol: float = 1e-4,
        ref_geometry_key: str = "_ref_geometry",
    ):
        super().__init__(keys)
        self.ref_key = ref_key
        self.require_modality = require_modality
        self.strict_geometry = strict_geometry
        self.geometry_atol = geometry_atol
        self.ref_geometry_key = ref_geometry_key

    def __call__(self, data):
        d = dict(data)
        ref_shape = d[self.ref_key].shape  # [1,D,H,W]
        # Pop the exported reference geometry so it never reaches the DataLoader collate.
        reference_geometry = d.pop(self.ref_geometry_key, None)
        for k in self.keys:
            records = _load_series_records(d[k], self.require_modality)
            if self.strict_geometry and reference_geometry is not None:
                _assert_same_geometry(reference_geometry, _series_geometry(records), self.geometry_atol)
            h, w = _most_frequent_shape(records)
            vol = _stack_rescaled_volume(records, h, w)  # [D,H,W]
            mask = (vol > 0.5).astype(np.float32)[None]  # [1,D,H,W]
            if mask.shape != ref_shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match reference '{self.ref_key}' shape {ref_shape} "
                    f"for key '{k}' (source: {d[k]})"
                )
            d[k] = mask
        return d


class BodyMaskd(MapTransform):
    """Generate a body mask from the input CT via an HU threshold.

    Used by the kVp->kVp task, where no external mask is provided: threshold at
    ``thr_hu``, keep the largest connected component, then per-slice fill holes
    and binary closing. Operates on the physical-HU input volume (shape
    [1,D,H,W]) produced by LoadCTSeriesd, i.e. before NormalizeInputd. Writes a
    float32 mask [1,D,H,W] to ``out_key``.
    """

    def __init__(
        self,
        keys=("image",),
        out_key: str = "mask",
        thr_hu: float = -600,
        keep_largest_cc: bool = True,
        close2d_iters: int = 2,
    ):
        super().__init__(keys)
        self.out_key = out_key
        self.thr_hu = float(thr_hu)
        self.keep_largest_cc = keep_largest_cc
        self.close2d_iters = int(close2d_iters)

    def __call__(self, data):
        import scipy.ndimage as ndi

        d = dict(data)
        x_hu = np.asarray(d[self.keys[0]])  # [1,D,H,W] physical HU
        vol = x_hu[0]  # [D,H,W]

        m = vol > self.thr_hu
        if self.keep_largest_cc:
            lbl, n = ndi.label(m)
            if n > 0:
                sizes = np.bincount(lbl.ravel())
                sizes[0] = 0
                m = lbl == sizes.argmax()
            else:
                m = np.zeros_like(m, dtype=bool)

        se2 = np.ones((5, 5), dtype=bool)
        for z in range(m.shape[0]):
            m[z] = ndi.binary_fill_holes(m[z])
            if self.close2d_iters > 0:
                m[z] = ndi.binary_closing(m[z], structure=se2, iterations=self.close2d_iters)

        d[self.out_key] = m.astype(np.float32)[None]  # [1,D,H,W]
        return d


class NormalizeInputd(MapTransform):
    """Clip input CT to [hu_min, hu_max] and rescale to [0, 1]."""

    def __init__(self, keys, hu_min: float, hu_max: float):
        super().__init__(keys)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = np.clip(d[k], self.hu_min, self.hu_max)
            d[k] = ((x - self.hu_min) / (self.hu_max - self.hu_min)).astype(np.float32)
        return d


class NormalizeTargetd(MapTransform):
    """Clip iodine target to [vmin, vmax] and rescale to [0, 1].

    Negative values (material-decomposition noise) are clipped to vmin (0 by default).
    """

    def __init__(self, keys, vmin: float, vmax: float):
        super().__init__(keys)
        self.vmin = float(vmin)
        self.vmax = float(vmax)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = np.clip(d[k], self.vmin, self.vmax)
            d[k] = ((x - self.vmin) / (self.vmax - self.vmin)).astype(np.float32)
        return d


class ValidateTripletShaped(MapTransform):
    """Assert that image/target/mask share the same shape (fail fast)."""

    def __init__(self, keys=("image", "target", "mask")):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        shapes = {k: d[k].shape for k in self.keys}
        first = shapes[self.keys[0]]
        for s in shapes.values():
            if s != first:
                raise AssertionError(f"Shape mismatch among {self.keys}: {shapes}")
        return d


# ----------------------------------------------------------------------------
# Denormalization helper (target only; shared by metrics/inference)
# ----------------------------------------------------------------------------
def denorm_target(x01: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Map a normalized [0,1] iodine prediction back to physical (iodine-equivalent HU)."""
    lo, hi = cfg.normalize.target_min, cfg.normalize.target_max
    return x01 * (hi - lo) + lo


# ----------------------------------------------------------------------------
# Compose pipelines
# ----------------------------------------------------------------------------
_REF_GEOMETRY_KEY = "_ref_geometry"


def _mask_transform(cfg: Config) -> MapTransform:
    """Return the transform that produces the 'mask' key, per task.mask_source.

    Both branches must run after LoadCTSeriesd (they read the physical-HU volumes)
    and before NormalizeInputd/NormalizeTargetd.
    """
    if cfg.task.mask_source == "external":
        return LoadMaskSeriesd(
            keys=["mask"],
            ref_key="target",
            require_modality="OT",
            strict_geometry=cfg.geometry.strict,
            geometry_atol=cfg.geometry.atol,
            ref_geometry_key=_REF_GEOMETRY_KEY,
        )
    if cfg.task.mask_source == "body_threshold":
        return BodyMaskd(keys=("image",), out_key="mask", thr_hu=cfg.task.body_thr_hu)
    raise ValueError(f"Unknown task.mask_source: {cfg.task.mask_source!r}")


def _common_head(cfg: Config) -> list:
    """Load + mask + validate + normalize, shared by train and val pipelines."""
    # Only export the reference geometry when an external mask will consume (and pop) it.
    export_key = _REF_GEOMETRY_KEY if (cfg.task.mask_source == "external" and cfg.geometry.strict) else None
    return [
        LoadCTSeriesd(
            keys=["image", "target"],
            require_modality="CT",
            strict_geometry=cfg.geometry.strict,
            geometry_atol=cfg.geometry.atol,
            export_geometry_key=export_key,
        ),
        _mask_transform(cfg),
        ValidateTripletShaped(keys=("image", "target", "mask")),
        NormalizeInputd(keys=["image"], hu_min=cfg.normalize.input_hu_min, hu_max=cfg.normalize.input_hu_max),
        NormalizeTargetd(keys=["target"], vmin=cfg.normalize.target_min, vmax=cfg.normalize.target_max),
    ]


def build_train_transforms(cfg: Config) -> Compose:
    roi = tuple(cfg.train.roi_size)
    return Compose(
        _common_head(cfg)
        + [
            SpatialPadd(keys=["image", "target"], spatial_size=roi, mode="replicate"),
            SpatialPadd(keys=["mask"], spatial_size=roi, mode="constant"),
            RandCropByPosNegLabeld(
                keys=["image", "target", "mask"],
                label_key="mask",
                spatial_size=roi,
                pos=1.0,
                neg=0.0,
                num_samples=cfg.train.num_samples_per_volume,
                image_key="image",
                image_threshold=0.0,
            ),
            EnsureTyped(keys=["image", "target", "mask"], dtype=torch.float32, track_meta=False),
        ]
    )


def build_val_transforms(cfg: Config) -> Compose:
    # No SpatialPadd here: validation/inference runs on whole volumes and
    # sliding_window_inference pads internally for volumes smaller than roi_size,
    # returning a result at the original size. This keeps the predicted volume the
    # same depth as the source DICOM series so it can be written back 1:1.
    return Compose(
        _common_head(cfg)
        + [
            EnsureTyped(keys=["image", "target", "mask"], dtype=torch.float32, track_meta=False),
        ]
    )
