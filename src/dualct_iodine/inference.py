# -*- coding: utf-8 -*-
"""Checkpoint loading, sliding-window inference, and DICOM export.

Outside the predicted mask, the output is filled per `infer.outside_fill`: "zero"
(physical `target_min`; the iodine task's non-lung tissue is exactly 0 in the
vendor's own convention) or "input" (restores the input CT value, for the
same-domain kVp task).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .checkpointing import load_weights
from .config import Config
from .engine import amp_settings, resolve_device
from .transforms import denorm_target, list_sorted_dicom_paths


def load_checkpoint(model: torch.nn.Module, ckpt_path, cfg: Config, *, strict: bool = True) -> None:
    """Strictly load weights after validating their model/task signature."""
    load_weights(model, ckpt_path, cfg, strict=strict)
    print(f"[load_checkpoint] loaded: {ckpt_path}")


@torch.inference_mode()
def predict_volume(model: torch.nn.Module, x01: torch.Tensor, mask01: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Sliding-window inference. Outside the mask the output is filled per
    infer.outside_fill: "zero" (== physical target_min; iodine task) or "input"
    (restore the input value; kVp->kVp same-domain task). Returns a [B,1,D,H,W]
    tensor in [0,1] space, on the model's device.
    """
    device = next(model.parameters()).device
    model.eval()

    x01 = x01.to(device, non_blocking=True)
    mask = (mask01 > 0.5).to(dtype=x01.dtype, device=device, non_blocking=True)

    amp_enabled, amp_dtype = amp_settings(cfg, device)
    roi = tuple(cfg.train.roi_size)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        pred01 = sliding_window_inference(
            inputs=x01,
            roi_size=roi,
            sw_batch_size=cfg.infer.sw_batch_size,
            predictor=lambda t: model(t, inference=True),
            overlap=cfg.infer.sw_overlap,
            mode=cfg.infer.sw_mode,
            padding_mode="replicate",
        )
    pred01 = torch.clamp(pred01, 0.0, 1.0)
    if cfg.infer.outside_fill == "input":
        return mask * pred01 + (1.0 - mask) * x01
    return mask * pred01


def _write_slice_as_new_series(
    ds,
    stored_arr: np.ndarray,
    z: int,
    out_dir: Path,
    *,
    modality: str,
    series_desc_suffix: str,
    new_series_uid: str,
    rescale_slope: float,
    rescale_intercept: float,
    pixel_representation: int,
) -> None:
    """Stamp `ds` (a template slice read from the source series) as one slice of a
    new derived series and write it to `out_dir`. Shared by save_prediction_as_dicom
    and save_mask_as_dicom -- only how `stored_arr` and the rescale/representation
    values are computed differs between the two.
    """
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    h, w = stored_arr.shape
    ds.Modality = modality
    ds.Rows, ds.Columns = int(h), int(w)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = pixel_representation
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept

    ds.SeriesInstanceUID = new_series_uid
    base_desc = getattr(ds, "SeriesDescription", "Series")
    ds.SeriesDescription = f"{base_desc}_{series_desc_suffix}"
    ds.SOPInstanceUID = generate_uid(prefix=None)
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.DerivationDescription = "AI-generated image; not for primary diagnosis"
    ds.InstanceNumber = int(z + 1)

    if not getattr(ds, "file_meta", None):
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    ds.PixelData = stored_arr.tobytes(order="C")
    out_path = out_dir / f"IMG_{z + 1:04d}.dcm"
    pydicom.dcmwrite(str(out_path), ds, enforce_file_format=True)


def save_prediction_as_dicom(
    pred_iodine: np.ndarray,
    src_series_dir,
    out_dir,
    *,
    series_desc_suffix: str = "SynthIodine",
) -> None:
    """Write a predicted iodine volume [D,H,W] (physical units) as a new DICOM
    series, reusing the header/geometry/Rescale of `src_series_dir` (the
    patient's original iodine-map series, so RescaleSlope/Intercept match).
    """
    import pydicom
    from pydicom.uid import generate_uid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = list_sorted_dicom_paths(src_series_dir, require_modality="CT")
    D, H, W = pred_iodine.shape
    if len(src_files) != D:
        raise ValueError(
            f"Source series has {len(src_files)} slices but prediction has {D} for {src_series_dir}"
        )

    new_series_uid = generate_uid(prefix=None)
    for z in range(D):
        ds = pydicom.dcmread(src_files[z])

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        stored = np.round((pred_iodine[z] - intercept) / slope).astype(np.int32)
        stored = np.clip(stored, -32768, 32767).astype(np.int16)

        _write_slice_as_new_series(
            ds,
            stored,
            z,
            out_dir,
            modality="CT",
            series_desc_suffix=series_desc_suffix,
            new_series_uid=new_series_uid,
            rescale_slope=slope,
            rescale_intercept=intercept,
            pixel_representation=1,
        )


def save_mask_as_dicom(
    mask_bool: np.ndarray,
    src_series_dir,
    out_dir,
    *,
    series_desc_suffix: str = "Mask",
) -> None:
    """Refuse the legacy pseudo-OT mask export.

    A binary medical segmentation must be encoded as DICOM SEG with source-image
    references. Retaining a CT header while changing Modality to OT is invalid.
    """
    raise NotImplementedError("Mask DICOM export is disabled until DICOM SEG support is implemented")


def _load_model_and_test_loader(cfg: Config, fold_idx: int, ckpt_path):
    from .data import build_test_loader
    from .model import build_model

    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, cfg)
    test_loader = build_test_loader(cfg, fold_idx)
    return model, test_loader


def eval_fold(cfg: Config, fold_idx: int, ckpt_path) -> Dict[str, float]:
    """Load a checkpoint and compute in-mask/full metrics on the fold's held-out test set."""
    from .metrics import evaluate

    model, test_loader = _load_model_and_test_loader(cfg, fold_idx, ckpt_path)
    return evaluate(model, test_loader, cfg)


def _list_case_dirs(input_dir) -> list:
    """Split an input path into case directories: if it has subdirectories, each
    subdirectory is one case (patient); otherwise the path itself is a single series."""
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"input directory not found: {p}")
    subdirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
    return subdirs if subdirs else [p]


def _load_single_case(cfg: Config, case_dir: Path, mask_dir):
    """Load one input series (+ mask) and return normalized (x01, mask01) with a
    leading batch dim, i.e. tensors of shape [1, 1, D, H, W]."""
    from .transforms import BodyMaskd, LoadCTSeriesd, LoadMaskSeriesd, NormalizeInputd

    use_external_mask = cfg.task.mask_source != "body_threshold" and mask_dir is not None
    export_key = "_ref_geometry" if (use_external_mask and cfg.geometry.strict) else None

    data = {"image": str(case_dir)}
    steps = [
        LoadCTSeriesd(
            keys=["image"],
            require_modality="CT",
            strict_geometry=cfg.geometry.strict,
            geometry_atol=cfg.geometry.atol,
            export_geometry_key=export_key,
        )
    ]

    if cfg.task.mask_source == "body_threshold":
        steps.append(BodyMaskd(keys=("image",), out_key="mask", thr_hu=cfg.task.body_thr_hu))
    elif mask_dir is not None:
        data["mask"] = str(mask_dir)
        steps.append(
            LoadMaskSeriesd(
                keys=["mask"],
                ref_key="image",
                require_modality="OT",
                strict_geometry=cfg.geometry.strict,
                geometry_atol=cfg.geometry.atol,
            )
        )
    else:
        raise ValueError(
            "This task requires an external mask. Pass --mask-dir containing "
            "TotalSegmentator-derived masks aligned to each input CT series."
        )

    steps.append(NormalizeInputd(keys=["image"], hu_min=cfg.normalize.input_hu_min, hu_max=cfg.normalize.input_hu_max))

    from monai.transforms import Compose

    out = Compose(steps)(data)
    x = torch.as_tensor(np.asarray(out["image"]), dtype=torch.float32)[None]  # [1,1,D,H,W]
    if "mask" in out:
        m = torch.as_tensor(np.asarray(out["mask"]), dtype=torch.float32)[None]
    else:
        m = torch.ones_like(x)
    return x, m


@torch.inference_mode()
def predict_directory(cfg: Config, input_dir, ckpt_path, out_dir, mask_dir=None) -> list:
    """Standalone inference on an arbitrary input folder (outside any CV split).

    `input_dir` may be a single series folder or a parent containing one subfolder per
    patient. For the kVp task the body mask is auto-generated from the input CT; for the
    iodine task pass `mask_dir` (a folder with the same patient subfolders) to restrict the
    output, otherwise the whole volume is predicted. Saves one synthesized DICOM series per
    case under `out_dir` and returns the list of output directories.
    """
    from .model import build_model

    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, cfg)
    model.eval()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    suffix = cfg.output.pred_suffix

    saved = []
    case_dirs = _list_case_dirs(input_dir)
    single_series = len(case_dirs) == 1 and case_dirs[0].resolve() == Path(input_dir).resolve()
    for case_dir in case_dirs:
        if mask_dir is None:
            case_mask_dir = None
        elif single_series:
            case_mask_dir = Path(mask_dir)
        else:
            case_mask_dir = Path(mask_dir) / case_dir.name
        x01, mask01 = _load_single_case(cfg, case_dir, case_mask_dir)
        pred01 = predict_volume(model, x01, mask01, cfg)
        pred_phys = denorm_target(pred01, cfg)
        pred_phys = torch.clamp(pred_phys, cfg.normalize.target_min, cfg.normalize.target_max)
        vol = pred_phys[0, 0].detach().cpu().numpy().astype(np.float32)
        case_out = out_root / f"{case_dir.name}_{suffix}"
        print(f"[predict_directory] saving {case_out}")
        save_prediction_as_dicom(vol, case_dir, case_out, series_desc_suffix=suffix)

        if cfg.output.save_mask:
            mask_vol = (mask01[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            mask_out = out_root / f"{case_dir.name}_{cfg.output.mask_suffix}"
            save_mask_as_dicom(mask_vol, case_dir, mask_out, series_desc_suffix=cfg.output.mask_suffix)

        saved.append(case_out)
    return saved


def predict_fold(cfg: Config, fold_idx: int, ckpt_path, save_dicom: bool = True) -> Dict[str, float]:
    """Predict the held-out test fold and evaluate the reloaded DICOM output.

    When ``save_dicom`` is false this falls back to in-memory evaluation. Formal
    image-export evaluation should keep it true so DICOM quantization is included.
    """
    from .metrics import _mean_sd, evaluate, mse, psnr, ssim_volume
    from .transforms import LoadCTSeriesd

    model, test_loader = _load_model_and_test_loader(cfg, fold_idx, ckpt_path)
    records = test_loader.dataset.data  # same order as iteration (shuffle=False)

    if save_dicom:
        per_case = {name: [] for name in ("mse_mask", "psnr_mask", "ssim_mask", "mse_full", "psnr_full", "ssim_full")}
        data_range = float(cfg.normalize.target_max - cfg.normalize.target_min)
        pred_root = Path(cfg.output.pred_dir) / f"fold{fold_idx}"
        pred_root.mkdir(parents=True, exist_ok=True)
        for rec, batch in zip(records, test_loader, strict=True):
            x01, mask01 = batch["image"], batch["mask"]
            pred01 = predict_volume(model, x01, mask01, cfg)
            pred_phys = denorm_target(pred01, cfg)
            pred_phys = torch.clamp(pred_phys, cfg.normalize.target_min, cfg.normalize.target_max)
            vol = pred_phys[0, 0].detach().cpu().numpy().astype(np.float32)
            suffix = cfg.output.pred_suffix
            out_dir = pred_root / f"{rec['pid']}_{suffix}"
            print(f"[predict_fold] saving {out_dir}")
            save_prediction_as_dicom(vol, rec["image"], out_dir, series_desc_suffix=suffix)

            reloaded = LoadCTSeriesd(
                keys=["prediction"],
                require_modality="CT",
                strict_geometry=cfg.geometry.strict,
                geometry_atol=cfg.geometry.atol,
            )({"prediction": str(out_dir)})["prediction"]
            pred_saved = torch.as_tensor(reloaded, dtype=torch.float32)[None]
            target_phys = denorm_target(batch["target"], cfg)
            region = batch["mask"] > 0.5
            full_region = torch.ones_like(region, dtype=torch.bool)
            per_case["mse_mask"].append(mse(pred_saved, target_phys, region))
            per_case["psnr_mask"].append(psnr(pred_saved, target_phys, region, data_range))
            per_case["ssim_mask"].append(ssim_volume(pred_saved, target_phys, region, data_range))
            per_case["mse_full"].append(mse(pred_saved, target_phys, full_region))
            per_case["psnr_full"].append(psnr(pred_saved, target_phys, full_region, data_range))
            per_case["ssim_full"].append(ssim_volume(pred_saved, target_phys, None, data_range))

            if cfg.output.save_mask:
                mask_vol = (mask01[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                mask_out = pred_root / f"{rec['pid']}_{cfg.output.mask_suffix}"
                save_mask_as_dicom(mask_vol, rec["image"], mask_out, series_desc_suffix=cfg.output.mask_suffix)

        result: Dict[str, float] = {"n_cases": float(len(records))}
        for name, values in per_case.items():
            result[name], result[f"{name}_sd"] = _mean_sd(values)
        return result
    return evaluate(model, test_loader, cfg)
