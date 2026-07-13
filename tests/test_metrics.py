# -*- coding: utf-8 -*-
"""Direct tests for metrics.evaluate() and its building blocks: the real (measured,
not extrapolated) per-case inference timing (infer_time_s / infer_time_s_sd), the
mean/SD convention (_mean_sd), and empty-mask-region edge case handling."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dualct_iodine.data import build_val_loader
from dualct_iodine.metrics import _mean_sd, evaluate, mae, psnr, rmse, w1
from dualct_iodine.model import build_model
from tests.test_smoke import _tiny_cfg


def test_evaluate_reports_measured_inference_time(tmp_path, synthetic_root):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "metrics_timing")
    device = "cpu"  # deterministic, no CUDA dependency for this unit test

    model = build_model(cfg).to(device)
    val_loader = build_val_loader(cfg, fold_idx=0)

    result = evaluate(model, val_loader, cfg)

    assert "infer_time_s" in result
    assert "infer_time_s_sd" in result
    assert result["n_cases"] > 0
    # A real, positive, finite measurement -- not zero, not NaN, not an extrapolation.
    assert result["infer_time_s"] > 0
    assert result["infer_time_s"] == result["infer_time_s"]  # not NaN
    assert result["infer_time_s_sd"] >= 0


# --- empty-mask-region edge case: mae/rmse/psnr must agree with w1 -----------
def test_mae_rmse_psnr_return_nan_for_empty_region_like_w1():
    pred = torch.rand(1, 1, 4, 4, 4)
    target = torch.rand(1, 1, 4, 4, 4)
    empty = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)

    assert math.isnan(mae(pred, target, empty))
    assert math.isnan(rmse(pred, target, empty))
    assert math.isnan(psnr(pred, target, empty, data_range=200.0))
    assert math.isnan(w1(pred, target, empty, M=16))


def test_mae_nonempty_region_is_unaffected_by_the_nan_change():
    pred = torch.tensor([0.2, 0.8]).view(1, 1, 1, 1, 2)
    target = torch.tensor([0.0, 1.0]).view(1, 1, 1, 1, 2)
    region = torch.ones(1, 1, 1, 1, 2, dtype=torch.bool)
    assert mae(pred, target, region) == pytest.approx(0.2, abs=1e-6)


# --- _mean_sd: sample SD (ddof=1), NaN-aware -----------------------------
def test_mean_sd_uses_sample_sd_not_population_sd():
    vals = [1.0, 2.0, 3.0, 4.0]
    mean, sd = _mean_sd(vals)
    assert mean == pytest.approx(2.5)
    # ddof=1 (sample SD), matching pandas' default used by engine.run_cv for the
    # fold-level "sd" row -- NOT np.std's default (ddof=0, population SD).
    assert sd == pytest.approx(np.std(vals, ddof=1))
    assert sd != pytest.approx(np.std(vals, ddof=0))


def test_mean_sd_ignores_nan_entries():
    mean, sd = _mean_sd([1.0, 2.0, float("nan"), 3.0])
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(1.0)


def test_mean_sd_sd_is_nan_for_a_single_value():
    """SD is mathematically undefined for n=1; report NaN, not a misleading 0."""
    mean, sd = _mean_sd([5.0])
    assert mean == pytest.approx(5.0)
    assert math.isnan(sd)


def test_mean_sd_all_nan_returns_nan_mean_and_sd():
    mean, sd = _mean_sd([float("nan"), float("nan")])
    assert math.isnan(mean)
    assert math.isnan(sd)


# --- evaluate(): a case with a fully empty mask must not corrupt other cases ----
def test_evaluate_warns_and_excludes_case_with_empty_mask(tmp_path, capsys):
    from tests._dicom_fixtures import _write_fake_dicom, make_synthetic_dataset

    root = tmp_path / "data"
    make_synthetic_dataset(root, ["01", "02"], n_slices=4, rows=8, cols=8)

    # Corrupt patient 02's externally-provided mask to be entirely empty (no lung
    # voxels) -- a data-quality anomaly discover_patients cannot currently catch.
    mask_dir = root / "MASK" / "02"
    for i, f in enumerate(sorted(mask_dir.iterdir()), start=1):
        _write_fake_dicom(f, "OT", np.zeros((8, 8), dtype=np.uint16), i)

    cfg = _tiny_cfg(root, tmp_path, "empty_mask")
    cfg.cv.shuffle = False  # deterministic contiguous split: fold0=["01"], fold1=["02"]
    cfg.validate()

    model = build_model(cfg).to("cpu")
    val_loader = build_val_loader(cfg, fold_idx=1)  # validation set == ["02"] only

    result = evaluate(model, val_loader, cfg)

    captured = capsys.readouterr()
    assert "empty mask region" in captured.out
    assert "02" in captured.out

    # The only case has an empty mask -> in-mask metrics are undefined (NaN), but
    # full-volume metrics (which don't depend on the corrupted mask) stay finite.
    assert math.isnan(result["mae_in"])
    assert math.isnan(result["mae_in_sd"])
    assert result["mae_full"] == result["mae_full"]  # not NaN
    assert result["n_cases"] == 1
