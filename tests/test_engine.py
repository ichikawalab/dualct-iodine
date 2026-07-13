# -*- coding: utf-8 -*-
"""Tests for engine.run_cv: the multi-fold driver and cv_summary.csv output.

train_one_fold itself is already exercised extensively by test_smoke.py; this
file covers the orchestration on top of it (looping over folds, writing the
per-fold + mean/SD summary CSV) which was previously only checked manually.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dualct_iodine.engine import run_cv
from tests.test_smoke import _tiny_cfg


def test_run_cv_writes_summary_csv_with_expected_rows(tmp_path, synthetic_root):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "cv")
    cfg.cv.n_folds = 3

    df = run_cv(cfg)

    assert len(df) == 5
    assert list(df["fold"]) == [0, 1, 2, "mean", "sd"]
    for metric in ("mse_mask", "mse_full", "psnr_mask", "ssim_mask", "train_time_s", "infer_time_s"):
        assert metric in df.columns

    # train_time_s is a real (measured) per-fold duration, not an extrapolation -- it must
    # be a small positive number for these tiny synthetic folds, not zero or NaN.
    fold_rows = df[df["fold"].isin([0, 1, 2])]
    assert (fold_rows["train_time_s"] > 0).all()
    assert (fold_rows["infer_time_s"] >= 0).all()

    out_csv = Path(cfg.output.metrics_dir) / "train_val_test" / "cv_summary.csv"
    assert out_csv.exists()
    on_disk = pd.read_csv(out_csv)
    assert len(on_disk) == 5
    assert list(on_disk.columns) == list(df.columns)


def test_run_cv_checkpoints_are_per_fold(tmp_path, synthetic_root):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "cv2")
    cfg.cv.n_folds = 3

    run_cv(cfg)

    ckpt_root = Path(cfg.output.ckpt_dir)
    assert (ckpt_root / "train_val_test" / "fold0" / "last.ckpt").exists()
    assert (ckpt_root / "train_val_test" / "fold1" / "last.ckpt").exists()
