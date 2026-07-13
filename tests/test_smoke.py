# -*- coding: utf-8 -*-
"""End-to-end smoke tests on a tiny synthetic dataset (CPU-friendly).

Exercises the full transforms -> loaders -> model -> loss -> checkpoint path:
all 4 loss region/aux combinations, and both residual on/off, must be able to
train, for both the iodine task (flat layout) and the kVp task (nested
DE-group layout, leave-one-group-out CV).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dualct_iodine.config import Config
from dualct_iodine.engine import train_one_fold


def _tiny_cfg(root, tmp_path, tag: str) -> Config:
    cfg = Config()
    cfg.data.root = str(root)
    cfg.model.feature_size = 12  # small network for a fast CPU smoke test
    cfg.train.roi_size = [32, 64, 64]  # divisible by 32 (SwinUNETR requirement), smaller than default
    cfg.train.batch_size = 1
    cfg.train.num_epochs = 1
    cfg.train.validate_every = 1
    cfg.train.num_samples_per_volume = 1
    cfg.train.num_workers = 0
    cfg.train.amp = False  # CPU
    cfg.cv.n_folds = 3
    cfg.cv.fold = 0
    cfg.output.ckpt_dir = str(tmp_path / f"ckpt_{tag}")
    cfg.output.metrics_dir = str(tmp_path / f"metrics_{tag}")
    cfg.validate()
    return cfg


def test_train_one_fold_smoke(tmp_path, synthetic_root):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "default")

    metrics = train_one_fold(cfg, fold_idx=0)

    assert "mse_mask" in metrics
    assert "mse_full" in metrics
    ckpt_last = tmp_path / "ckpt_default" / "train_val_test" / "fold0" / "last.ckpt"
    assert ckpt_last.exists()


@pytest.mark.parametrize("region,aux", [("mask", "none"), ("mask", "cdf"), ("full", "none"), ("full", "cdf")])
def test_loss_region_and_aux_combinations_train(tmp_path, synthetic_root, region, aux):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, f"{region}_{aux}")
    cfg.loss.region = region
    cfg.loss.aux = aux

    metrics = train_one_fold(cfg, fold_idx=0)
    assert "mse_mask" in metrics


@pytest.mark.parametrize("residual", [False])
def test_residual_on_and_off_train(tmp_path, synthetic_root, residual):
    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, f"residual_{residual}")
    cfg.model.residual = residual

    metrics = train_one_fold(cfg, fold_idx=0)
    assert "mse_mask" in metrics


def test_predict_and_eval_work_when_batch_size_exceeds_train_split(tmp_path, synthetic_root):
    """Regression: predict/eval build only the val loader, so a large train.batch_size
    (relative to the tiny training split) must not trip the empty-train-loader guard."""
    from dualct_iodine.inference import eval_fold, predict_fold

    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "predict")
    cfg.output.pred_dir = str(tmp_path / "pred_out")
    train_one_fold(cfg, fold_idx=0)  # trains with tiny batch_size from _tiny_cfg

    # Now bump batch_size well above the training-split size; eval/predict must still work.
    cfg.train.batch_size = 4
    ckpt = tmp_path / "ckpt_predict" / "train_val_test" / "fold0" / "best.safetensors"

    metrics = eval_fold(cfg, 0, ckpt)
    assert "mse_mask" in metrics

    predict_fold(cfg, 0, ckpt, save_dicom=True)
    saved = list((tmp_path / "pred_out" / "fold0").glob("*/"))
    assert len(saved) >= 1

    mask_dirs = list((tmp_path / "pred_out" / "fold0").glob(f"*_{cfg.output.mask_suffix}"))
    assert mask_dirs == []
    assert all(len(list(d.glob("*.dcm"))) > 0 for d in saved)


def test_output_save_mask_false_skips_mask_export(tmp_path, synthetic_root):
    from dualct_iodine.inference import predict_fold

    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "nomask")
    cfg.output.save_mask = False
    train_one_fold(cfg, fold_idx=0)
    ckpt = tmp_path / "ckpt_nomask" / "train_val_test" / "fold0" / "best.safetensors"

    predict_fold(cfg, 0, ckpt, save_dicom=True)
    pred_root = Path(cfg.output.pred_dir) / "fold0"
    mask_dirs = list(pred_root.glob(f"*_{cfg.output.mask_suffix}"))
    assert len(mask_dirs) == 0
    pred_dirs = list(pred_root.glob(f"*_{cfg.output.pred_suffix}"))
    assert len(pred_dirs) >= 1


def test_kvp_task_body_threshold_train(tmp_path, synthetic_kvp_root):
    """kVp->kVp task: flat layout, auto body mask, same-domain target norm, residual on."""
    root, _ = synthetic_kvp_root
    cfg = _tiny_cfg(root, tmp_path, "kvp")
    # Task-specific settings (mirrors configs/kvp.yaml).
    cfg.data.input_subdir = "120kV"
    cfg.data.target_subdir = "80kV"
    cfg.task.mask_source = "body_threshold"
    cfg.task.body_thr_hu = -600
    cfg.normalize.target_min = -1024
    cfg.normalize.target_max = 3071
    cfg.model.residual = True
    cfg.infer.outside_fill = "input"
    cfg.validate()

    metrics = train_one_fold(cfg, fold_idx=0)
    assert "mse_mask" in metrics
    assert "mse_full" in metrics
    assert (tmp_path / "ckpt_kvp" / "train_val_test" / "fold0" / "last.ckpt").exists()


def _kvp_nested_cfg(root, tmp_path, tag: str) -> Config:
    cfg = _tiny_cfg(root, tmp_path, tag)
    cfg.data.input_subdir = "120kV"
    cfg.data.target_subdir = "80kV"
    cfg.data.nested_groups = True
    cfg.task.mask_source = "body_threshold"
    cfg.normalize.target_min = -1024
    cfg.normalize.target_max = 3071
    cfg.model.residual = True
    cfg.infer.outside_fill = "input"
    cfg.cv.group_folds = True
    cfg.validate()
    return cfg


def test_kvp_nested_group_fold_train(tmp_path, synthetic_nested_kvp_root):
    """kVp nested DE-group layout + leave-one-group-out CV: fold 0 == first group."""
    root, groups = synthetic_nested_kvp_root
    cfg = _kvp_nested_cfg(root, tmp_path, "kvpnest")
    metrics = train_one_fold(cfg, fold_idx=0)
    assert "mse_mask" in metrics
    assert (tmp_path / "ckpt_kvpnest" / "train_val_test" / "fold0" / "last.ckpt").exists()


def test_predict_directory_standalone(tmp_path, synthetic_nested_kvp_root):
    """Standalone inference on an arbitrary folder (a parent of per-patient folders)."""
    from dualct_iodine.inference import predict_directory

    root, groups = synthetic_nested_kvp_root
    cfg = _kvp_nested_cfg(root, tmp_path, "pdir")
    train_one_fold(cfg, fold_idx=0)
    ckpt = tmp_path / "ckpt_pdir" / "train_val_test" / "fold0" / "best.safetensors"

    # DE2/120kV contains patient folders "1","2" -> two standalone cases.
    input_dir = root / "DE2" / "120kV"
    out_dir = tmp_path / "standalone_out"
    saved = predict_directory(cfg, input_dir, ckpt, out_dir, mask_dir=None)
    assert len(saved) == 2
    assert all(len(list(d.glob("*.dcm"))) > 0 for d in saved)


def test_predict_directory_iodine_with_external_mask_dir(tmp_path, synthetic_root):
    """Iodine task predict-dir with an explicit --mask-dir (mask_source=external)."""
    from dualct_iodine.inference import predict_directory

    root, patient_ids = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "iodine_pdir_mask")
    train_one_fold(cfg, fold_idx=0)
    ckpt = tmp_path / "ckpt_iodine_pdir_mask" / "train_val_test" / "fold0" / "best.safetensors"

    input_dir = root / "120kV_Iodinemap" / "120 kVp"
    mask_dir = root / "MASK"
    out_dir = tmp_path / "standalone_iodine_masked"
    saved = predict_directory(cfg, input_dir, ckpt, out_dir, mask_dir=mask_dir)
    assert len(saved) == len(patient_ids)
    assert all(len(list(d.glob("*.dcm"))) > 0 for d in saved)


def test_predict_directory_iodine_requires_external_mask(tmp_path, synthetic_root):
    """Iodine inference must not silently replace a missing lung mask."""
    from dualct_iodine.inference import predict_directory

    root, patient_ids = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "iodine_pdir_nomask")
    train_one_fold(cfg, fold_idx=0)
    ckpt = tmp_path / "ckpt_iodine_pdir_nomask" / "train_val_test" / "fold0" / "best.safetensors"

    input_dir = root / "120kV_Iodinemap" / "120 kVp"
    out_dir = tmp_path / "standalone_iodine_unmasked"
    with pytest.raises(ValueError, match="requires an external mask"):
        predict_directory(cfg, input_dir, ckpt, out_dir, mask_dir=None)
