# -*- coding: utf-8 -*-
"""End-to-end CLI tests: exercises argparse wiring via `dualct_iodine.cli.main`,
not just the underlying functions directly (as the other smoke tests do).
"""
from __future__ import annotations

from pathlib import Path

from dualct_iodine.cli import main
from tests.test_smoke import _tiny_cfg


def test_cli_train_and_predict_dir(tmp_path, synthetic_root):
    root, patient_ids = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "cli")
    cfg_path = tmp_path / "cli_cfg.yaml"
    cfg.to_yaml(cfg_path)

    main(["train", "--config", str(cfg_path)])

    ckpt = Path(cfg.output.ckpt_dir) / "train_val_test" / "fold0" / "best.safetensors"
    assert ckpt.exists()

    input_dir = root / "120kV_Iodinemap" / "120 kVp"
    out_dir = tmp_path / "cli_predict_dir_out"
    main(
        [
            "predict-dir",
            "--config",
            str(cfg_path),
            "--ckpt",
            str(ckpt),
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--mask-dir",
            str(root / "MASK"),
        ]
    )

    all_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    pred_dirs = [d for d in all_dirs if d.name.endswith(f"_{cfg.output.pred_suffix}")]
    mask_dirs = [d for d in all_dirs if d.name.endswith(f"_{cfg.output.mask_suffix}")]
    assert len(all_dirs) == len(patient_ids)
    assert len(pred_dirs) == len(patient_ids)
    assert len(mask_dirs) == 0
    assert all(len(list(d.glob("*.dcm"))) > 0 for d in all_dirs)

    resolved = Path(cfg.output.metrics_dir) / "resolved_config_predict_dir_fold0.yaml"
    assert resolved.exists()


def test_cli_eval_requires_ckpt(tmp_path, synthetic_root):
    import pytest

    root, _ = synthetic_root
    cfg = _tiny_cfg(root, tmp_path, "cli_eval_noargs")
    cfg_path = tmp_path / "cli_cfg.yaml"
    cfg.to_yaml(cfg_path)

    with pytest.raises(SystemExit):
        main(["eval", "--config", str(cfg_path)])
