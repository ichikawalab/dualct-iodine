# -*- coding: utf-8 -*-
import argparse

import pytest

from dualct_iodine.cli import _dedicated_overrides
from dualct_iodine.config import Config, load_config


def test_defaults_match_iodine_yaml():
    cfg = Config.from_yaml("configs/iodine.yaml")
    assert cfg.normalize.input_hu_min == -1024
    assert cfg.normalize.input_hu_max == 3071
    assert cfg.normalize.target_min == 0
    assert cfg.normalize.target_max == 200
    assert cfg.loss.region == "mask"
    assert cfg.loss.aux == "cdf"
    assert cfg.loss.lam_cdf == pytest.approx(0.3)
    assert cfg.model.residual is False
    assert cfg.train.roi_size == [64, 128, 128]
    assert cfg.train.batch_size == 4
    assert cfg.train.num_epochs == 500
    assert cfg.cv.n_folds == 5


def test_apply_overrides_dotted_keys_with_type_casting():
    cfg = Config()
    cfg.apply_overrides(
        [
            "loss.region=full",
            "loss.aux=none",
            "train.num_epochs=5",
            "train.lr=1e-3",
            "train.roi_size=[32,64,64]",
        ]
    )
    assert cfg.loss.region == "full"
    assert cfg.loss.aux == "none"
    assert cfg.train.num_epochs == 5
    assert isinstance(cfg.train.num_epochs, int)
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.model.residual is False
    assert cfg.train.roi_size == [32, 64, 64]


def test_apply_overrides_comma_list_syntax():
    cfg = Config()
    cfg.apply_overrides(["train.roi_size=32,64,64"])
    assert cfg.train.roi_size == [32, 64, 64]


def test_validation_rejects_bad_target_range():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["normalize.target_min=200", "normalize.target_max=100"])


def test_validation_rejects_bad_loss_region():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["loss.region=invalid"])


def test_validation_rejects_bad_loss_aux():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["loss.aux=invalid"])


def test_validation_rejects_non_l1_main_loss():
    """loss.main is not actually read by build_loss (main loss is always L1);
    reject any other value instead of silently ignoring it."""
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["loss.main=l2"])


def test_validation_rejects_unsafe_mask_dicom_export():
    cfg = Config()
    with pytest.raises(ValueError, match="DICOM SEG"):
        cfg.apply_overrides(["output.save_mask=true"])


def test_mask_suffix_collision_is_allowed_when_save_mask_is_false():
    cfg = Config()
    cfg.apply_overrides(["output.save_mask=false", "output.mask_suffix=" + cfg.output.pred_suffix])
    assert cfg.output.save_mask is False


def test_validation_rejects_bad_mask_source():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["task.mask_source=invalid"])


def test_validation_rejects_bad_outside_fill():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["infer.outside_fill=invalid"])


def test_validation_rejects_group_folds_without_nested_groups():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["cv.group_folds=true"])  # data.nested_groups still false


def test_group_folds_allowed_with_nested_groups():
    cfg = Config()
    cfg.apply_overrides(["data.nested_groups=true", "cv.group_folds=true"])
    assert cfg.cv.group_folds is True
    assert cfg.data.nested_groups is True


def test_cv_shuffle_default_and_override():
    cfg = Config.from_yaml("configs/iodine.yaml")
    assert cfg.cv.shuffle is True
    cfg.apply_overrides(["cv.shuffle=false"])
    assert cfg.cv.shuffle is False


def test_kvp_preset_loads_and_overrides_defaults():
    cfg = Config.from_yaml("configs/kvp.yaml")
    assert cfg.task.mask_source == "body_threshold"
    assert cfg.normalize.target_min == -1024
    assert cfg.normalize.target_max == 3071
    assert cfg.model.residual is True
    assert cfg.infer.outside_fill == "input"
    assert cfg.data.nested_groups is True
    assert cfg.cv.group_folds is True
    assert cfg.data.input_subdir == "120kV"
    assert cfg.data.target_subdir == "80kV"
    # Shared training hyperparameters: explicitly listed in both files with the same
    # values (not inherited -- configs/kvp.yaml is a complete, standalone file).
    assert cfg.train.roi_size == [64, 128, 128]
    assert cfg.train.batch_size == 4
    assert cfg.loss.lam_cdf == pytest.approx(0.3)


def _dotted_keys(d: dict, prefix: str = "") -> set:
    """Flatten a nested dict into a set of dotted-path keys (leaves only)."""
    keys = set()
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= _dotted_keys(v, path)
        else:
            keys.add(path)
    return keys


def test_kvp_yaml_is_a_complete_standalone_config_not_a_diff():
    """Regression guard: configs/kvp.yaml must explicitly list every key that
    configs/iodine.yaml has (same key set), so it never silently falls back to
    the dataclass defaults (the iodine task's values) for an omitted key."""
    import yaml

    with open("configs/iodine.yaml", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)
    with open("configs/kvp.yaml", encoding="utf-8") as f:
        kvp_raw = yaml.safe_load(f)

    default_keys = _dotted_keys(default_raw)
    kvp_keys = _dotted_keys(kvp_raw)
    missing_in_kvp = default_keys - kvp_keys
    assert not missing_in_kvp, f"configs/kvp.yaml is missing keys present in iodine.yaml: {missing_in_kvp}"


def test_validation_rejects_bad_roi_size_length():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["train.roi_size=[64,128]"])


def test_validation_rejects_out_of_range_fold():
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["cv.fold=10"])  # default n_folds=5


def test_to_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.apply_overrides(["loss.region=full"])
    out_path = tmp_path / "resolved.yaml"
    cfg.to_yaml(out_path)
    cfg2 = Config.from_yaml(out_path)
    assert cfg2.loss.region == "full"
    assert cfg2.model.residual is False


def test_load_config_helper_applies_set_overrides():
    cfg = load_config("configs/iodine.yaml", ["train.num_epochs=3"])
    assert cfg.train.num_epochs == 3


def test_unknown_top_level_key_raises(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not_a_real_section:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Config.from_yaml(bad_yaml)


def test_cli_dedicated_overrides_translation():
    ns = argparse.Namespace(
        fold=2,
        epochs=10,
        batch_size=2,
        loss_region="full",
        loss_aux="none",
        residual=True,
        data_root="/tmp/data",
    )
    overrides = _dedicated_overrides(ns)
    assert "cv.fold=2" in overrides
    assert "train.num_epochs=10" in overrides
    assert "train.batch_size=2" in overrides
    assert "loss.region=full" in overrides
    assert "loss.aux=none" in overrides
    assert "model.residual=true" in overrides
    assert "data.root=/tmp/data" in overrides


def test_cli_dedicated_overrides_none_when_unset():
    ns = argparse.Namespace(
        fold=None,
        epochs=None,
        batch_size=None,
        loss_region=None,
        loss_aux=None,
        residual=None,
        data_root=None,
    )
    assert _dedicated_overrides(ns) == []
