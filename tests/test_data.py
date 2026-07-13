# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest

from dualct_iodine.config import Config
from dualct_iodine.data import (
    build_records,
    discover_patients,
    make_folds,
    num_folds,
    split_for_fold,
    split_for_protocol,
)
from tests._dicom_fixtures import _write_fake_dicom, make_synthetic_dataset, make_synthetic_kvp_dataset


def test_discover_patients_finds_all_matching_patients(synthetic_root):
    root, patient_ids = synthetic_root
    cfg = Config()
    cfg.data.root = str(root)
    found = discover_patients(cfg)
    assert found == sorted(patient_ids)


def test_discover_patients_skips_slice_count_mismatch(tmp_path):
    make_synthetic_dataset(tmp_path, ["01", "02"], n_slices=8, rows=16, cols=16)
    # Corrupt patient 02's mask series by deleting one slice -> count mismatch.
    mask_dir = tmp_path / "MASK" / "02"
    files = sorted(mask_dir.iterdir())
    files[0].unlink()

    cfg = Config()
    cfg.data.root = str(tmp_path)
    found = discover_patients(cfg)
    assert found == ["01"]


def test_discover_patients_skips_when_instance_numbers_differ_despite_equal_count(tmp_path):
    """Regression: equal slice *counts* are not sufficient -- two series can each be
    independently missing a different slice and still end up with the same total
    count. Without checking the actual InstanceNumber sets, this would silently
    pair up mismatched anatomy voxel-for-voxel during training/evaluation."""
    make_synthetic_dataset(tmp_path, ["01"], n_slices=8, rows=16, cols=16)

    input_dir = tmp_path / "120kV_Iodinemap" / "120 kVp" / "02"
    target_dir = tmp_path / "120kV_Iodinemap" / "iodinemaps" / "02"
    mask_dir = tmp_path / "MASK" / "02"
    for d in (input_dir, target_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(1)
    for inst in range(1, 9):  # input: InstanceNumbers 1..8
        arr = rng.randint(0, 2000, size=(16, 16)).astype(np.uint16)
        _write_fake_dicom(input_dir / f"{inst:04d}", "CT", arr, inst, rescale_slope=1.0, rescale_intercept=-1024.0)
    for inst in range(2, 10):  # target: InstanceNumbers 2..9 -- same count (8), shifted by one
        arr = rng.randint(0, 2000, size=(16, 16)).astype(np.uint16)
        _write_fake_dicom(target_dir / f"{inst:04d}", "CT", arr, inst, rescale_slope=1.0, rescale_intercept=-1024.0)
    for inst in range(1, 9):  # mask: InstanceNumbers 1..8, matches input
        arr = (rng.rand(16, 16) > 0.5).astype(np.uint16)
        _write_fake_dicom(mask_dir / f"{inst:04d}", "OT", arr, inst)

    cfg = Config()
    cfg.data.root = str(tmp_path)
    found = discover_patients(cfg)
    assert found == ["01"]  # "02" skipped: counts match (8/8/8) but InstanceNumbers don't


def test_strict_geometry_rejects_spatially_misaligned_mask(tmp_path):
    """Regression (M1): under geometry.strict, a mask series whose spatial positions do
    not match the input/target (a misregistered or flipped mask) must be rejected, even
    though its slice count, in-plane shape, and InstanceNumber set all match -- so it
    passes discover_patients but fails at load time rather than silently corrupting the
    in-mask metrics/loss."""
    import pydicom

    from dualct_iodine.transforms import build_val_transforms

    make_synthetic_dataset(tmp_path, ["01"], n_slices=6, rows=8, cols=8)

    # Offset the mask's ImagePositionPatient by 100 mm in z (a misregistered mask),
    # leaving slice count, shape, and InstanceNumbers identical to the CT series.
    mask_dir = tmp_path / "MASK" / "01"
    for inst, f in enumerate(sorted(mask_dir.iterdir()), start=1):
        _write_fake_dicom(f, "OT", np.ones((8, 8), dtype=np.uint16), inst)
        ds = pydicom.dcmread(str(f))
        ds.ImagePositionPatient = [0.0, 0.0, float(inst - 1) + 100.0]
        pydicom.dcmwrite(str(f), ds, enforce_file_format=True)

    cfg = Config()
    cfg.data.root = str(tmp_path)  # geometry.strict is True by default
    records = build_records(cfg, ["01"])
    transform = build_val_transforms(cfg)
    with pytest.raises(Exception) as excinfo:
        transform(records[0])
    # MONAI's Compose re-raises transform errors as RuntimeError; the original
    # geometry ValueError is preserved in the exception cause chain.
    messages, err = [], excinfo.value
    while err is not None:
        messages.append(str(err))
        err = err.__cause__
    assert any("geometry mismatch" in m for m in messages), messages


def test_discover_patients_raises_when_series_dir_missing(tmp_path):
    make_synthetic_dataset(tmp_path, ["01"], n_slices=4, rows=8, cols=8)
    cfg = Config()
    cfg.data.root = str(tmp_path)
    cfg.data.mask_subdir = "DOES_NOT_EXIST"
    with pytest.raises(FileNotFoundError):
        discover_patients(cfg)


def test_discover_patients_body_threshold_needs_no_mask(tmp_path):
    """kVp task (mask_source=body_threshold) discovers patients from input+target only."""
    make_synthetic_kvp_dataset(tmp_path, ["01", "02"], n_slices=4, rows=8, cols=8)
    cfg = Config()
    cfg.data.root = str(tmp_path)
    cfg.data.input_subdir = "120kV"
    cfg.data.target_subdir = "80kV"
    cfg.task.mask_source = "body_threshold"
    found = discover_patients(cfg)
    assert found == ["01", "02"]

    # build_records must not include a mask path in body_threshold mode
    records = build_records(cfg, found)
    assert all("mask" not in r for r in records)
    assert all(set(r.keys()) == {"pid", "image", "target"} for r in records)


def test_build_records_shape(synthetic_root):
    root, patient_ids = synthetic_root
    cfg = Config()
    cfg.data.root = str(root)
    records = build_records(cfg, sorted(patient_ids))
    assert len(records) == len(patient_ids)
    for rec in records:
        assert set(rec.keys()) == {"pid", "image", "target", "mask"}
        assert Path(rec["image"]).exists()
        assert Path(rec["target"]).exists()
        assert Path(rec["mask"]).exists()


def test_make_folds_is_deterministic_and_covers_all_ids():
    ids = [f"{i:02d}" for i in range(1, 51)]
    folds_a = make_folds(ids, n_folds=5, seed=42)
    folds_b = make_folds(ids, n_folds=5, seed=42)
    assert folds_a == folds_b

    assert len(folds_a) == 5
    all_val_ids = [pid for fold in folds_a for pid in fold]
    assert sorted(all_val_ids) == ids  # every id appears exactly once across folds

    sizes = sorted(len(f) for f in folds_a)
    assert sizes == [10, 10, 10, 10, 10]


def test_make_folds_different_seed_gives_different_split():
    ids = [f"{i:02d}" for i in range(1, 51)]
    folds_a = make_folds(ids, n_folds=5, seed=42)
    folds_b = make_folds(ids, n_folds=5, seed=1)
    assert folds_a != folds_b


def test_make_folds_no_shuffle_gives_contiguous_blocks():
    ids = [f"{i:02d}" for i in range(1, 51)]  # 01..50
    folds = make_folds(ids, n_folds=5, shuffle=False)
    # contiguous blocks 01-10, 11-20, ... == the DE groups
    assert folds[0] == [f"{i:02d}" for i in range(1, 11)]
    assert folds[4] == [f"{i:02d}" for i in range(41, 51)]
    # seed is irrelevant when shuffle is off
    assert make_folds(ids, n_folds=5, shuffle=False, seed=7) == folds


def test_split_for_fold_partitions_records():
    ids = [f"{i:02d}" for i in range(1, 11)]
    records = [{"pid": pid} for pid in ids]
    folds = make_folds(ids, n_folds=5, seed=42)
    train, val = split_for_fold(records, folds, fold_idx=0)
    assert len(train) + len(val) == len(records)
    assert set(r["pid"] for r in val) == set(folds[0])
    assert set(r["pid"] for r in train).isdisjoint(set(folds[0]))


def test_three_way_protocol_is_disjoint_and_uses_next_fold_for_validation():
    ids = [f"{i:02d}" for i in range(1, 11)]
    records = [{"pid": pid} for pid in ids]
    folds = make_folds(ids, n_folds=5, shuffle=False)
    train, val, test = split_for_protocol(records, folds, fold_idx=2, protocol="train_val_test")
    assert {r["pid"] for r in test} == set(folds[2])
    assert {r["pid"] for r in val} == set(folds[3])
    assert len(train) + len(val) + len(test) == len(records)


def test_paper_two_way_has_no_validation_records():
    ids = [f"{i:02d}" for i in range(1, 11)]
    records = [{"pid": pid} for pid in ids]
    folds = make_folds(ids, n_folds=5, shuffle=False)
    train, val, test = split_for_protocol(records, folds, fold_idx=0, protocol="paper_two_way")
    assert val == []
    assert len(train) + len(test) == len(records)


# --- nested DE-group (kVp) layout -------------------------------------------
def test_make_folds_group_folds_leave_one_group_out():
    ids = [f"DE{g}/{p}" for g in range(1, 6) for p in range(1, 11)]  # 5 groups x 10
    folds = make_folds(ids, group_folds=True)
    assert len(folds) == 5
    # each fold is exactly one group's patients
    for i, g in enumerate([f"DE{k}" for k in range(1, 6)]):
        assert all(pid.startswith(g + "/") for pid in folds[i])
        assert len(folds[i]) == 10
    # every id appears exactly once
    assert sorted(p for f in folds for p in f) == sorted(ids)


def test_discover_patients_nested_composite_ids(synthetic_nested_kvp_root):
    root, groups = synthetic_nested_kvp_root
    cfg = Config()
    cfg.data.root = str(root)
    cfg.data.input_subdir = "120kV"
    cfg.data.target_subdir = "80kV"
    cfg.data.nested_groups = True
    cfg.task.mask_source = "body_threshold"

    found = discover_patients(cfg)
    # 3 groups x 2 patients, composite ids "<group>/<patient>"
    assert set(found) == {f"{g}/{p}" for g in groups for p in ("1", "2")}

    records = build_records(cfg, found)
    rec = {r["pid"]: r for r in records}["DE1/1"]
    assert rec["image"].replace("\\", "/").endswith("DE1/120kV/1")
    assert rec["target"].replace("\\", "/").endswith("DE1/80kV/1")
    assert "mask" not in rec  # body_threshold


def test_num_folds_group_mode(synthetic_nested_kvp_root):
    root, groups = synthetic_nested_kvp_root
    cfg = Config()
    cfg.data.root = str(root)
    cfg.data.input_subdir = "120kV"
    cfg.data.target_subdir = "80kV"
    cfg.data.nested_groups = True
    cfg.task.mask_source = "body_threshold"
    cfg.cv.group_folds = True
    assert num_folds(cfg) == len(groups)
