# -*- coding: utf-8 -*-
"""Patient discovery, record building, and deterministic cross-validation splits.

Supports both dataset layouts: a flat patient directory (iodine task) and a nested
DE-group directory (kVp task, `data.nested_groups`). Folds are either a seeded
patient-level shuffle/split or, for the kVp layout, leave-one-DE-group-out
(`cv.group_folds`).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .config import Config
from .transforms import EXCLUDED_IMAGE_TYPES, build_train_transforms, build_val_transforms


def _scan_series(dirpath: Path, modality: str) -> Tuple[int, set]:
    """Header-only scan (no pixel decode) of a series: (usable slice count,
    set of InstanceNumbers among those usable slices).

    A slice without a parseable InstanceNumber is counted but omitted from the
    set, so `len(instance_numbers) < count` is possible; callers should treat
    the set as unusable for cross-series identity checks in that case.
    """
    import pydicom

    p = Path(dirpath)
    if not p.exists():
        return 0, set()
    cnt = 0
    nums = set()
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
        except Exception:
            # rglob yields every file under the series directory, including non-DICOM
            # files; anything pydicom cannot parse as a header is skipped by design.
            continue
        if modality and getattr(ds, "Modality", None) != modality:
            continue
        imgtype = [str(s).upper() for s in getattr(ds, "ImageType", [])]
        if any(t in imgtype for t in EXCLUDED_IMAGE_TYPES):
            continue
        cnt += 1
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                nums.add(int(inst))
            except (TypeError, ValueError):
                pass
    return cnt, nums


def _count_usable_slices(dirpath: Path, modality: str) -> int:
    """Lightweight slice count (header-only, no pixel decode) used for discovery."""
    return _scan_series(dirpath, modality)[0]


def pid_series_dirs(cfg: Config, pid: str) -> Tuple[Path, Path, Path]:
    """Resolve (input_dir, target_dir, mask_dir) for a (possibly composite) patient id.

    Flat layout:   <root>/<subdir>/<pid>
    Nested layout: pid == "<group>/<patient>" -> <root>/<group>/<subdir>/<patient>
    """
    root = Path(cfg.data.root)
    if cfg.data.nested_groups:
        group, patient = pid.split("/", 1)
        base, leaf = root / group, patient
    else:
        base, leaf = root, pid
    return (
        base / cfg.data.input_subdir / leaf,
        base / cfg.data.target_subdir / leaf,
        base / cfg.data.mask_subdir / leaf,
    )


def _enumerate_candidate_pids(cfg: Config) -> List[str]:
    """List candidate patient ids (composite "<group>/<patient>" when nested) present in
    both the input and target series (and the mask series when external)."""
    root = Path(cfg.data.root)
    use_external_mask = cfg.task.mask_source == "external"

    def _subdir_pids(base: Path) -> dict:
        input_dir = base / cfg.data.input_subdir
        target_dir = base / cfg.data.target_subdir
        required = [(input_dir, "input"), (target_dir, "target")]
        if use_external_mask:
            required.append((base / cfg.data.mask_subdir, "mask"))
        for d, label in required:
            if not d.exists():
                raise FileNotFoundError(f"{label} directory not found: {d}")
        sets = {
            "input": {p.name for p in input_dir.iterdir() if p.is_dir()},
            "target": {p.name for p in target_dir.iterdir() if p.is_dir()},
        }
        if use_external_mask:
            sets["mask"] = {p.name for p in (base / cfg.data.mask_subdir).iterdir() if p.is_dir()}
        common = sorted(set.intersection(*sets.values()))
        for pid in sorted(set.union(*sets.values()) - set(common)):
            present = [name for name, s in sets.items() if pid in s]
            print(f"[discover_patients] skip {pid}: only present in {present}")
        return common

    if cfg.data.nested_groups:
        if not root.exists():
            raise FileNotFoundError(f"data root not found: {root}")
        groups = sorted(d.name for d in root.iterdir() if d.is_dir())
        candidates = []
        for g in groups:
            for patient in _subdir_pids(root / g):
                candidates.append(f"{g}/{patient}")
        return candidates
    return _subdir_pids(root)


def discover_patients(cfg: Config) -> List[str]:
    """Return sorted patient IDs present in every required series with matching slice counts.

    IDs are composite "<group>/<patient>" for the nested (kVp DE-group) layout. The mask
    series is required only when task.mask_source == "external"; for "body_threshold" the
    mask is generated from the input CT, so only the input and target series must match.

    Beyond a matching slice *count*, this also checks -- when every series has a fully
    numbered InstanceNumber sequence -- that the series share the same InstanceNumber
    *set*. Two series can have equal slice counts while actually covering different
    anatomical slices (e.g. each is independently missing a different slice), which
    would otherwise silently pair up mismatched anatomy voxel-for-voxel during
    training/evaluation.
    """
    use_external_mask = cfg.task.mask_source == "external"
    candidates = list(cfg.data.patient_ids) if cfg.data.patient_ids else _enumerate_candidate_pids(cfg)

    kept = []
    for pid in candidates:
        input_dir, target_dir, mask_dir = pid_series_dirs(cfg, pid)
        input_count, input_nums = _scan_series(input_dir, "CT")
        target_count, target_nums = _scan_series(target_dir, "CT")
        counts = {"input": input_count, "target": target_count}
        nums = {"input": input_nums, "target": target_nums}
        if use_external_mask:
            mask_count, mask_nums = _scan_series(mask_dir, "OT")
            counts["mask"] = mask_count
            nums["mask"] = mask_nums
        if any(v == 0 for v in counts.values()):
            print(f"[discover_patients] skip {pid}: empty series ({counts})")
            continue
        if len(set(counts.values())) != 1:
            print(f"[discover_patients] skip {pid}: slice count mismatch ({counts})")
            continue

        # Only trust the InstanceNumber sets for this check when every series had one
        # fully-numbered slice per usable slice; otherwise fall back to the count-only
        # check above (permissive, matching prior behavior) rather than risk a false skip.
        fully_numbered = all(len(nums[k]) == counts[k] for k in counts)
        if fully_numbered:
            reference = next(iter(nums.values()))
            mismatched = {k: v for k, v in nums.items() if v != reference}
            if mismatched:
                print(
                    f"[discover_patients] skip {pid}: slice counts match ({counts}) but "
                    f"InstanceNumber sets differ across series -- likely misaligned slices "
                    f"({ {k: sorted(v) for k, v in nums.items()} })"
                )
                continue

        kept.append(pid)

    if not kept:
        raise RuntimeError(f"No usable patients found under {cfg.data.root}")
    return sorted(kept)


def build_records(cfg: Config, patient_ids: List[str]) -> List[dict]:
    use_external_mask = cfg.task.mask_source == "external"
    records = []
    for pid in patient_ids:
        input_dir, target_dir, mask_dir = pid_series_dirs(cfg, pid)
        rec = {"pid": pid, "image": str(input_dir), "target": str(target_dir)}
        if use_external_mask:
            # For body_threshold the mask is generated in the transform pipeline from
            # the input CT, so no mask path is needed in the record.
            rec["mask"] = str(mask_dir)
        records.append(rec)
    return records


def _group_of(pid: str) -> str:
    """Group prefix of a composite '<group>/<patient>' id."""
    return pid.split("/", 1)[0]


def make_folds(
    patient_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
    group_folds: bool = False,
    shuffle: bool = True,
) -> List[List[str]]:
    """Split patient IDs into folds.

    group_folds=True:  leave-one-group-out -- each fold is one group (by the '<group>/...'
                       prefix), in sorted group order. n_folds/shuffle are ignored.
    group_folds=False: split sorted IDs into n_folds chunks. shuffle=True applies a seeded
                       shuffle first (random folds); shuffle=False keeps sorted order, giving
                       deterministic contiguous blocks (e.g. 01-10, 11-20, ...).
    """
    ids = sorted(patient_ids)
    if group_folds:
        groups = sorted({_group_of(p) for p in ids})
        return [[p for p in ids if _group_of(p) == g] for g in groups]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)
    chunks = np.array_split(np.array(ids, dtype=object), n_folds)
    return [list(map(str, chunk)) for chunk in chunks]


def split_for_fold(
    records: List[dict], folds: List[List[str]], fold_idx: int
) -> Tuple[List[dict], List[dict]]:
    val_ids = set(folds[fold_idx])
    train_records = [r for r in records if r["pid"] not in val_ids]
    val_records = [r for r in records if r["pid"] in val_ids]
    return train_records, val_records


def split_for_protocol(
    records: List[dict], folds: List[List[str]], fold_idx: int, protocol: str, val_offset: int = 1
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return train/validation/test records without ever mixing subject IDs.

    ``train_val_test`` uses fold ``k`` as test and ``k + val_offset`` as
    validation. ``paper_two_way`` uses fold ``k`` as test and has no validation
    records. The legacy :func:`split_for_fold` remains a two-way helper.
    """
    if not (0 <= fold_idx < len(folds)):
        raise ValueError(f"fold {fold_idx} out of range: only {len(folds)} folds available")
    test_ids = set(folds[fold_idx])
    if protocol == "train_val_test":
        if len(folds) < 3:
            raise ValueError("train_val_test requires at least three non-empty folds")
        val_idx = (fold_idx + val_offset) % len(folds)
        if val_idx == fold_idx:
            raise ValueError("validation fold must differ from the test fold")
        val_ids = set(folds[val_idx])
    elif protocol == "paper_two_way":
        val_ids = set()
    else:
        raise ValueError(f"Unknown cv.protocol: {protocol!r}")
    train_records = [r for r in records if r["pid"] not in test_ids | val_ids]
    val_records = [r for r in records if r["pid"] in val_ids]
    test_records = [r for r in records if r["pid"] in test_ids]
    if not train_records or not test_records or (protocol == "train_val_test" and not val_records):
        raise RuntimeError(
            f"Empty protocol split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
        )
    return train_records, val_records, test_records


def num_folds(cfg: Config) -> int:
    """Actual number of CV folds. For group_folds this is the number of DE groups
    (data-determined); otherwise cfg.cv.n_folds."""
    if cfg.cv.group_folds:
        patient_ids = discover_patients(cfg)
        return len({_group_of(p) for p in patient_ids})
    return cfg.cv.n_folds


def _split_records_for_fold(cfg: Config, fold_idx: int) -> Tuple[List[dict], List[dict]]:
    from monai.utils import set_determinism

    set_determinism(cfg.seed)
    patient_ids = discover_patients(cfg)
    records = build_records(cfg, patient_ids)
    folds = make_folds(
        patient_ids,
        n_folds=cfg.cv.n_folds,
        seed=cfg.seed,
        group_folds=cfg.cv.group_folds,
        shuffle=cfg.cv.shuffle,
    )
    if not (0 <= fold_idx < len(folds)):
        raise ValueError(f"fold {fold_idx} out of range: only {len(folds)} folds available")
    return split_for_fold(records, folds, fold_idx)


def _split_records_for_protocol(
    cfg: Config, fold_idx: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    from monai.utils import set_determinism

    set_determinism(cfg.seed)
    patient_ids = discover_patients(cfg)
    records = build_records(cfg, patient_ids)
    folds = make_folds(
        patient_ids,
        n_folds=cfg.cv.n_folds,
        seed=cfg.seed,
        group_folds=cfg.cv.group_folds,
        shuffle=cfg.cv.shuffle,
    )
    if any(len(fold) == 0 for fold in folds):
        raise ValueError("Cross-validation contains an empty fold; reduce cv.n_folds or add subjects")
    return split_for_protocol(records, folds, fold_idx, cfg.cv.protocol, cfg.cv.val_offset)


def _dataset(data, transform, cfg: Config, *, training: bool):
    from monai.data import Dataset, PersistentDataset

    if cfg.data.cache_dir:
        # PersistentDataset keys its cache by the transform pipeline, not by the input
        # file contents. If the DICOM data at an unchanged path is replaced, delete
        # cfg.data.cache_dir so the stale cache is not silently reused.
        cache_dir = Path(cfg.data.cache_dir) / ("train" if training else "eval")
        return PersistentDataset(data=data, transform=transform, cache_dir=cache_dir)
    return Dataset(data=data, transform=transform)


def _loader(dataset, cfg: Config, *, batch_size: int, shuffle: bool, drop_last: bool):
    from monai.data import DataLoader, list_data_collate

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
        pin_memory=torch.cuda.is_available() and cfg.runtime.device != "cpu",
        drop_last=drop_last,
        collate_fn=list_data_collate,
    )


def build_protocol_loaders(cfg: Config, fold_idx: int) -> dict:
    """Build protocol-aware train/validation/test loaders and record manifests."""
    train_records, val_records, test_records = _split_records_for_protocol(cfg, fold_idx)
    train_ds = _dataset(train_records, build_train_transforms(cfg), cfg, training=True)
    val_ds = _dataset(val_records, build_val_transforms(cfg), cfg, training=False) if val_records else None
    test_ds = _dataset(test_records, build_val_transforms(cfg), cfg, training=False)
    loaders = {
        "train": _loader(train_ds, cfg, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True),
        "val": _loader(val_ds, cfg, batch_size=1, shuffle=False, drop_last=False) if val_ds else None,
        "test": _loader(test_ds, cfg, batch_size=1, shuffle=False, drop_last=False),
        "records": {"train": train_records, "val": val_records, "test": test_records},
    }
    if len(loaders["train"]) == 0:
        raise RuntimeError(
            f"Fold {fold_idx}: training loader is empty; lower train.batch_size or add subjects"
        )
    return loaders


def build_val_loader(cfg: Config, fold_idx: int):
    """Build only the two-way validation DataLoader for a fold.

    Used by unit tests and diagnostics (not by the predict/eval CLI commands, which
    build the protocol's held-out test loader). Does not build or validate the
    training loader, so it works regardless of train.batch_size vs. the split size.
    """
    from monai.data import DataLoader, Dataset, list_data_collate

    _, val_records = _split_records_for_fold(cfg, fold_idx)
    val_ds = Dataset(data=val_records, transform=build_val_transforms(cfg))
    return DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=list_data_collate,
    )


def build_test_loader(cfg: Config, fold_idx: int):
    """Build the held-out test loader selected by ``cv.protocol``."""
    _, _, test_records = _split_records_for_protocol(cfg, fold_idx)
    test_ds = _dataset(test_records, build_val_transforms(cfg), cfg, training=False)
    return _loader(test_ds, cfg, batch_size=1, shuffle=False, drop_last=False)


def build_loaders(cfg: Config, fold_idx: int):
    """Build (train_loader, val_loader) for the given fold index (used by training)."""
    from monai.data import DataLoader, Dataset, list_data_collate

    train_records, val_records = _split_records_for_fold(cfg, fold_idx)

    train_ds = Dataset(data=train_records, transform=build_train_transforms(cfg))
    val_ds = Dataset(data=val_records, transform=build_val_transforms(cfg))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        persistent_workers=(cfg.train.num_workers > 0),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=list_data_collate,
    )

    # drop_last=True means a training set with fewer patients than batch_size yields
    # zero batches and silently trains on nothing. Fail loudly instead.
    if len(train_loader) == 0:
        raise RuntimeError(
            f"Fold {fold_idx}: the training loader is empty. The training split has "
            f"{len(train_records)} patient(s), which is fewer than train.batch_size="
            f"{cfg.train.batch_size} (with drop_last=True). Lower train.batch_size, "
            f"reduce cv.n_folds, or provide more patients."
        )
    return train_loader, val_loader
