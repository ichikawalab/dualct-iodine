# -*- coding: utf-8 -*-
"""Configuration system: nested dataclasses + YAML I/O + dotted-key CLI overrides.

Precedence (highest to lowest), enforced by callers (see cli.py):
    dedicated CLI flags > --set key.path=value > YAML file > dataclass defaults

Note: `Config.from_yaml` merges the given YAML onto a fresh `Config()`, i.e. onto the
dataclass defaults defined below (not onto any other YAML file). This is why
`configs/iodine.yaml` and `configs/kvp.yaml` are each written as complete, standalone
files listing every key explicitly -- neither one inherits values from the other, only
from these dataclass defaults for any key it happens to omit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml


# ----------------------------------------------------------------------------
# Nested config sections
# ----------------------------------------------------------------------------
@dataclass
class DataCfg:
    root: str = "./"
    input_subdir: str = "120kV_Iodinemap/120 kVp"
    target_subdir: str = "120kV_Iodinemap/iodinemaps"
    mask_subdir: str = "MASK"
    patient_ids: Optional[List[str]] = None
    # False (iodine): flat layout <root>/<input_subdir>/<patient>/...
    # True (kVp): nested layout <root>/<group>/<input_subdir>/<patient>/..., where <group>
    #   is a DE-group folder (DE1..DE5). Patient IDs become composite "<group>/<patient>"
    #   because patient numbers are only unique within a group.
    nested_groups: bool = False
    cache_dir: Optional[str] = None


@dataclass
class TaskCfg:
    # external: read a provided binary mask series (iodine task, MASK/ folder).
    # body_threshold: auto-generate a body mask from the input CT via an HU threshold
    #   (kVp->kVp task; see transforms.BodyMaskd).
    mask_source: str = "external"       # {external | body_threshold}
    body_thr_hu: float = -600           # used only when mask_source == body_threshold


@dataclass
class NormalizeCfg:
    input_hu_min: float = -1024
    input_hu_max: float = 3071
    target_min: float = 0
    target_max: float = 200


@dataclass
class ModelCfg:
    name: str = "swinunetr"
    feature_size: int = 48
    use_checkpoint: bool = True
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    residual: bool = False
    residual_alpha: float = 1.0
    # MONAI UNet settings (used only when name == "unet").
    unet_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    unet_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    unet_num_res_units: int = 2
    unet_norm: str = "INSTANCE"
    unet_dropout: float = 0.0


@dataclass
class LossCfg:
    region: str = "mask"       # {mask | full}
    main: str = "l1"
    aux: str = "cdf"           # {none | cdf}
    lam_cdf: float = 0.3
    cdf_M: int = 1024
    cdf_max_voxels: int = 65536


@dataclass
class TrainCfg:
    roi_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    batch_size: int = 4
    num_epochs: int = 500
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    amp: bool = True
    amp_dtype: str = "float16"  # {float16 | bfloat16}; autocast dtype, used only on CUDA when amp is True
    num_workers: int = 0
    validate_every: int = 20
    num_samples_per_volume: int = 2


@dataclass
class EarlyStoppingCfg:
    enabled: bool = True
    monitor: str = "mse_mask"
    patience_checks: int = 5
    min_delta_relative: float = 0.001
    min_epochs: int = 200
    restore_best: bool = True


@dataclass
class CheckpointCfg:
    selection: str = "auto"  # auto -> best for three-way, final for two-way
    monitor: str = "auto"


@dataclass
class RuntimeCfg:
    device: str = "auto"  # auto | cpu | cuda | cuda:N


@dataclass
class GeometryCfg:
    strict: bool = True
    # Shared tolerance for PixelSpacing/ImagePositionPatient (mm) and
    # ImageOrientationPatient (unitless direction cosines) equality checks between
    # co-registered series. 0.5 tolerates sub-voxel DICOM encoding/rounding noise
    # while still catching genuine misalignment (typically several mm or more).
    atol: float = 0.5


@dataclass
class InferCfg:
    sw_batch_size: int = 4
    sw_overlap: float = 0.25
    sw_mode: str = "gaussian"
    eval_M: int = 256
    # zero: fill outside the mask with 0 (== target_min; iodine task, non-lung is 0).
    # input: restore the input value outside the mask (kVp->kVp task, same domain).
    outside_fill: str = "zero"          # {zero | input}


@dataclass
class CvCfg:
    n_folds: int = 5
    fold: int = 0
    # False: random patient-level split into n_folds.
    # True: leave-one-group-out, i.e. each fold is one DE group (requires nested_groups).
    #   n_folds is then set to the number of groups.
    group_folds: bool = False
    # Only applies when group_folds is False. True: shuffle patient IDs before splitting.
    # False: deterministic contiguous blocks of sorted IDs (e.g. 01-10, 11-20, ... -- this
    #   aligns the flat iodine cohort with the kVp DE groups, since iodine 01-50 == DE1..DE5).
    shuffle: bool = True
    protocol: str = "train_val_test"  # {train_val_test | paper_two_way}
    val_offset: int = 1


@dataclass
class OutputCfg:
    ckpt_dir: str = "./checkpoints"
    pred_dir: str = "./predictions"
    metrics_dir: str = "./metrics"
    pred_suffix: str = "SynthIodine"    # appended to SeriesDescription of saved predictions
    # Mask export remains disabled until a standards-compliant DICOM SEG writer exists.
    save_mask: bool = False
    # Reserved for a future DICOM SEG mask export; mask writing is currently disabled
    # (see Config.validate and inference.save_mask_as_dicom), so this value is unused today.
    mask_suffix: str = "Mask"


@dataclass
class Config:
    seed: int = 42
    data: DataCfg = field(default_factory=DataCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    infer: InferCfg = field(default_factory=InferCfg)
    cv: CvCfg = field(default_factory=CvCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        _merge_dict_into(cfg, raw)
        cfg.validate()
        return cfg

    # -- mutation -------------------------------------------------------
    def apply_overrides(self, overrides: List[str]) -> None:
        """Apply a list of ``key.path=value`` strings (dotted-key overrides)."""
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Invalid override (expected key.path=value): {ov!r}")
            key, raw_value = ov.split("=", 1)
            _set_dotted(self, key.strip(), raw_value.strip())
        self.validate()

    # -- serialization -------------------------------------------------------
    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, allow_unicode=True, sort_keys=False)

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        if self.normalize.target_min >= self.normalize.target_max:
            raise ValueError("normalize.target_min must be < normalize.target_max")
        if self.normalize.input_hu_min >= self.normalize.input_hu_max:
            raise ValueError("normalize.input_hu_min must be < normalize.input_hu_max")
        if self.loss.region not in ("mask", "full"):
            raise ValueError("loss.region must be one of {'mask', 'full'}")
        if self.loss.main != "l1":
            raise ValueError("loss.main must be 'l1' -- no other main loss is implemented")
        if self.loss.aux not in ("none", "cdf"):
            raise ValueError("loss.aux must be one of {'none', 'cdf'}")
        if self.task.mask_source not in ("external", "body_threshold"):
            raise ValueError("task.mask_source must be one of {'external', 'body_threshold'}")
        if self.infer.outside_fill not in ("zero", "input"):
            raise ValueError("infer.outside_fill must be one of {'zero', 'input'}")
        if len(self.train.roi_size) != 3:
            raise ValueError("train.roi_size must have length 3")
        if self.cv.n_folds < 2:
            raise ValueError("cv.n_folds must be >= 2")
        if not (0 <= self.cv.fold < self.cv.n_folds):
            raise ValueError("cv.fold must satisfy 0 <= fold < cv.n_folds")
        if self.cv.group_folds and not self.data.nested_groups:
            raise ValueError("cv.group_folds=True requires data.nested_groups=True")
        if self.output.save_mask:
            raise ValueError(
                "output.save_mask is not supported: use a standards-compliant DICOM SEG exporter"
            )
        if self.model.name not in ("swinunetr", "unet"):
            raise ValueError("model.name must be one of {'swinunetr', 'unet'}")
        if self.model.in_channels < 1 or self.model.out_channels < 1:
            raise ValueError("model in_channels/out_channels must be >= 1")
        if self.model.name == "swinunetr" and (
            self.model.feature_size <= 0 or self.model.feature_size % 12 != 0
        ):
            raise ValueError("model.feature_size must be a positive multiple of 12")
        if len(self.model.unet_channels) != len(self.model.unet_strides) + 1:
            raise ValueError("model.unet_channels must have one more entry than model.unet_strides")
        if any(v <= 0 for v in self.model.unet_channels + self.model.unet_strides):
            raise ValueError("UNet channels and strides must be positive")
        if not (0.0 <= self.model.unet_dropout < 1.0):
            raise ValueError("model.unet_dropout must satisfy 0 <= dropout < 1")
        if any(v <= 0 for v in self.train.roi_size):
            raise ValueError("train.roi_size entries must be positive")
        if self.model.name == "unet":
            downsample = 1
            for stride in self.model.unet_strides:
                downsample *= stride
            if any(v % downsample != 0 for v in self.train.roi_size):
                raise ValueError(f"train.roi_size must be divisible by UNet total stride {downsample}")
        if self.train.batch_size < 1 or self.train.num_epochs < 1 or self.train.validate_every < 1:
            raise ValueError("train batch_size/num_epochs/validate_every must be >= 1")
        if self.train.lr <= 0 or self.train.weight_decay < 0:
            raise ValueError("train.lr must be > 0 and weight_decay must be >= 0")
        if self.train.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("train.amp_dtype must be one of {'float16', 'bfloat16'}")
        if self.loss.cdf_M < 2 or self.loss.cdf_max_voxels < 2 or self.loss.lam_cdf < 0:
            raise ValueError("CDF settings must be positive (cdf_M/cdf_max_voxels >= 2)")
        if self.cv.protocol not in ("train_val_test", "paper_two_way"):
            raise ValueError("cv.protocol must be one of {'train_val_test', 'paper_two_way'}")
        if self.cv.val_offset < 1:
            raise ValueError("cv.val_offset must be >= 1")
        if self.checkpoint.selection not in ("auto", "best", "final"):
            raise ValueError("checkpoint.selection must be one of {'auto', 'best', 'final'}")
        if self.cv.protocol == "paper_two_way":
            if self.checkpoint.selection == "best":
                raise ValueError("paper_two_way does not permit best checkpoint selection")
            if self.early_stopping.enabled:
                raise ValueError("paper_two_way does not permit early stopping")
        if self.early_stopping.patience_checks < 1:
            raise ValueError("early_stopping.patience_checks must be >= 1")
        if not (0 <= self.early_stopping.min_delta_relative < 1):
            raise ValueError("early_stopping.min_delta_relative must satisfy 0 <= value < 1")
        if self.early_stopping.min_epochs < 0:
            raise ValueError("early_stopping.min_epochs must be >= 0")
        if self.runtime.device != "auto" and self.runtime.device != "cpu" and not self.runtime.device.startswith("cuda"):
            raise ValueError("runtime.device must be auto, cpu, cuda, or cuda:N")
        if self.geometry.atol < 0:
            raise ValueError("geometry.atol must be >= 0")
        if self.model.residual and (
            self.model.in_channels != self.model.out_channels
            or self.normalize.input_hu_min != self.normalize.target_min
            or self.normalize.input_hu_max != self.normalize.target_max
        ):
            raise ValueError("residual model requires matching channels and input/target normalization ranges")
        if self.infer.outside_fill == "input" and (
            self.normalize.input_hu_min != self.normalize.target_min
            or self.normalize.input_hu_max != self.normalize.target_max
        ):
            raise ValueError("infer.outside_fill=input requires identical input/target normalization ranges")


# ----------------------------------------------------------------------------
# Helpers: nested-dict merge, dotted-key set, typed casting
# ----------------------------------------------------------------------------
def _merge_dict_into(obj: Any, data: dict) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key!r} on {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dict_into(current, value)
        else:
            setattr(obj, key, value)


def _resolve_path(cfg: Config, dotted_key: str):
    """Return (parent_obj, leaf_attr_name) for a dotted key, validating existence."""
    parts = dotted_key.split(".")
    target = cfg
    for p in parts[:-1]:
        if not hasattr(target, p):
            raise ValueError(f"Unknown config key segment {p!r} in {dotted_key!r}")
        target = getattr(target, p)
        if not is_dataclass(target):
            raise ValueError(f"Config key {dotted_key!r} does not resolve to a nested section")
    last = parts[-1]
    if not hasattr(target, last):
        raise ValueError(f"Unknown config key: {dotted_key!r}")
    return target, last


def _set_dotted(cfg: Config, dotted_key: str, raw_value: str) -> None:
    target, last = _resolve_path(cfg, dotted_key)
    current = getattr(target, last)
    setattr(target, last, _cast_value(current, raw_value))


def _cast_value(current: Any, raw: str) -> Any:
    """Cast a raw CLI string to the type of the existing default value."""
    if current is None:
        # No type information available (e.g. data.patient_ids): parse as YAML literal.
        return yaml.safe_load(raw)
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Cannot parse bool from {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        s = raw.strip()
        if s.startswith("["):
            parsed = yaml.safe_load(s)
        else:
            parsed = [p.strip() for p in s.split(",") if p.strip() != ""]
        if current and isinstance(current[0], bool):
            parsed = [bool(_cast_value(current[0], str(x))) for x in parsed]
        elif current and isinstance(current[0], int):
            parsed = [int(x) for x in parsed]
        elif current and isinstance(current[0], float):
            parsed = [float(x) for x in parsed]
        return list(parsed)
    if isinstance(current, str):
        return raw
    # Fallback: best-effort YAML parse.
    return yaml.safe_load(raw)


# ----------------------------------------------------------------------------
# Convenience loader used by the CLI
# ----------------------------------------------------------------------------
def load_config(config_path: Union[str, Path], set_overrides: Optional[List[str]] = None) -> Config:
    """Load a YAML config and apply ``--set key=value`` overrides, in that order."""
    cfg = Config.from_yaml(config_path)
    if set_overrides:
        cfg.apply_overrides(set_overrides)
    return cfg
