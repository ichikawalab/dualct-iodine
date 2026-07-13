# DualCT Image Translation

PyTorch and MONAI implementation for 3D translation of 120 kVp CT into iodine
maps or 80/140 kVp CT volumes, using cross-validation.

> **Research use only.** This software and its generated images are not
> validated medical devices and must not be used for clinical decision-making.

## Evaluation protocol

The default experiment (`configs/iodine.yaml`) is patient-level five-fold
cross-validation. For fold `k`, fold `k` is the test set, fold `(k + 1) mod 5`
is the validation set, and the remaining three folds are the training set. The
test set is never used for training, early stopping, or checkpoint selection.

`best.safetensors` is selected by masked validation MSE. Reported metrics are
MSE, PSNR, and slice-wise SSIM for the full image and evaluation mask.

The kVp configuration (`configs/kvp.yaml`, 120 kVp -> 80/140 kVp CT) instead
uses leave-one-DE-group-out cross-validation: five DE groups yield five folds,
each with three groups for training, one for validation, and one held out for
test.

Both tasks also support a two-way protocol
(`--set cv.protocol=paper_two_way early_stopping.enabled=false`): fold `k` is the
test set and the remaining four folds are the training set, with no validation
split. Without a validation set there is no early stopping or best-checkpoint
selection, so the model trains for the full `train.num_epochs` and the final
checkpoint (`final.safetensors`) is evaluated on the test fold.

## Data format and protection

The iodine configuration expects geometrically aligned input, target, and mask
series:

```text
<DATA_ROOT>/
|-- 120kV_Iodinemap/
|   |-- 120 kVp/<patient_id>/*.dcm
|   `-- iodinemaps/<patient_id>/*.dcm
`-- MASK/<patient_id>/*
```

The kVp configuration expects nested DE groups instead, with no separate mask
series (a body mask is auto-generated from the input CT):

```text
<DATA_ROOT>/
|-- DE1/
|   |-- 120kV/<patient_id>/*.dcm
|   |-- 80kV/<patient_id>/*.dcm
|   `-- 140kV/<patient_id>/*.dcm
|-- DE2/
|   `-- ...
`-- DE5/
    `-- ...
```

Set `data.root` or `--data-root` to approved local storage. Clinical images,
derived DICOM, masks, predictions, patient-level tables, and trained weights are
not included and must never be committed to this repository. DICOM headers and
derived images can contain patient information; de-identify them with a validated
local workflow before sharing.

## Installation

Python 3.10--3.12 and [uv](https://docs.astral.sh/uv/) are recommended. The lock
file uses PyTorch 2.5.1 with CUDA 11.8.

```powershell
git clone https://github.com/ichikawalab/dualct-iodine.git
cd dualct-iodine
uv sync --extra dev
uv run dualct --help
```

For CPU execution, set `runtime.device=cpu` and disable AMP. For NVIDIA GPU
execution, ensure that the installed PyTorch build is compatible with your driver.
On large datasets, raise `train.num_workers` to speed up data loading, and set
`train.amp_dtype=bfloat16` on hardware where it is more numerically robust than
float16.

Training seeds the Python, NumPy, and PyTorch RNGs, but results are not guaranteed
to be bit-for-bit identical across different hardware, drivers, or library versions.

## Training

Train one fold with the default SwinUNETR model:

```powershell
uv run dualct train --config configs/iodine.yaml --data-root D:/secure/dualct --fold 0
```

Run all five folds:

```powershell
uv run dualct cv --config configs/iodine.yaml --data-root D:/secure/dualct
```

Use the MONAI 3D UNet:

```powershell
uv run dualct train --config configs/iodine.yaml --data-root D:/secure/dualct --set model.name=unet --fold 0
```

Resolved configurations, split-specific metrics, and checkpoints are saved with
each run. `.safetensors` is the preferred inference checkpoint format.

Run the kVp task (120 kVp -> 80 kVp CT) with all five DE-group folds:

```powershell
uv run dualct cv --config configs/kvp.yaml --data-root D:/secure/dualct_kvp
```

For 120 kVp -> 140 kVp, override the target subdirectory and prediction suffix:

```powershell
uv run dualct cv --config configs/kvp.yaml --data-root D:/secure/dualct_kvp --set data.target_subdir=140kV output.pred_suffix=Synth_140kV
```

## Evaluation and inference

Evaluate a checkpoint, iodine task:

```powershell
uv run dualct eval --config configs/iodine.yaml --fold 0 --ckpt checkpoints/train_val_test/fold0/best.safetensors
```

Evaluate a checkpoint, kVp task:

```powershell
uv run dualct eval --config configs/kvp.yaml --fold 0 --ckpt checkpoints/train_val_test/fold0/best.safetensors
```

Run inference on independent data, iodine task (`--mask-dir` is required
because this task expects an external mask):

```powershell
uv run dualct predict-dir --config configs/iodine.yaml --ckpt checkpoints/train_val_test/fold0/best.safetensors --input-dir D:/secure/external/120kVp --mask-dir D:/secure/external/lung_masks --out-dir D:/secure/derived/iodine
```

Run inference on independent data, kVp task (no `--mask-dir`; the body mask is
auto-generated from the input CT):

```powershell
uv run dualct predict-dir --config configs/kvp.yaml --ckpt checkpoints/train_val_test/fold0/best.safetensors --input-dir D:/secure/external/120kVp --out-dir D:/secure/derived/kvp
```

Generated DICOM instances receive new UIDs and are marked `DERIVED/SECONDARY`;
this is not a substitute for validated de-identification.

## Tests

```powershell
uv run ruff check .
uv run pytest -q
uv build
```

GitHub Actions runs linting, synthetic-data tests, package builds, and command
checks on Python 3.10, 3.11, and 3.12.

## Limitations

- Cross-validation is not external validation.
- Performance may not generalize across scanners, protocols, institutions, or populations.
- Generated images require task-specific clinical and technical validation.

## License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
