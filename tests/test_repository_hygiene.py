from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = {
    "data",
    "outputs",
    "predictions",
    "checkpoints",
    "metrics",
    "runs",
    "artifacts",
    "models",
    "weights",
    "logs",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".onnx",
    ".dcm",
    ".dicom",
    ".nii",
    ".nrrd",
    ".mha",
    ".mhd",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".tsv",
    ".xls",
    ".xlsx",
    ".parquet",
}


def test_repository_tracks_no_patient_data_or_model_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Repository hygiene check requires a Git checkout.")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    violations: list[str] = []
    for item in tracked:
        path = Path(item)
        if path.parts and path.parts[0].lower() in FORBIDDEN_ROOTS:
            violations.append(item)
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".nii.gz") or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(item)
            continue
        if path.suffix.lower() == ".csv" and path.parts[:1] != ("examples",):
            violations.append(item)
    assert not violations, f"Patient data or model/experiment artifacts are tracked: {violations}"
