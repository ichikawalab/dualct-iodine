from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import Config


def config_digest(cfg: Config) -> str:
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_signature(cfg: Config) -> dict[str, Any]:
    return {
        "model": dataclasses.asdict(cfg.model),
        "normalize": dataclasses.asdict(cfg.normalize),
        "task": dataclasses.asdict(cfg.task),
        "protocol": cfg.cv.protocol,
    }


def save_full_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    fold_idx: int,
    cfg: Config,
    metrics: dict[str, float],
    selection_source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "fold": fold_idx,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "config": dataclasses.asdict(cfg),
            "config_digest": config_digest(cfg),
            "model_signature": model_signature(cfg),
            "selection_source": selection_source,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def save_weights(path: Path, model: torch.nn.Module, cfg: Config, *, epoch: int, selection_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    metadata = {
        "config_digest": config_digest(cfg),
        "model_signature": json.dumps(model_signature(cfg), sort_keys=True),
        "epoch": str(epoch),
        "selection_source": selection_source,
    }
    save_file(tensors, str(path), metadata=metadata)


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "net.", "state_dict."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        if new_key in stripped:
            raise ValueError(f"Checkpoint prefix stripping produced duplicate key: {new_key}")
        stripped[new_key] = value
    return stripped


def load_weights(model: torch.nn.Module, path: str | Path, cfg: Config, *, strict: bool = True) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        state_dict = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        expected_signature = json.dumps(model_signature(cfg), sort_keys=True)
        if metadata.get("model_signature") != expected_signature:
            raise ValueError("Checkpoint model/task signature does not match the resolved configuration")
    else:
        # weights_only prevents arbitrary pickle globals. Full resume checkpoints
        # are intentionally not accepted by this inference loader.
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected checkpoint type: {type(obj)}")
        state_dict = obj.get("state_dict", obj)
        metadata = {}
        signature = obj.get("model_signature")
        if signature is not None and signature != model_signature(cfg):
            raise ValueError("Checkpoint model signature does not match configuration")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no usable state_dict")
    model.load_state_dict(_strip_prefixes(state_dict), strict=strict)
    return metadata
