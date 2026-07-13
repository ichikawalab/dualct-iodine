"""Protocol-aware training, validation-only model selection, and cross-validation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .checkpointing import load_weights, save_full_checkpoint, save_weights
from .config import Config
from .data import build_protocol_loaders
from .losses import build_loss
from .metrics import evaluate
from .model import build_model

PUBLIC_METRICS = ("mse_mask", "psnr_mask", "ssim_mask", "mse_full", "psnr_full", "ssim_full")


def resolve_device(cfg: Config) -> torch.device:
    requested = cfg.runtime.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def amp_settings(cfg: Config, device: torch.device) -> tuple[bool, torch.dtype]:
    """Return ``(autocast_enabled, autocast_dtype)`` shared by training and inference.

    Autocast is enabled only on CUDA when ``train.amp`` is set; its dtype follows
    ``train.amp_dtype`` (float16 by default, bfloat16 where it is more numerically
    robust). On CPU, or when amp is off, autocast is disabled and the dtype is float32.
    """
    if device.type == "cuda" and cfg.train.amp:
        dtype = torch.bfloat16 if cfg.train.amp_dtype == "bfloat16" else torch.float16
        return True, dtype
    return False, torch.float32


def _publication_view(metrics: dict[str, float]) -> dict[str, float]:
    allowed = {"n_cases", "infer_time_s", "infer_time_s_sd"}
    for name in PUBLIC_METRICS:
        allowed.add(name)
        allowed.add(f"{name}_sd")
    return {key: value for key, value in metrics.items() if key in allowed}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def _is_improved(score: float, best: float, relative_delta: float) -> bool:
    if not torch.isfinite(torch.tensor(score)):
        return False
    if best == float("inf"):
        return True
    return score < best * (1.0 - relative_delta)


def train_one_fold(cfg: Config, fold_idx: int) -> dict[str, float]:
    """Train one fold and evaluate its held-out test fold exactly once."""
    cfg.validate()
    device = resolve_device(cfg)
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    # GradScaler is needed only for float16 autocast; bfloat16 has enough dynamic
    # range that loss scaling is unnecessary (and the scaler stays a no-op on CPU).
    use_scaler = amp_enabled and amp_dtype == torch.float16
    bundle = build_protocol_loaders(cfg, fold_idx)
    train_loader = bundle["train"]
    val_loader = bundle["val"]
    test_loader = bundle["test"]

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = build_loss(cfg)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except AttributeError:  # pragma: no cover - older torch compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    fold_dir = Path(cfg.output.ckpt_dir) / cfg.cv.protocol / f"fold{fold_idx}"
    metrics_dir = Path(cfg.output.metrics_dir) / cfg.cv.protocol / f"fold{fold_idx}"
    best_score = float("inf")
    best_epoch = 0
    checks_without_improvement = 0
    last_val_metrics: dict[str, float] = {}
    stop_epoch = cfg.train.num_epochs
    train_start = time.perf_counter()

    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        running = {"total": 0.0, "l1": 0.0, "cdf": 0.0}
        nimg = 0
        pbar = tqdm(train_loader, desc=f"[{cfg.cv.protocol} fold{fold_idx}] {epoch}/{cfg.train.num_epochs}", ncols=120)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = model(x, inference=False)
                total, logs = loss_fn(prediction, y, mask)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = x.size(0)
            for key in running:
                running[key] += logs.get(key, 0.0) * batch_size
            nimg += batch_size
            pbar.set_postfix(**{key: value / max(nimg, 1) for key, value in running.items()})

        train_metrics = {f"train_{key}": value / max(nimg, 1) for key, value in running.items()}
        save_full_checkpoint(
            fold_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            fold_idx=fold_idx,
            cfg=cfg,
            metrics={**train_metrics, **last_val_metrics},
            selection_source="latest_epoch",
        )

        do_validate = val_loader is not None and (
            epoch == 1 or epoch == cfg.train.num_epochs or epoch % cfg.train.validate_every == 0
        )
        if do_validate:
            last_val_metrics = _publication_view(evaluate(model, val_loader, cfg))
            score = last_val_metrics["mse_mask"]
            improved = _is_improved(score, best_score, cfg.early_stopping.min_delta_relative)
            if improved:
                best_score = score
                best_epoch = epoch
                checks_without_improvement = 0
                save_full_checkpoint(
                    fold_dir / "best.ckpt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    fold_idx=fold_idx,
                    cfg=cfg,
                    metrics={**train_metrics, **last_val_metrics},
                    selection_source="validation_mse_mask",
                )
                save_weights(
                    fold_dir / "best.safetensors",
                    model,
                    cfg,
                    epoch=epoch,
                    selection_source="validation_mse_mask",
                )
            else:
                checks_without_improvement += 1
            print(
                f"[fold{fold_idx} epoch {epoch}] val mse_mask={score:.6f} "
                f"best={best_score:.6f}@{best_epoch} no_improve={checks_without_improvement}"
            )
            if (
                cfg.cv.protocol == "train_val_test"
                and cfg.early_stopping.enabled
                and epoch >= cfg.early_stopping.min_epochs
                and checks_without_improvement >= cfg.early_stopping.patience_checks
            ):
                stop_epoch = epoch
                print(f"[fold{fold_idx}] early stopping at epoch {epoch}")
                break

    save_full_checkpoint(
        fold_dir / "final.ckpt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=stop_epoch,
        fold_idx=fold_idx,
        cfg=cfg,
        metrics=last_val_metrics,
        selection_source="final_epoch",
    )
    save_weights(
        fold_dir / "final.safetensors", model, cfg, epoch=stop_epoch, selection_source="final_epoch"
    )

    selection = cfg.checkpoint.selection
    if selection == "auto":
        selection = "best" if cfg.cv.protocol == "train_val_test" else "final"
    selected_path = fold_dir / f"{selection}.safetensors"
    load_weights(model, selected_path, cfg, strict=True)
    test_metrics = _publication_view(evaluate(model, test_loader, cfg))
    test_metrics["train_time_s"] = time.perf_counter() - train_start
    test_metrics["selected_epoch"] = float(best_epoch if selection == "best" else stop_epoch)
    test_metrics["stop_epoch"] = float(stop_epoch)
    _write_json(
        metrics_dir / "metrics.json",
        {
            "protocol": cfg.cv.protocol,
            "fold": fold_idx,
            "selection": selection,
            "selection_source": "validation_mse_mask" if selection == "best" else "final_epoch",
            "metrics": test_metrics,
        },
    )
    return test_metrics


def run_cv(cfg: Config):
    """Run every fold and write a protocol-specific summary."""
    import pandas as pd

    from .data import num_folds

    n_folds = num_folds(cfg)
    rows = []
    for fold_idx in range(n_folds):
        print(f"=== {cfg.cv.protocol} fold {fold_idx}/{n_folds - 1} ===")
        rows.append({"fold": fold_idx, **train_one_fold(cfg, fold_idx)})
    frame = pd.DataFrame(rows)
    mean_row: dict[str, Any] = {"fold": "mean"}
    sd_row: dict[str, Any] = {"fold": "sd"}
    for column in frame.columns:
        if column != "fold":
            mean_row[column] = frame[column].mean()
            sd_row[column] = frame[column].std()
    summary = pd.concat([frame, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
    output = Path(cfg.output.metrics_dir) / cfg.cv.protocol / "cv_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[run_cv] saved {output}")
    return summary
