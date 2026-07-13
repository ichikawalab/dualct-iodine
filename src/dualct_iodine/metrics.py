# -*- coding: utf-8 -*-
"""Evaluation metrics: MAE / RMSE / PSNR / W1(Wasserstein), computed in-mask and full.

All metrics are computed in physical (target-domain) units. `evaluate()` always
reports both `*_in` (mask region) and `*_full` (whole volume) variants per case,
then aggregates the per-case mean and SD -- independent of whichever region
`loss.region` is currently training on, so in-mask vs. full performance can
always be compared for the paper.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch

from .config import Config
from .losses import region_mask
from .transforms import denorm_target


def mae(pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor) -> float:
    m = region_bool.to(pred_phys.dtype)
    denom = m.sum()
    if denom == 0:
        return float("nan")
    return ((pred_phys - target_phys).abs() * m).sum().div(denom).item()


def mse(pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor) -> float:
    m = region_bool.to(pred_phys.dtype)
    denom = m.sum()
    if denom == 0:
        return float("nan")
    return ((((pred_phys - target_phys) ** 2) * m).sum() / denom).item()


def rmse(pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor) -> float:
    m = region_bool.to(pred_phys.dtype)
    denom = m.sum()
    if denom == 0:
        return float("nan")
    mse = (((pred_phys - target_phys) ** 2) * m).sum().div(denom)
    return torch.sqrt(mse).item()


def psnr(pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor, data_range: float) -> float:
    m = region_bool.to(pred_phys.dtype)
    denom = m.sum()
    if denom == 0:
        return float("nan")
    mse = (((pred_phys - target_phys) ** 2) * m).sum().div(denom)
    return (10.0 * torch.log10((data_range ** 2) / mse.clamp(min=1e-12))).item()


def _quantiles_1d(x: torch.Tensor, M: int) -> torch.Tensor:
    x = x.to(torch.float32)
    n = x.numel()
    if n == 0:
        return torch.zeros(M, device=x.device, dtype=x.dtype)
    xs = torch.sort(x)[0]
    if n == 1:
        return xs.new_full((M,), xs[0])
    u = torch.linspace(0, 1, steps=M, device=x.device, dtype=x.dtype)
    pos = u * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    t = pos - i0.float()
    return (1 - t) * xs[i0] + t * xs[i0 + 1]


def w1(pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor, M: int) -> float:
    p = pred_phys[region_bool].to(torch.float32)
    t = target_phys[region_bool].to(torch.float32)
    if p.numel() == 0:
        return float("nan")
    qa = _quantiles_1d(p, M)
    qb = _quantiles_1d(t, M)
    return torch.mean(torch.abs(qa - qb)).item()


def _mean_sd(vals) -> Tuple[float, float]:
    """Sample mean and sample SD (ddof=1), ignoring NaN entries (e.g. cases with an
    empty mask region). SD is NaN when fewer than 2 finite values are available --
    a sample SD is not defined for a single observation, so this is reported as
    "undefined", not silently as 0.
    """
    arr = np.asarray(vals, dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
    return mean, sd


def _ssim_map_2d(x: np.ndarray, y: np.ndarray, data_range: float, win_size: int = 11) -> np.ndarray:
    """Gaussian-window SSIM map with explicit, publication-stable parameters."""
    import scipy.ndimage as ndi

    # Production CT slices use the fixed 11-pixel window. The adaptive odd
    # window keeps small synthetic unit tests mathematically defined.
    win_size = min(win_size, min(x.shape))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(f"SSIM requires slices at least 3x3, got {x.shape}")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    sigma = 1.5
    mu_x = ndi.gaussian_filter(x, sigma=sigma, mode="reflect", truncate=3.5)
    mu_y = ndi.gaussian_filter(y, sigma=sigma, mode="reflect", truncate=3.5)
    sigma_x = ndi.gaussian_filter(x * x, sigma=sigma, mode="reflect", truncate=3.5) - mu_x * mu_x
    sigma_y = ndi.gaussian_filter(y * y, sigma=sigma, mode="reflect", truncate=3.5) - mu_y * mu_y
    sigma_xy = ndi.gaussian_filter(x * y, sigma=sigma, mode="reflect", truncate=3.5) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def ssim_volume(
    pred_phys: torch.Tensor, target_phys: torch.Tensor, region_bool: torch.Tensor | None, data_range: float
) -> float:
    pred = pred_phys.detach().cpu().numpy()[0, 0]
    target = target_phys.detach().cpu().numpy()[0, 0]
    mask = region_bool.detach().cpu().numpy()[0, 0].astype(bool) if region_bool is not None else None
    if mask is not None and not mask.any():
        return float("nan")
    values = []
    for z in range(pred.shape[0]):
        ssim_map = _ssim_map_2d(pred[z], target[z], data_range=data_range)
        if mask is None:
            values.append(ssim_map.reshape(-1))
        elif mask[z].any():
            values.append(ssim_map[mask[z]])
    if not values:
        return float("nan")
    return float(np.mean(np.concatenate(values)))


@torch.inference_mode()
def evaluate(model, loader, cfg: Config) -> Dict[str, float]:
    """Run inference over `loader` and report in-mask & full metrics (mean +/- SD).

    Mean/SD are computed across cases, ignoring any case whose mask region is
    empty (that case's mae_in/rmse_in/psnr_in/w1_in are NaN -- see `_mean_sd`), and
    SD is the sample standard deviation (ddof=1), matching the fold-level SD that
    `engine.run_cv` computes over `cv_summary.csv` via pandas.

    Also reports `infer_time_s` / `infer_time_s_sd`: the real (measured, not
    extrapolated) per-case sliding-window inference time.
    """
    from .inference import predict_volume  # lazy import: avoids a module import cycle

    model.eval()
    data_range = float(cfg.normalize.target_max - cfg.normalize.target_min)
    M = cfg.infer.eval_M
    records = getattr(loader.dataset, "data", None)  # for pid in the empty-mask warning below

    publication_keys = ["mse_mask", "psnr_mask", "ssim_mask", "mse_full", "psnr_full", "ssim_full"]
    legacy_keys = ["mae_in", "rmse_in", "psnr_in", "w1_in", "mae_full", "rmse_full", "w1_full"]
    keys = publication_keys + legacy_keys
    per_case = {k: [] for k in keys}
    infer_times = []

    n_cases = 0
    for i, batch in enumerate(loader):
        x01 = batch["image"]
        y01 = batch["target"]
        mask01 = batch["mask"]

        t0 = time.perf_counter()
        pred01 = predict_volume(model, x01, mask01, cfg)
        if pred01.is_cuda:
            torch.cuda.synchronize(pred01.device)  # accurate wall time; called once per case, not per batch
        infer_times.append(time.perf_counter() - t0)
        device = pred01.device

        pred_phys = denorm_target(pred01, cfg).clamp(cfg.normalize.target_min, cfg.normalize.target_max)
        target_phys = denorm_target(y01.to(device), cfg).clamp(cfg.normalize.target_min, cfg.normalize.target_max)
        mask_dev = mask01.to(device)

        m_mask = region_mask(mask_dev, "mask")
        m_full = region_mask(mask_dev, "full")

        if m_mask.sum() == 0:
            pid = records[i]["pid"] if records is not None else i
            print(
                f"[evaluate] warning: case {pid!r} has an empty mask region (0 voxels) -- "
                "mae_in/rmse_in/psnr_in/w1_in for this case are NaN and excluded from the mean/SD below"
            )

        per_case["mse_mask"].append(mse(pred_phys, target_phys, m_mask))
        per_case["psnr_mask"].append(psnr(pred_phys, target_phys, m_mask, data_range))
        per_case["ssim_mask"].append(ssim_volume(pred_phys, target_phys, m_mask, data_range))
        per_case["mse_full"].append(mse(pred_phys, target_phys, m_full))
        per_case["psnr_full"].append(psnr(pred_phys, target_phys, m_full, data_range))
        per_case["ssim_full"].append(ssim_volume(pred_phys, target_phys, None, data_range))

        # Backward-compatible internal diagnostics. Public summaries select only
        # the six publication metrics above.
        per_case["mae_in"].append(mae(pred_phys, target_phys, m_mask))
        per_case["rmse_in"].append(rmse(pred_phys, target_phys, m_mask))
        per_case["psnr_in"].append(psnr(pred_phys, target_phys, m_mask, data_range))
        per_case["w1_in"].append(w1(pred_phys, target_phys, m_mask, M))
        per_case["mae_full"].append(mae(pred_phys, target_phys, m_full))
        per_case["rmse_full"].append(rmse(pred_phys, target_phys, m_full))
        per_case["w1_full"].append(w1(pred_phys, target_phys, m_full, M))
        n_cases += 1

    result: Dict[str, float] = {"n_cases": float(n_cases)}
    for k, vals in per_case.items():
        result[k], result[f"{k}_sd"] = _mean_sd(vals)

    result["infer_time_s"], result["infer_time_s_sd"] = _mean_sd(infer_times)
    return result
