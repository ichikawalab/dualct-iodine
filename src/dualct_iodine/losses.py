# -*- coding: utf-8 -*-
"""Loss functions: masked/full L1 (main) + optional CDF/Wasserstein auxiliary loss.

The spatial region the loss is computed over (`loss.region`: "mask" or "full") and
the auxiliary term (`loss.aux`: "none" or "cdf") are both config/CLI switches,
independent of each other.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import Config


def region_mask(mask01: torch.Tensor, region: str) -> torch.Tensor:
    """Boolean mask selecting the loss/metric region.

    region == "mask": voxels where mask01 > 0.5 (the supplied lung mask).
    region == "full": every voxel (an all-ones mask of the same shape).
    """
    if region == "mask":
        return mask01 > 0.5
    if region == "full":
        return torch.ones_like(mask01, dtype=torch.bool)
    raise ValueError(f"Unknown loss/metric region: {region!r} (expected 'mask' or 'full')")


class MaskedL1(nn.Module):
    """Mean L1 (in [0,1] space) over the selected region.

    Returns a gradient-safe zero (not NaN) if the region is empty for a batch.
    """

    def __init__(self, region: str):
        super().__init__()
        self.region = region

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
        m = region_mask(mask01, self.region).to(pred01.dtype)
        if m.sum() == 0:
            return (pred01 * 0.0).sum()
        # Do not hard-clamp during training: clamp has zero gradient outside [0, 1].
        diff = (pred01 - target01).abs()
        return (m * diff).sum() / (m.sum() + 1e-6)


class CDFQuantileLoss(nn.Module):
    """1D-Wasserstein distance (fixed-quantile L1) between pred/target distributions.

    Computed in physical (iodine-equivalent HU) units within the selected region,
    then normalized by (target_max - target_min).
    """

    def __init__(self, region: str, M: int, target_min: float, target_max: float, max_voxels: int = 65536):
        super().__init__()
        self.region = region
        self.M = int(M)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.max_voxels = int(max_voxels)
        self.register_buffer("u", torch.linspace(0.0, 1.0, steps=self.M))

    @staticmethod
    def _quantiles_1d(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u = u.to(device=x.device, dtype=x.dtype)
        n = x.numel()
        if n == 0:
            return u * 0.0 + x.sum()
        xs = torch.sort(x)[0]
        if n == 1:
            return xs.new_full((u.numel(),), xs[0])
        pos = u.clamp(0, 1) * (n - 1)
        i0 = pos.floor().long().clamp(0, n - 2)
        t = pos - i0.float()
        return (1 - t) * xs[i0] + t * xs[i0 + 1]

    def _to_physical(self, x01: torch.Tensor) -> torch.Tensor:
        return x01 * (self.target_max - self.target_min) + self.target_min

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
        m = region_mask(mask01, self.region)
        if m.sum() == 0:
            return (pred01 * 0.0).sum()
        # Compute each sample independently. Flattening the complete batch couples
        # unrelated patients and makes the objective batch-composition dependent.
        p_phys_all = self._to_physical(pred01.to(torch.float32))
        t_phys_all = self._to_physical(target01.to(torch.float32)).detach()
        sample_losses = []
        for i in range(pred01.shape[0]):
            selected = m[i]
            p_phys = p_phys_all[i][selected]
            t_phys = t_phys_all[i][selected]
            n = p_phys.numel()
            if n == 0:
                continue
            if n > self.max_voxels:
                idx = torch.randperm(n, device=p_phys.device)[: self.max_voxels]
                p_phys = p_phys[idx]
                t_phys = t_phys[idx]
            qa = self._quantiles_1d(p_phys, self.u)
            qb = self._quantiles_1d(t_phys, self.u)
            sample_losses.append(torch.mean(torch.abs(qa - qb)))
        if not sample_losses:
            return (pred01 * 0.0).sum()
        loss = torch.stack(sample_losses).mean()
        return loss / float(self.target_max - self.target_min)


def build_loss(cfg: Config):
    """Build a `loss(pred01, target01, mask01) -> (total, log_dict)` callable.

    total = L1(region) + (lam_cdf * CDF(region) if loss.aux == "cdf" else 0)
    """
    l1 = MaskedL1(region=cfg.loss.region)
    cdf = None
    if cfg.loss.aux == "cdf":
        cdf = CDFQuantileLoss(
            region=cfg.loss.region,
            M=cfg.loss.cdf_M,
            target_min=cfg.normalize.target_min,
            target_max=cfg.normalize.target_max,
            max_voxels=cfg.loss.cdf_max_voxels,
        )
    elif cfg.loss.aux != "none":
        raise ValueError(f"Unknown loss.aux: {cfg.loss.aux!r} (expected 'none' or 'cdf')")
    lam_cdf = cfg.loss.lam_cdf

    def loss_fn(pred01: torch.Tensor, target01: torch.Tensor, mask01: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_val = l1(pred01, target01, mask01)
        if cdf is not None:
            cdf_val = cdf(pred01, target01, mask01)
            total = l1_val + lam_cdf * cdf_val
            return total, {"l1": float(l1_val.item()), "cdf": float(cdf_val.item()), "total": float(total.item())}
        total = l1_val
        return total, {"l1": float(l1_val.item()), "total": float(total.item())}

    return loss_fn
