# -*- coding: utf-8 -*-
"""Model construction: SwinUNETR wrapped in either a residual or a direct-prediction head.

For the cross-domain CT -> iodine task the default is direct prediction
(`residual: false`). The residual connection (`y = x + alpha * f(x)` with
zero-initialized final layer, i.e. identity-mapping warm start) is available
behind `model.residual: true` for the same-domain 120kV -> 80kV/140kV task.

Both wrappers expose the same call interface `forward(x, inference=False)` so
that `engine.py` / `inference.py` never need to branch on which mode is active.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, UNet

from .config import Config


class DirectWrapper(nn.Module):
    """y = f(x). Standard initialization; used when model.residual is False."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        y = self.base(x)
        if inference:
            y = torch.clamp(y, 0.0, 1.0)
        return y


class ResidualWrapper(nn.Module):
    """y = x + alpha * f(x). Final conv is zero-initialized -> identity warm start."""

    def __init__(self, base: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        delta = self.base(x)
        y = x + self.alpha * delta
        if inference:
            y = torch.clamp(y, 0.0, 1.0)
        return y


def _zero_init_last_conv(base: nn.Module, out_channels: int) -> None:
    # Assumes the last Conv3d whose out_channels equals the model output channels is the
    # output projection. This holds for MONAI SwinUNETR and UNet in the pinned version
    # range (monai>=1.4,<1.6); revisit if the upstream network structure changes.
    candidates = [m for m in base.modules() if isinstance(m, nn.Conv3d) and m.out_channels == out_channels]
    if not candidates:
        raise RuntimeError("Could not locate the model output convolution for residual initialization")
    output_conv = candidates[-1]
    nn.init.zeros_(output_conv.weight)
    if output_conv.bias is not None:
        nn.init.zeros_(output_conv.bias)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.name == "swinunetr":
        base = SwinUNETR(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            feature_size=cfg.model.feature_size,
            use_checkpoint=cfg.model.use_checkpoint,
            spatial_dims=cfg.model.spatial_dims,
        )
    elif cfg.model.name == "unet":
        base = UNet(
            spatial_dims=cfg.model.spatial_dims,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            channels=tuple(cfg.model.unet_channels),
            strides=tuple(cfg.model.unet_strides),
            num_res_units=cfg.model.unet_num_res_units,
            norm=cfg.model.unet_norm,
            dropout=cfg.model.unet_dropout,
        )
    else:  # guarded by Config.validate, retained for direct programmatic use
        raise ValueError(f"Unsupported model.name: {cfg.model.name!r}")
    if cfg.model.residual:
        _zero_init_last_conv(base, cfg.model.out_channels)
        return ResidualWrapper(base, alpha=cfg.model.residual_alpha)
    return DirectWrapper(base)
