# -*- coding: utf-8 -*-
"""Model construction: SwinUNETR / UNet wrapped in either a residual or a direct-prediction head.

For the cross-domain CT -> iodine task the default is direct prediction
(`residual: false`). The residual connection (`y = x + alpha * gate(f(x))` with a
zero-initialized gate, i.e. identity-mapping warm start) is available behind
`model.residual: true` for the same-domain 120kV -> 80kV/140kV task.

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
    """y = x + alpha * gate(f(x)), where gate is a zero-initialized 1x1x1 convolution.

    The base network keeps its standard initialization; only the gate starts at
    zero, so the model is exactly the identity at initialization and, under an
    adaptive optimizer, moves away from it by roughly one learning-rate step per
    update. The gate sits outside the base network on purpose: zeroing a layer
    *inside* the network is identity only at step 0 when a normalization layer
    or an identity skip follows it. MONAI UNet's top level is
    "ConvTranspose3d -> InstanceNorm -> PReLU -> ResidualUnit(conv(x') + x')";
    after the first optimizer step InstanceNorm rescales the (formerly zero)
    transposed-conv output to unit variance and the identity skip passes it
    straight to the output, so f(x) jumps to O(1) (measured mean|f(x)| ~ 0.34 on
    a [0,1] input). Gating f(x) as a whole avoids that for any base architecture.
    """

    def __init__(self, base: nn.Module, out_channels: int, alpha: float = 1.0, spatial_dims: int = 3):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.gate = conv(out_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        delta = self.gate(self.base(x))
        y = x + self.alpha * delta
        if inference:
            y = torch.clamp(y, 0.0, 1.0)
        return y


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
        return ResidualWrapper(
            base,
            out_channels=cfg.model.out_channels,
            alpha=cfg.model.residual_alpha,
            spatial_dims=cfg.model.spatial_dims,
        )
    return DirectWrapper(base)
