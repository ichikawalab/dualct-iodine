# -*- coding: utf-8 -*-
"""Direct unit tests for model construction (no training needed).

Verifies the invariant the residual design depends on: at initialization
(zero-initialized final conv), the residual model must be the identity
(y == x), while the direct-prediction model must not be.
"""
from __future__ import annotations

import torch

from dualct_iodine.config import Config
from dualct_iodine.model import DirectWrapper, ResidualWrapper, build_model

# Divisible by 32 (SwinUNETR's downsampling requirement).
_X = torch.rand(1, 1, 32, 64, 64)


def _small_cfg(residual: bool) -> Config:
    cfg = Config()
    cfg.model.feature_size = 12  # small network for a fast CPU test
    cfg.model.residual = residual
    return cfg


def test_residual_model_is_identity_at_init():
    model = build_model(_small_cfg(residual=True))
    assert isinstance(model, ResidualWrapper)
    model.eval()
    with torch.no_grad():
        y = model(_X, inference=False)
    assert torch.allclose(y, _X, atol=1e-6)


def test_residual_model_inference_clamp_still_identity_in_range():
    model = build_model(_small_cfg(residual=True))
    model.eval()
    with torch.no_grad():
        y = model(_X, inference=True)
    # _X is already in [0,1], so clamping at init (y == x) must be a no-op.
    assert torch.allclose(y, _X, atol=1e-6)


def test_direct_model_is_not_identity_at_init():
    model = build_model(_small_cfg(residual=False))
    assert isinstance(model, DirectWrapper)
    model.eval()
    with torch.no_grad():
        y = model(_X, inference=False)
    # Standard (non-zero) initialization: output should not equal the input.
    assert not torch.allclose(y, _X, atol=1e-3)


def test_direct_model_inference_clamps_to_unit_range():
    model = build_model(_small_cfg(residual=False))
    model.eval()
    with torch.no_grad():
        y = model(_X, inference=True)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_unet_is_selectable_and_preserves_shape():
    cfg = Config()
    cfg.model.name = "unet"
    cfg.model.unet_channels = [4, 8, 16]
    cfg.model.unet_strides = [2, 2]
    cfg.train.roi_size = [16, 16, 16]
    cfg.validate()
    model = build_model(cfg).eval()
    x = torch.rand(1, 1, 16, 16, 16)
    with torch.no_grad():
        assert model(x, inference=True).shape == x.shape
