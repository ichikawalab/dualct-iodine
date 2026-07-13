# -*- coding: utf-8 -*-
import pytest
import torch

from dualct_iodine.config import Config
from dualct_iodine.losses import CDFQuantileLoss, MaskedL1, build_loss, region_mask


def test_region_mask_full_is_all_true():
    mask01 = torch.zeros(1, 1, 2, 2, 2)
    m = region_mask(mask01, "full")
    assert bool(m.all())
    assert m.shape == mask01.shape


def test_region_mask_mask_thresholds_at_half():
    mask01 = torch.tensor([0.0, 0.6, 1.0]).view(1, 1, 1, 1, 3)
    m = region_mask(mask01, "mask")
    assert m.tolist() == [[[[[False, True, True]]]]]


def test_region_mask_invalid_raises():
    with pytest.raises(ValueError):
        region_mask(torch.zeros(1, 1, 1, 1, 1), "bogus")


def test_masked_l1_zero_and_gradient_safe_when_region_empty():
    pred = torch.rand(1, 1, 2, 2, 2, requires_grad=True)
    target = torch.rand(1, 1, 2, 2, 2)
    mask = torch.zeros(1, 1, 2, 2, 2)
    loss = MaskedL1(region="mask")(pred, target, mask)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_masked_l1_matches_manual_computation():
    pred = torch.tensor([0.2, 0.8]).view(1, 1, 1, 1, 2)
    target = torch.tensor([0.0, 1.0]).view(1, 1, 1, 1, 2)
    mask = torch.ones(1, 1, 1, 1, 2)
    loss = MaskedL1(region="mask")(pred, target, mask)
    assert loss.item() == pytest.approx(0.2, abs=1e-5)


def test_masked_l1_full_region_ignores_mask_values():
    pred = torch.tensor([0.2, 0.8]).view(1, 1, 1, 1, 2)
    target = torch.tensor([0.0, 1.0]).view(1, 1, 1, 1, 2)
    mask = torch.zeros(1, 1, 1, 1, 2)  # would zero out everything in "mask" mode
    loss = MaskedL1(region="full")(pred, target, mask)
    assert loss.item() == pytest.approx(0.2, abs=1e-5)


def test_cdf_quantile_loss_zero_for_identical_distributions():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 4, 4, 4)
    mask = torch.ones_like(x)
    loss_fn = CDFQuantileLoss(region="full", M=64, target_min=0, target_max=200)
    val = loss_fn(x, x, mask)
    assert val.item() == pytest.approx(0.0, abs=1e-5)


def test_cdf_quantile_loss_zero_grad_safe_when_region_empty():
    pred = torch.rand(1, 1, 4, 4, 4, requires_grad=True)
    target = torch.rand(1, 1, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, 4)
    loss_fn = CDFQuantileLoss(region="mask", M=32, target_min=0, target_max=200)
    val = loss_fn(pred, target, mask)
    assert val.item() == pytest.approx(0.0)
    val.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("region", ["mask", "full"])
@pytest.mark.parametrize("aux", ["none", "cdf"])
def test_build_loss_region_and_aux_combinations(region, aux):
    cfg = Config()
    cfg.loss.region = region
    cfg.loss.aux = aux
    loss_fn = build_loss(cfg)

    pred = torch.rand(1, 1, 4, 4, 4)
    target = torch.rand(1, 1, 4, 4, 4)
    mask = (torch.rand(1, 1, 4, 4, 4) > 0.5).float()

    total, logs = loss_fn(pred, target, mask)
    assert torch.isfinite(total)
    assert "l1" in logs and "total" in logs
    if aux == "cdf":
        assert "cdf" in logs
    else:
        assert "cdf" not in logs


def test_build_loss_rejects_unknown_aux():
    cfg = Config()
    cfg.loss.aux = "bogus"
    with pytest.raises(ValueError):
        build_loss(cfg)
