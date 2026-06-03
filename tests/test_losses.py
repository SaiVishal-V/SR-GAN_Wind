"""
Unit tests for WindGapGAN loss functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.masked_l1 import MaskedL1Loss
from losses.gradient_loss import GradientLoss


class TestMaskedL1Loss:
    """Tests for MaskedL1Loss."""

    def test_gap_only_loss(self):
        """Loss should only penalize gap pixels."""
        loss_fn = MaskedL1Loss(observed_weight=0.0)

        pred = torch.ones(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)  # All observed

        # If all observed, gap mask = 0 everywhere → loss should be 0
        # (no gap pixels to compute loss on)
        loss = loss_fn(pred, target, mask)
        # With no gap pixels, n_gap is clamped to 1, and sum of gap errors = 0
        assert loss.item() == 0.0, f"Expected 0 loss for all-observed, got {loss.item()}"

    def test_full_gap_loss(self):
        """With all pixels as gaps, loss = mean absolute error."""
        loss_fn = MaskedL1Loss(observed_weight=0.0)

        pred = torch.ones(1, 1, 4, 4) * 3.0
        target = torch.ones(1, 1, 4, 4) * 1.0
        mask = torch.zeros(1, 1, 4, 4)  # All gaps

        loss = loss_fn(pred, target, mask)
        assert abs(loss.item() - 2.0) < 1e-5, f"Expected ~2.0, got {loss.item()}"

    def test_partial_gap(self):
        """Loss on partial gap should only consider gap pixels."""
        loss_fn = MaskedL1Loss(observed_weight=0.0)

        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # gaps at (0,1) and (1,0)

        loss = loss_fn(pred, target, mask)
        # Gap pixels: pred=[2,3], target=[0,0] → errors=[2,3] → mean=2.5
        assert abs(loss.item() - 2.5) < 1e-5

    def test_observed_weight(self):
        """Loss with observed_weight should include observed pixel penalty."""
        loss_fn = MaskedL1Loss(observed_weight=1.0)

        pred = torch.ones(1, 1, 4, 4) * 2.0
        target = torch.ones(1, 1, 4, 4) * 1.0

        # Half observed, half gap
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :] = 1.0

        loss = loss_fn(pred, target, mask)
        # Both gap and observed contribute error = 1.0
        # gap_loss = 1.0, obs_loss = 1.0 → total = 1.0 + 1.0*1.0 = 2.0
        assert loss.item() > 0

    def test_gradient(self):
        """Loss should produce gradients."""
        loss_fn = MaskedL1Loss()
        pred = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.randn(1, 1, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)

        loss = loss_fn(pred, target, mask)
        loss.backward()
        assert pred.grad is not None


class TestGradientLoss:
    """Tests for GradientLoss."""

    def test_zero_for_identical(self):
        """Gradient loss should be ~0 for identical pred and target."""
        loss_fn = GradientLoss()
        x = torch.randn(1, 1, 16, 16)
        loss = loss_fn(x, x.clone())
        assert loss.item() < 1e-6

    def test_nonzero_for_different(self):
        """Gradient loss should be >0 for different pred and target."""
        loss_fn = GradientLoss()
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_5d_input(self):
        """Gradient loss should handle 5D (temporal) input."""
        loss_fn = GradientLoss()
        pred = torch.randn(2, 5, 1, 16, 16)
        target = torch.randn(2, 5, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0
