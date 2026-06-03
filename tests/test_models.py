"""
Unit tests for WindGapGAN models.

Tests:
    - MaskedUNet forward pass shape
    - Observed pixel preservation
    - Classical baselines
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet import MaskedUNet
from models.classical_baselines import (
    PersistenceBaseline,
    LinearInterpolationBaseline,
    NearestNeighborBaseline,
    MeanFillingBaseline,
    run_all_baselines,
)


class TestMaskedUNet:
    """Tests for the MaskedUNet model."""

    def test_forward_shape_4d(self):
        """Test output shape for 4D input (single frame)."""
        model = MaskedUNet(in_channels=2, out_channels=1, base_features=16, depth=3)
        x = torch.randn(2, 2, 64, 64)  # (B, C, H, W)
        out = model(x)
        assert out.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {out.shape}"

    def test_forward_shape_5d(self):
        """Test output shape for 5D input (temporal sequence)."""
        model = MaskedUNet(in_channels=2, out_channels=1, base_features=16, depth=3)
        x = torch.randn(2, 5, 2, 64, 64)  # (B, T, C, H, W)
        out = model(x)
        assert out.shape == (2, 5, 1, 64, 64), f"Expected (2,5,1,64,64), got {out.shape}"

    def test_observed_pixels_preserved(self):
        """Test that observed pixels are preserved exactly (hard constraint)."""
        model = MaskedUNet(in_channels=2, out_channels=1, base_features=16, depth=3)
        model.eval()

        B, T, H, W = 1, 3, 64, 64
        field = torch.randn(B, T, 1, H, W)
        mask = torch.ones(B, T, 1, H, W)  # All observed
        # Set some pixels as gaps
        mask[:, :, :, 10:30, 10:30] = 0.0
        input_field = field * mask

        x = torch.cat([input_field, mask], dim=2)  # (B, T, 2, H, W)

        with torch.no_grad():
            out = model(x)

        # Where mask=1, output should equal input field
        observed = mask.bool()
        assert torch.allclose(
            out[observed], input_field[observed], atol=1e-5
        ), "Observed pixels were not preserved!"

    def test_different_depths(self):
        """Test model works with different depths."""
        for depth in [2, 3, 4]:
            model = MaskedUNet(in_channels=2, out_channels=1, base_features=16, depth=depth)
            x = torch.randn(1, 2, 64, 64)
            out = model(x)
            assert out.shape == (1, 1, 64, 64)


class TestClassicalBaselines:
    """Tests for classical gap-filling baselines."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with gaps."""
        T, H, W = 10, 32, 32
        np.random.seed(42)
        data = np.random.randn(T, H, W).astype(np.float32) * 5 + 10  # wind-like values
        mask = np.ones_like(data)
        # Create some gaps
        mask[3:6, 10:20, 10:20] = 0.0  # Block gap
        data_with_gaps = data.copy()
        data_with_gaps[mask == 0] = np.nan
        return data_with_gaps, mask, data

    def test_persistence(self, sample_data):
        data, mask, truth = sample_data
        filled = PersistenceBaseline.fill(data, mask)
        assert not np.isnan(filled).any(), "Persistence left NaN values"
        assert filled.shape == data.shape

    def test_linear_interpolation(self, sample_data):
        data, mask, truth = sample_data
        filled = LinearInterpolationBaseline.fill(data, mask)
        assert not np.isnan(filled).any(), "Linear interp left NaN values"
        assert filled.shape == data.shape

    def test_nearest_neighbor(self, sample_data):
        data, mask, truth = sample_data
        filled = NearestNeighborBaseline.fill(data, mask)
        assert not np.isnan(filled).any(), "Nearest neighbor left NaN values"
        assert filled.shape == data.shape

    def test_mean_filling(self, sample_data):
        data, mask, truth = sample_data
        filled = MeanFillingBaseline.fill(data, mask)
        assert not np.isnan(filled).any(), "Mean filling left NaN values"
        assert filled.shape == data.shape

    def test_observed_pixels_unchanged(self, sample_data):
        """All baselines must preserve observed pixels."""
        data, mask, truth = sample_data
        for name, baseline_cls in [
            ("Persistence", PersistenceBaseline),
            ("Nearest Neighbor", NearestNeighborBaseline),
            ("Mean Filling", MeanFillingBaseline),
        ]:
            filled = baseline_cls.fill(data, mask)
            observed = mask > 0
            np.testing.assert_array_almost_equal(
                filled[observed],
                data[observed],
                decimal=5,
                err_msg=f"{name} modified observed pixels!",
            )

    def test_run_all_baselines(self, sample_data):
        data, mask, truth = sample_data
        results = run_all_baselines(data, mask)
        assert len(results) == 4
        for name, filled in results.items():
            assert not np.isnan(filled).any(), f"{name} left NaN values"
