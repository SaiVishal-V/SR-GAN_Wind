"""
Unit tests for WindGapGAN dataset and mask generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.mask_generator import MaskGenerator
from datasets.nc_dataset import (
    WindGapDataset,
    compute_normalization_stats,
    normalize,
    denormalize,
)


class TestMaskGenerator:
    """Tests for MaskGenerator."""

    def test_random_mask_shape(self):
        gen = MaskGenerator(strategy="random", mask_ratio=0.3, seed=42)
        mask = gen.generate(64, 64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_random_mask_ratio(self):
        gen = MaskGenerator(strategy="random", mask_ratio=0.3, seed=42)
        mask = gen.generate(100, 100)
        missing_ratio = 1.0 - mask.mean()
        assert abs(missing_ratio - 0.3) < 0.05, f"Expected ~30% missing, got {missing_ratio*100:.1f}%"

    def test_block_mask(self):
        gen = MaskGenerator(strategy="block", seed=42)
        mask = gen.generate(64, 64)
        assert mask.shape == (64, 64)
        assert (mask == 0).any(), "Block mask should have some gaps"

    def test_mixed_mask(self):
        gen = MaskGenerator(strategy="mixed", seed=42)
        mask = gen.generate(64, 64)
        assert mask.shape == (64, 64)
        assert (mask == 0).any(), "Mixed mask should have some gaps"

    def test_sequence_mask(self):
        gen = MaskGenerator(strategy="random", mask_ratio=0.3, seed=42)
        masks = gen.generate_sequence(5, 64, 64)
        assert masks.shape == (5, 64, 64)
        # At least one timestep should have gaps
        assert (masks == 0).any()

    def test_mask_values_binary(self):
        gen = MaskGenerator(strategy="mixed", seed=42)
        mask = gen.generate(64, 64)
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0}), f"Mask should be binary, got unique values: {unique}"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            MaskGenerator(strategy="invalid")


class TestNormalization:
    """Tests for normalization utilities."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(1000).astype(np.float32) * 5 + 10

    def test_zscore(self, sample_data):
        stats = compute_normalization_stats(sample_data, method="zscore")
        assert stats["method"] == "zscore"
        assert "mean" in stats
        assert "std" in stats

        normalized = normalize(sample_data, stats)
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

        denormalized = denormalize(normalized, stats)
        np.testing.assert_array_almost_equal(denormalized, sample_data, decimal=4)

    def test_minmax(self, sample_data):
        stats = compute_normalization_stats(sample_data, method="minmax")
        assert stats["method"] == "minmax"

        normalized = normalize(sample_data, stats)
        assert normalized.min() >= -0.01
        assert normalized.max() <= 1.01

        denormalized = denormalize(normalized, stats)
        np.testing.assert_array_almost_equal(denormalized, sample_data, decimal=4)

    def test_robust(self, sample_data):
        stats = compute_normalization_stats(sample_data, method="robust")
        assert stats["method"] == "robust"
        assert "median" in stats
        assert "iqr" in stats

        normalized = normalize(sample_data, stats)
        denormalized = denormalize(normalized, stats)
        np.testing.assert_array_almost_equal(denormalized, sample_data, decimal=4)

    def test_auto(self, sample_data):
        stats = compute_normalization_stats(sample_data, method="auto")
        assert stats["method"] in {"zscore", "minmax", "robust"}

    def test_none(self, sample_data):
        stats = compute_normalization_stats(sample_data, method="none")
        normalized = normalize(sample_data, stats)
        np.testing.assert_array_equal(normalized, sample_data)

    def test_nan_handling(self):
        data = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
        stats = compute_normalization_stats(data, method="zscore")
        assert not np.isnan(stats["mean"])


class TestWindGapDataset:
    """Tests for WindGapDataset."""

    @pytest.fixture
    def sample_dataset(self):
        T, H, W = 20, 64, 64
        np.random.seed(42)
        data = np.random.randn(T, H, W).astype(np.float32) * 5 + 10
        mask = np.ones_like(data)
        mask[5:8, 20:40, 20:40] = 0.0

        norm_stats = compute_normalization_stats(data[mask > 0], method="zscore")
        mask_gen = MaskGenerator(strategy="mixed", mask_ratio=0.3, seed=42)

        return WindGapDataset(
            data=data,
            real_mask=mask,
            split="train",
            sequence_length=5,
            patch_size=32,
            stride=32,
            norm_stats=norm_stats,
            mask_generator=mask_gen,
        )

    def test_dataset_length(self, sample_dataset):
        assert len(sample_dataset) > 0

    def test_sample_shape(self, sample_dataset):
        sample = sample_dataset[0]
        assert "input" in sample
        assert "target" in sample
        assert "mask" in sample

        T, C_in = 5, 2
        assert sample["input"].shape == (T, C_in, 32, 32), f"Got {sample['input'].shape}"
        assert sample["target"].shape == (T, 1, 32, 32), f"Got {sample['target'].shape}"
        assert sample["mask"].shape == (T, 1, 32, 32), f"Got {sample['mask'].shape}"

    def test_sample_dtype(self, sample_dataset):
        sample = sample_dataset[0]
        assert sample["input"].dtype == torch.float32
        assert sample["target"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32

    def test_mask_binary(self, sample_dataset):
        sample = sample_dataset[0]
        mask = sample["mask"].numpy()
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0})
