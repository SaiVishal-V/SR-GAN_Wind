"""
Unit tests for WindGapGAN evaluation metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluators.metrics import (
    GapFillingMetrics,
    compute_gap_metrics,
    compute_observed_metrics,
    compute_stratified_metrics,
    compute_all_metrics,
)


class TestGapFillingMetrics:
    """Tests for individual metric functions."""

    @pytest.fixture
    def perfect_data(self):
        """Prediction = Target everywhere."""
        pred = np.random.randn(100).astype(np.float32)
        target = pred.copy()
        mask = np.ones(100, dtype=np.float32)
        mask[50:] = 0  # 50% gap
        return pred, target, mask

    @pytest.fixture
    def imperfect_data(self):
        """Prediction != Target."""
        np.random.seed(42)
        target = np.random.randn(100).astype(np.float32) * 5 + 10
        pred = target + np.random.randn(100).astype(np.float32) * 0.5
        mask = np.ones(100, dtype=np.float32)
        mask[50:] = 0
        return pred, target, mask

    def test_rmse_perfect(self, perfect_data):
        pred, target, mask = perfect_data
        gap_mask = 1.0 - mask
        rmse = GapFillingMetrics.rmse(pred, target, gap_mask)
        assert abs(rmse) < 1e-6, f"RMSE should be ~0 for perfect prediction, got {rmse}"

    def test_mae_perfect(self, perfect_data):
        pred, target, mask = perfect_data
        gap_mask = 1.0 - mask
        mae = GapFillingMetrics.mae(pred, target, gap_mask)
        assert abs(mae) < 1e-6

    def test_bias_perfect(self, perfect_data):
        pred, target, mask = perfect_data
        gap_mask = 1.0 - mask
        bias = GapFillingMetrics.bias(pred, target, gap_mask)
        assert abs(bias) < 1e-6

    def test_correlation_perfect(self, perfect_data):
        pred, target, mask = perfect_data
        gap_mask = 1.0 - mask
        corr = GapFillingMetrics.correlation(pred, target, gap_mask)
        assert abs(corr - 1.0) < 1e-5, f"Correlation should be ~1.0, got {corr}"

    def test_rmse_positive(self, imperfect_data):
        pred, target, mask = imperfect_data
        gap_mask = 1.0 - mask
        rmse = GapFillingMetrics.rmse(pred, target, gap_mask)
        assert rmse > 0

    def test_empty_region(self):
        """Metrics should return NaN for empty regions."""
        pred = np.random.randn(10)
        target = np.random.randn(10)
        mask = np.zeros(10)  # No pixels in region
        rmse = GapFillingMetrics.rmse(pred, target, mask)
        assert np.isnan(rmse)


class TestMetricGroups:
    """Tests for metric group computation functions."""

    def test_compute_gap_metrics(self):
        np.random.seed(42)
        pred = np.random.randn(50).astype(np.float32)
        target = np.random.randn(50).astype(np.float32)
        mask = np.ones(50, dtype=np.float32)
        mask[25:] = 0

        metrics = compute_gap_metrics(pred, target, mask)
        assert "rmse_gap" in metrics
        assert "mae_gap" in metrics
        assert "bias_gap" in metrics
        assert "corr_gap" in metrics
        assert "kl_div_gap" in metrics
        assert "wasserstein_gap" in metrics

    def test_compute_observed_metrics(self):
        np.random.seed(42)
        pred = np.random.randn(50).astype(np.float32)
        target = pred.copy()  # Perfect on observed
        mask = np.ones(50, dtype=np.float32)
        mask[25:] = 0

        metrics = compute_observed_metrics(pred, target, mask)
        assert abs(metrics["rmse_observed"]) < 1e-5
        assert abs(metrics["mae_observed"]) < 1e-5

    def test_stratified_metrics(self):
        np.random.seed(42)
        target = np.random.uniform(0, 20, 200).astype(np.float32)
        pred = target + np.random.randn(200).astype(np.float32) * 0.5
        mask = np.ones(200, dtype=np.float32)
        mask[100:] = 0

        regimes = [[0, 5], [5, 10], [10, 15], [15, None]]
        results = compute_stratified_metrics(pred, target, mask, regimes)

        assert "0-5" in results
        assert "5-10" in results
        assert "10-15" in results
        assert "15-inf" in results

    def test_compute_all_metrics(self):
        np.random.seed(42)
        pred = np.random.randn(100).astype(np.float32)
        target = np.random.randn(100).astype(np.float32)
        mask = np.ones(100, dtype=np.float32)
        mask[50:] = 0

        result = compute_all_metrics(pred, target, mask)
        assert "gap" in result
        assert "observed" in result
        assert "stratified" in result
