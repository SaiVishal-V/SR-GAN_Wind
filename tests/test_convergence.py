import numpy as np
import pytest

from utils.config import ConvergenceConfig
from utils.convergence import ConvergenceDetector


def create_detector(window_size=10, min_epoch=10, req_consecutive=1) -> ConvergenceDetector:
    config = ConvergenceConfig(
        enabled=True,
        minimum_epoch_before_detection=min_epoch,
        window_size=window_size,
        slope_threshold=1e-4,
        variance_threshold=1e-5,
        relative_improvement_threshold=1e-3,
        smoothing_method="ema",
        ema_alpha=1.0, # Disable EMA for exact test values
        required_consecutive_windows=req_consecutive,
        max_plateau_cycles=3
    )
    return ConvergenceDetector(config)


def test_flat_curve_converges():
    detector = create_detector()
    
    level = 0
    promoted = False
    for epoch in range(15):
        # Perfectly flat at 0.1
        val_loss = 0.1
        grad_norm = 0.001
        level, promoted = detector.step(epoch, val_loss, grad_norm)
        
        if promoted:
            assert epoch >= 10 # Cannot promote before min_epoch
            assert level == 1
            break
            
    assert promoted is True


def test_noisy_curve_does_not_converge():
    detector = create_detector()
    
    np.random.seed(42)
    for epoch in range(20):
        # Noise around 0.5, variance will be way above 1e-5
        val_loss = 0.5 + np.random.normal(0, 0.1)
        grad_norm = 1.0
        level, promoted = detector.step(epoch, val_loss, grad_norm)
        assert promoted is False


def test_slowly_descending_curve_does_not_converge():
    detector = create_detector()
    
    for epoch in range(20):
        # Decreasing by 0.01 each step. Slope is -0.01, which is > 1e-4
        val_loss = 1.0 - (epoch * 0.01)
        grad_norm = 0.1
        level, promoted = detector.step(epoch, val_loss, grad_norm)
        assert promoted is False


def test_oscillating_curve_does_not_converge():
    detector = create_detector()
    
    for epoch in range(20):
        # Oscillating significantly
        val_loss = 0.5 + 0.1 * np.sin(epoch)
        grad_norm = 0.1
        level, promoted = detector.step(epoch, val_loss, grad_norm)
        assert promoted is False


def test_sharp_drop_followed_by_flat_region():
    detector = create_detector(window_size=5, min_epoch=5, req_consecutive=2)
    
    promoted = False
    for epoch in range(20):
        if epoch < 5:
            # Sharp drop
            val_loss = 1.0 - (epoch * 0.1)
        else:
            # Flat region
            val_loss = 0.5
            
        level, promoted = detector.step(epoch, val_loss, grad_norm=0.01)
        
        if promoted:
            assert epoch >= 10 # 5 for sharp drop, then needs 2 consecutive windows of 5? No, sliding window.
            # Window becomes flat at epoch 9 (epochs 5,6,7,8,9). Then consecutive=2 means epoch 10.
            assert level == 1
            break
            
    assert promoted is True


def test_multi_plateau_levels():
    detector = create_detector(window_size=5, min_epoch=5, req_consecutive=1)
    
    # Plateau 1
    for epoch in range(10):
        level, promoted = detector.step(epoch, val_loss=0.5, grad_norm=0.01)
        if promoted:
            assert level == 1
            
    # Need 5 more epochs to fill the next window with flat data
    for epoch in range(10, 15):
        level, promoted = detector.step(epoch, val_loss=0.4, grad_norm=0.01)
        if promoted:
            assert level == 2
            
    # Plateau 3
    for epoch in range(15, 20):
        level, promoted = detector.step(epoch, val_loss=0.3, grad_norm=0.01)
        if promoted:
            assert level == 3
