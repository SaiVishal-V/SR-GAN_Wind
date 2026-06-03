"""
Advanced Convergence Detection for WindGapGAN.

Replaces basic early stopping with mathematical convergence detection:
- Validates trends via linear regression slope.
- Validates stability via variance.
- Validates saturation via relative improvement.
- Tracks multiple plateau cycles and triggers SGDR restarts.
- Monitors gradient norm for optimization flatness.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from utils.config import ConvergenceConfig

logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """
    State machine for tracking mathematical convergence of the loss curve.
    
    Levels:
        0: Normal training.
        1: First plateau detected (Trigger SGDR 1).
        2: Second plateau detected (Trigger SGDR 2).
        3: Third plateau detected (Trigger SGDR 3).
        4: True convergence (Training exhausted).
    """

    def __init__(self, config: ConvergenceConfig) -> None:
        self.config = config
        
        self.level = 0
        self.plateau_count = 0
        self.best_loss_ever = float("inf")
        self.best_previous_window_mean = float("inf")
        
        self.consecutive_converged_windows = 0
        
        self.losses: deque[float] = deque(maxlen=config.window_size)
        self.smoothed_losses: deque[float] = deque(maxlen=config.window_size)
        self.grad_norms: deque[float] = deque(maxlen=config.window_size)
        
        self.last_smoothed_loss = None

    def step(self, epoch: int, val_loss: float, grad_norm: float) -> tuple[int, bool]:
        """
        Record validation loss and gradient norm, and evaluate convergence.
        
        Args:
            epoch: Current epoch.
            val_loss: Validation loss for the current epoch.
            grad_norm: Average gradient norm for the epoch.
            
        Returns:
            Tuple of (current_level, just_promoted_flag).
        """
        if not self.config.enabled:
            return 0, False

        # EMA Smoothing
        if self.last_smoothed_loss is None:
            smoothed = val_loss
        else:
            alpha = self.config.ema_alpha
            smoothed = alpha * val_loss + (1 - alpha) * self.last_smoothed_loss
            
        self.last_smoothed_loss = smoothed
        
        self.losses.append(val_loss)
        self.smoothed_losses.append(smoothed)
        self.grad_norms.append(grad_norm)
        
        if val_loss < self.best_loss_ever:
            self.best_loss_ever = val_loss
            
        # Do not detect during warmup
        if epoch < self.config.minimum_epoch_before_detection:
            return self.level, False
            
        # Need a full window
        if len(self.smoothed_losses) < self.config.window_size:
            return self.level, False

        # Calculate metrics
        y = np.array(self.smoothed_losses)
        x = np.arange(len(y))
        
        # 1. Slope (Trend)
        # Linear regression: slope = cov(x, y) / var(x)
        slope = np.cov(x, y)[0, 1] / np.var(x)
        
        # 2. Variance (Oscillation)
        variance = np.var(y)
        
        # 3. Relative Improvement
        current_mean = np.mean(y)
        if self.best_previous_window_mean != float("inf"):
            relative_improvement = (self.best_previous_window_mean - current_mean) / (self.best_previous_window_mean + 1e-8)
        else:
            relative_improvement = 1.0 # First window
            
        self.best_previous_window_mean = min(self.best_previous_window_mean, current_mean)
        
        # 4. Gradient Norm (Stability)
        mean_grad = np.mean(self.grad_norms)
        
        # Convergence conditions
        trend_flat = abs(slope) < self.config.slope_threshold
        low_variance = variance < self.config.variance_threshold
        saturated = relative_improvement < self.config.relative_improvement_threshold
        
        # Using slope, variance and relative improvement as per review
        is_converged = trend_flat and low_variance and saturated
        
        if is_converged:
            self.consecutive_converged_windows += 1
        else:
            self.consecutive_converged_windows = 0
            
        # Promote level if consecutive windows are converged
        just_promoted = False
        if self.consecutive_converged_windows >= self.config.required_consecutive_windows:
            self.plateau_count += 1
            self.level = min(self.plateau_count, self.config.max_plateau_cycles + 1)
            
            logger.info(
                "Convergence detected! Level %d -> %d. "
                "(Slope: %.2e, Var: %.2e, Rel_Imp: %.2e, Grad_Norm: %.2e)",
                self.level - 1, self.level, slope, variance, relative_improvement, mean_grad
            )
            
            # Reset window to avoid immediate re-trigger
            self.losses.clear()
            self.smoothed_losses.clear()
            self.grad_norms.clear()
            self.last_smoothed_loss = None
            self.consecutive_converged_windows = 0
            just_promoted = True
            
        return self.level, just_promoted
