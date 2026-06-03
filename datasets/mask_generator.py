"""
Mask generation for WindGapGAN.

Handles two distinct concerns:
    1. Detection of real missing values in the dataset (observation mask).
    2. Generation of synthetic masks for training (data augmentation).

The observation mask is derived from the data itself.
Synthetic masks are used during training to create artificial gaps
so the model can learn to reconstruct missing regions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MaskGenerator:
    """
    Generate synthetic observation masks for training.

    During training, we need ground truth to supervise the model.
    We take observed (complete) patches, apply synthetic masks to
    create artificial gaps, and train the model to reconstruct them.

    Strategies:
        - 'random': Random pixel dropout with configurable ratio.
        - 'block': Random rectangular blocks of missing data.
        - 'real': Sample from actual missing patterns in the data.
        - 'mixed': Combination of random + block (default).

    Mask convention:
        1 = observed (valid)
        0 = missing (to be reconstructed)
    """

    def __init__(
        self,
        strategy: str = "mixed",
        mask_ratio: float = 0.3,
        block_size_range: tuple[int, int] = (8, 32),
        num_blocks_range: tuple[int, int] = (1, 5),
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            strategy: Masking strategy ('random', 'block', 'real', 'mixed').
            mask_ratio: Target fraction of pixels to mask (for 'random').
            block_size_range: (min, max) block size for 'block' strategy.
            num_blocks_range: (min, max) number of blocks for 'block' strategy.
            seed: Random seed for reproducibility.
        """
        valid_strategies = {"random", "block", "real", "mixed"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid mask strategy '{strategy}'. Choose from {valid_strategies}")

        self.strategy = strategy
        self.mask_ratio = mask_ratio
        self.block_size_range = block_size_range
        self.num_blocks_range = num_blocks_range
        self.rng = np.random.RandomState(seed)

        logger.info(
            "MaskGenerator initialized: strategy=%s, ratio=%.2f",
            strategy,
            mask_ratio,
        )

    def generate(self, height: int, width: int) -> np.ndarray:
        """
        Generate a synthetic mask for a single spatial frame.

        Args:
            height: Spatial height.
            width: Spatial width.

        Returns:
            Binary mask of shape (H, W), dtype float32.
            1 = observed, 0 = missing.
        """
        if self.strategy == "random":
            return self._random_mask(height, width)
        elif self.strategy == "block":
            return self._block_mask(height, width)
        elif self.strategy == "mixed":
            return self._mixed_mask(height, width)
        elif self.strategy == "real":
            # For 'real' strategy, masks are sampled from actual data
            # and injected externally. Fall back to mixed.
            return self._mixed_mask(height, width)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def generate_sequence(self, seq_len: int, height: int, width: int) -> np.ndarray:
        """
        Generate synthetic masks for a temporal sequence.

        Masks may be correlated across time (realistic cloud patterns
        tend to persist for multiple timesteps).

        Args:
            seq_len: Number of timesteps.
            height: Spatial height.
            width: Spatial width.

        Returns:
            Binary mask of shape (T, H, W), dtype float32.
        """
        masks = np.ones((seq_len, height, width), dtype=np.float32)

        # Decide how many timesteps to mask (at least 1, at most all)
        num_masked_steps = max(1, self.rng.randint(1, seq_len + 1))
        masked_indices = self.rng.choice(seq_len, size=num_masked_steps, replace=False)

        for t in masked_indices:
            masks[t] = self.generate(height, width)

        return masks

    def _random_mask(self, height: int, width: int) -> np.ndarray:
        """Random pixel dropout."""
        mask = np.ones((height, width), dtype=np.float32)
        dropout = self.rng.random((height, width)) < self.mask_ratio
        mask[dropout] = 0.0
        return mask

    def _block_mask(self, height: int, width: int) -> np.ndarray:
        """Random rectangular blocks of missing data."""
        mask = np.ones((height, width), dtype=np.float32)
        num_blocks = self.rng.randint(self.num_blocks_range[0], self.num_blocks_range[1] + 1)

        for _ in range(num_blocks):
            bh = self.rng.randint(self.block_size_range[0], min(self.block_size_range[1] + 1, height + 1))
            bw = self.rng.randint(self.block_size_range[0], min(self.block_size_range[1] + 1, width + 1))
            top = self.rng.randint(0, max(1, height - bh + 1))
            left = self.rng.randint(0, max(1, width - bw + 1))
            mask[top : top + bh, left : left + bw] = 0.0

        return mask

    def _mixed_mask(self, height: int, width: int) -> np.ndarray:
        """Combination of random pixel dropout + block masks."""
        # 50% chance of each strategy, then combine
        mask = np.ones((height, width), dtype=np.float32)

        if self.rng.random() < 0.5:
            # Random dropout with reduced ratio
            dropout = self.rng.random((height, width)) < (self.mask_ratio * 0.5)
            mask[dropout] = 0.0

        # Always add at least one block
        num_blocks = self.rng.randint(1, self.num_blocks_range[1] + 1)
        for _ in range(num_blocks):
            bh = self.rng.randint(
                self.block_size_range[0],
                min(self.block_size_range[1] + 1, height + 1),
            )
            bw = self.rng.randint(
                self.block_size_range[0],
                min(self.block_size_range[1] + 1, width + 1),
            )
            top = self.rng.randint(0, max(1, height - bh + 1))
            left = self.rng.randint(0, max(1, width - bw + 1))
            mask[top : top + bh, left : left + bw] = 0.0

        return mask
