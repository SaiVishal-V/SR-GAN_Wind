"""
Reproducibility utilities for WindGapGAN.

Sets random seeds across all relevant libraries to ensure
deterministic training behavior.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable PyTorch deterministic algorithms.
            May reduce performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations don't support deterministic mode
            logger.warning(
                "Could not enable fully deterministic algorithms. "
                "Some operations may be non-deterministic."
            )

    logger.info("Random seed set to %d (deterministic=%s)", seed, deterministic)
