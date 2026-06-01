"""
Reproducibility utilities.
Sets seeds for random, numpy, torch, and cuda.
Enables deterministic training when requested.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enable PyTorch deterministic algorithms.
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
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
