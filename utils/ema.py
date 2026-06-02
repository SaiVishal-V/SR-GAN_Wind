"""
Exponential Moving Average (EMA) for generator weights.

Maintains a shadow copy of model parameters as an exponential moving average,
which typically provides better generalization than the final trained weights.

Usage:
    ema = EMAGenerator(generator, decay=0.999)

    for batch in train_loader:
        # ... train generator ...
        ema.update(generator)

    # Evaluate with EMA weights
    ema.apply_shadow(generator)
    val_metrics = validate(generator, ...)
    ema.restore(generator)
"""

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMAGenerator:
    """
    Exponential Moving Average of generator weights.

    shadow_param = decay * shadow_param + (1 - decay) * model_param

    Args:
        model: Generator model to track.
        decay: EMA decay rate. Higher = smoother (0.999 typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow with current model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with current model parameters.

        Args:
            model: Generator model with updated weights.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Swap model parameters with shadow (EMA) parameters.
        Call restore() to undo this.

        Args:
            model: Generator model to apply EMA weights to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """
        Restore model parameters from backup (undo apply_shadow).

        Args:
            model: Generator model to restore original weights to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
