"""WindGapGAN loss functions."""

from losses.masked_l1 import MaskedL1Loss
from losses.gradient_loss import GradientLoss

__all__ = ["MaskedL1Loss", "GradientLoss"]
