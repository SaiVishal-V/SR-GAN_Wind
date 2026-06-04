"""WindGapGAN loss functions."""

from losses.masked_l1 import MaskedL1Loss
from losses.gradient_loss import GradientLoss
from losses.adversarial import AdversarialLoss
from losses.perceptual_loss import PerceptualLoss
from losses.spectral_loss import SpectralLoss

__all__ = [
    "MaskedL1Loss",
    "GradientLoss",
    "AdversarialLoss",
    "PerceptualLoss",
    "SpectralLoss",
]
