"""
Perceptual (Feature-Matching) loss for WindGapGAN.

Computes L1 distance between intermediate discriminator features
for real and generated images.  This forces the generator to produce
outputs that look similar to real data at multiple abstraction levels,
which is critical for learning textures and spatial patterns.

Reference:
    Larsen et al., "Autoencoding beyond pixels using a learned similarity metric"
    Wang & Gupta, "Generative Image Inpainting with Contextual Attention"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """
    Feature-matching loss using discriminator intermediate features.

    loss = Σᵢ wᵢ * || D_feat_i(fake) - D_feat_i(real) ||₁

    Args:
        n_layers: Number of feature layers to match.
        weights: Per-layer weights (default: equal weighting).
    """

    def __init__(
        self,
        n_layers: int = 4,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if weights is None:
            # Equal weights, normalized by number of layers
            self.weights = [1.0 / n_layers] * n_layers
        else:
            assert len(weights) == n_layers
            self.weights = weights

    def forward(
        self,
        fake_features: list[torch.Tensor],
        real_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature-matching loss.

        Args:
            fake_features: List of discriminator features for generated images.
            real_features: List of discriminator features for real images.
                Both lists should have the same length and tensor shapes.

        Returns:
            Scalar loss.
        """
        loss = torch.tensor(0.0, device=fake_features[0].device)

        n = min(len(fake_features), len(real_features), len(self.weights))
        for i in range(n):
            # Detach real features — we only want gradients through generator
            loss = loss + self.weights[i] * nn.functional.l1_loss(
                fake_features[i], real_features[i].detach()
            )

        return loss
