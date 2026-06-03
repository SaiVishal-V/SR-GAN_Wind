"""
Adversarial losses for GAN training (Phase 3).

Supports multiple GAN loss formulations:
    - vanilla: Binary cross-entropy (standard GAN)
    - lsgan: Least-squares GAN (MSE-based)
    - wgan-gp: Wasserstein GAN with gradient penalty

Placeholder for Phase 3 — not used in Phase 1 or Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Flexible adversarial loss supporting multiple formulations.

    Args:
        mode: Loss formulation ('vanilla', 'lsgan', 'wgan-gp').
    """

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()
        self.mode = mode

        if mode == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.criterion = nn.MSELoss()
        elif mode == "wgan-gp":
            self.criterion = None  # WGAN uses raw scores
        else:
            raise ValueError(f"Unknown adversarial loss mode: '{mode}'")

    def generator_loss(self, disc_fake: torch.Tensor) -> torch.Tensor:
        """
        Generator loss: wants discriminator to classify fakes as real.

        Args:
            disc_fake: Discriminator output on generated samples.

        Returns:
            Scalar loss.
        """
        if self.mode == "vanilla":
            target = torch.ones_like(disc_fake)
            return self.criterion(disc_fake, target)
        elif self.mode == "lsgan":
            target = torch.ones_like(disc_fake)
            return self.criterion(disc_fake, target)
        elif self.mode == "wgan-gp":
            return -disc_fake.mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def discriminator_loss(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator loss: classify reals as real, fakes as fake.

        Args:
            disc_real: Discriminator output on real samples.
            disc_fake: Discriminator output on generated samples.

        Returns:
            Scalar loss.
        """
        if self.mode == "vanilla":
            real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
            fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            return (real_loss + fake_loss) / 2
        elif self.mode == "lsgan":
            real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
            fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            return (real_loss + fake_loss) / 2
        elif self.mode == "wgan-gp":
            return disc_fake.mean() - disc_real.mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @staticmethod
    def gradient_penalty(
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            discriminator: Discriminator model.
            real: Real samples.
            fake: Generated samples.
            lambda_gp: Gradient penalty coefficient.

        Returns:
            Gradient penalty loss.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        disc_interpolated = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty
