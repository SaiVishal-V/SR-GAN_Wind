"""
PatchGAN Discriminator for wind-speed super-resolution.

Architecture (per project plan):
    - 70×70 PatchGAN receptive field
    - 1-channel input (HR wind field)
    - Output: patch-level realism map (not a single scalar)
    - Spectral Normalization on all conv layers (for GAN stability)
    - LeakyReLU activation
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator with Spectral Normalization.

    Progressive channel increase: 64 → 128 → 256 → 512 → 1.
    LeakyReLU(0.2) activation.
    Output is a spatial map of patch realism scores.

    Args:
        in_channels: Number of input channels (1 for wind speed).
        base_channels: Base number of feature channels.
        use_spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        def _conv_block(
            in_ch: int,
            out_ch: int,
            stride: int = 2,
            use_sn: bool = True,
            use_bn: bool = True,
        ) -> nn.Sequential:
            layers = []
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
            if use_sn:
                conv = spectral_norm(conv)
            layers.append(conv)
            if use_bn:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        sn = use_spectral_norm

        # First layer: no normalization
        first_conv = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        if sn:
            first_conv = spectral_norm(first_conv)

        self.model = nn.Sequential(
            # (B, 1, H, W) → (B, 64, H/2, W/2)
            first_conv,
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64, H/2, W/2) → (B, 128, H/4, W/4)
            _conv_block(base_channels, base_channels * 2, stride=2, use_sn=sn),
            # (B, 128, H/4, W/4) → (B, 256, H/8, W/8)
            _conv_block(base_channels * 2, base_channels * 4, stride=2, use_sn=sn),
            # (B, 256, H/8, W/8) → (B, 512, H/16, W/16)
            _conv_block(base_channels * 4, base_channels * 8, stride=1, use_sn=sn),
        )

        # Final output: patch realism map
        final_conv = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        if sn:
            final_conv = spectral_norm(final_conv)
        self.final = final_conv

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HR wind field tensor of shape (B, 1, H, W).

        Returns:
            Patch realism map of shape (B, 1, H', W').
        """
        features = self.model(x)
        out = self.final(features)
        return out
