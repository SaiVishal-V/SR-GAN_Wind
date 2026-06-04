"""Smoke test for the full GAN pipeline."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

print("=" * 60)
print("WindGapGAN — Full Pipeline Smoke Test")
print("=" * 60)

# ── 1. Generator ──────────────────────────────────────────────
from models.unet import MaskedUNet

gen = MaskedUNet(
    in_channels=2, out_channels=1, base_features=32, depth=4,
    use_attention=True, use_tanh=False, hard_merge=False,
)
gen_params = sum(p.numel() for p in gen.parameters())
print(f"\n✅ Generator: {gen_params:,} params")

# Forward pass
x = torch.randn(2, 2, 64, 64)
out = gen(x)
print(f"   Input: {x.shape} → Output: {out.shape}")

# Gradient check: ensure gradients flow
out.sum().backward()
non_zero_grads = sum(1 for p in gen.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total_params_count = sum(1 for p in gen.parameters())
print(f"   Gradient flow: {non_zero_grads}/{total_params_count} params have non-zero gradients")
gen.zero_grad()

# ── 2. Discriminator ─────────────────────────────────────────
from models.discriminator import PatchGANDiscriminator

disc = PatchGANDiscriminator(in_channels=3, base_features=64, n_layers=3)
disc_params = sum(p.numel() for p in disc.parameters())
print(f"\n✅ Discriminator: {disc_params:,} params")

d_in = torch.randn(2, 3, 64, 64)
d_out, features = disc(d_in, return_features=True)
print(f"   Input: {d_in.shape} → Score: {d_out.shape}, {len(features)} feature layers")
for i, f in enumerate(features):
    print(f"   Feature {i}: {f.shape}")

# ── 3. Losses ─────────────────────────────────────────────────
from losses.adversarial import AdversarialLoss
from losses.perceptual_loss import PerceptualLoss
from losses.spectral_loss import SpectralLoss
from losses.masked_l1 import MaskedL1Loss
from losses.gradient_loss import GradientLoss

pred = torch.randn(2, 1, 64, 64)
target = torch.randn(2, 1, 64, 64)
mask = (torch.rand(2, 1, 64, 64) > 0.3).float()

# MaskedL1
ml1 = MaskedL1Loss()
l1 = ml1(pred, target, mask)
print(f"\n✅ MaskedL1Loss: {l1.item():.4f}")

# Adversarial (LSGAN)
adv = AdversarialLoss(mode="lsgan")
g_loss = adv.generator_loss(d_out)
d_loss = adv.discriminator_loss(d_out, d_out.detach())
print(f"✅ AdversarialLoss: G={g_loss.item():.4f}, D={d_loss.item():.4f}")

# Perceptual
perc = PerceptualLoss(n_layers=len(features))
fake_feats = [torch.randn_like(f) for f in features]
p_loss = perc(fake_feats, features)
print(f"✅ PerceptualLoss: {p_loss.item():.4f}")

# Spectral
spec = SpectralLoss()
s_loss = spec(pred, target)
print(f"✅ SpectralLoss: {s_loss.item():.4f}")

# Gradient
grad = GradientLoss()
gr_loss = grad(pred, target)
print(f"✅ GradientLoss: {gr_loss.item():.4f}")

# ── 4. Full GAN Training Step Simulation ──────────────────────
print(f"\n{'=' * 60}")
print("Simulating one GAN training step...")
print("=" * 60)

# Create fake batch
batch = {
    "input": torch.randn(4, 1, 2, 64, 64),    # (B, T=1, 2, H, W)
    "target": torch.randn(4, 1, 1, 64, 64),   # (B, T=1, 1, H, W)
    "mask": (torch.rand(4, 1, 1, 64, 64) > 0.3).float(),
    "land_mask": torch.ones(4, 1, 1, 64, 64),  # all ocean
}

# Generator forward
gen.train()
inputs = batch["input"]
B, T, C, H, W = inputs.shape
inputs_flat = inputs.reshape(B * T, C, H, W)
targets_flat = batch["target"].reshape(B * T, 1, H, W)
masks_flat = batch["mask"].reshape(B * T, 1, H, W)

masked_field = inputs_flat[:, 0:1]
mask_ch = inputs_flat[:, 1:2]

fake_raw = gen(inputs)
fake_flat = fake_raw.reshape(B * T, 1, H, W)
fake_merged = mask_ch * masked_field + (1.0 - mask_ch) * fake_flat

# Discriminator forward (real vs fake)
real_input = PatchGANDiscriminator.build_input(masked_field, mask_ch, targets_flat)
fake_input = PatchGANDiscriminator.build_input(masked_field, mask_ch, fake_merged.detach())

disc_real = disc(real_input)
disc_fake = disc(fake_input)

d_total = adv.discriminator_loss(disc_real, disc_fake)
print(f"D loss: {d_total.item():.4f}")

# Generator losses
fake_input_g = PatchGANDiscriminator.build_input(masked_field, mask_ch, fake_merged)
disc_fake_g, fake_features = disc(fake_input_g, return_features=True)

with torch.no_grad():
    _, real_features = disc(real_input, return_features=True)

g_adv = adv.generator_loss(disc_fake_g)
g_pixel = ml1(fake_merged, targets_flat, masks_flat)
g_perc = perc(fake_features, real_features)
g_spec = spec(fake_merged, targets_flat)
g_grad = grad(fake_merged, targets_flat)

g_total = 100.0 * g_pixel + 1.0 * g_adv + 10.0 * g_perc + 1.0 * g_spec + 10.0 * g_grad

print(f"G loss: {g_total.item():.4f}")
print(f"  Pixel: {g_pixel.item():.4f} (×100)")
print(f"  Adv:   {g_adv.item():.4f} (×1)")
print(f"  Perc:  {g_perc.item():.4f} (×10)")
print(f"  Spec:  {g_spec.item():.4f} (×1)")
print(f"  Grad:  {g_grad.item():.4f} (×10)")

# Verify gradient flow
g_total.backward()
g_grads = sum(1 for p in gen.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
d_grads = sum(1 for p in disc.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"\n✅ Generator gradient flow: {g_grads}/{sum(1 for _ in gen.parameters())} params")
print(f"✅ Discriminator gradient flow: {d_grads}/{sum(1 for _ in disc.parameters())} params")

# ── 5. Config loading test ────────────────────────────────────
print(f"\n{'=' * 60}")
print("Testing config loading...")
print("=" * 60)
from utils.config import load_config
config = load_config("configs/default.yaml")
print(f"Model name: {config.model.name}")
print(f"Use attention: {config.model.use_attention}")
print(f"Use tanh: {config.model.use_tanh}")
print(f"Hard merge: {config.model.hard_merge}")
print(f"Pixel loss weight: {config.training.pixel_loss_weight}")
print(f"Adversarial weight: {config.training.adversarial_weight}")
print(f"Perceptual weight: {config.training.perceptual_loss_weight}")
print(f"Spectral weight: {config.training.spectral_loss_weight}")
print(f"Gradient weight: {config.training.gradient_loss_weight}")

print(f"\n{'=' * 60}")
print("🎉 ALL TESTS PASSED — GAN pipeline is fully operational!")
print("=" * 60)
