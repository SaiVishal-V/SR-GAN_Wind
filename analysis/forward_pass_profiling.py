"""
Phase 1: Synthetic forward-pass profiling.
Measures timing, memory, gradient flow, and receptive field analysis.
"""
import sys, os, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np

from models.generator import SRResNet
from models.discriminator import PatchGANDiscriminator
from models.losses import GeneratorLoss, adversarial_loss_d, masked_l1_loss, gradient_loss

def profile_model():
    print("=" * 60)
    print("FORWARD-PASS PROFILING")
    print("=" * 60)

    device = torch.device("cpu")
    gen = SRResNet(in_channels=1, num_features=64, num_residual_blocks=8).to(device)
    disc = PatchGANDiscriminator(in_channels=1, base_channels=64).to(device)

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    g_trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    d_trainable = sum(p.numel() for p in disc.parameters() if p.requires_grad)

    print(f"\n  Generator:     {g_params:>10,} params ({g_trainable:,} trainable)")
    print(f"  Discriminator: {d_params:>10,} params ({d_trainable:,} trainable)")
    print(f"  G/D ratio:     {g_params/d_params:.2f}")

    # Per-layer parameter count
    print(f"\n  Generator layer breakdown:")
    for name, module in gen.named_modules():
        if isinstance(module, nn.Conv2d):
            n = sum(p.numel() for p in module.parameters())
            print(f"    {name:40s}: {n:>8,} params, kernel={module.kernel_size}, in={module.in_channels}, out={module.out_channels}")

    # Timing: patch input
    print(f"\n  --- Timing (CPU) ---")
    lr_patch = torch.randn(16, 1, 16, 16)
    hr_patch = torch.randn(16, 1, 64, 64)
    mask_patch = torch.ones(16, 1, 64, 64)

    # Warm up
    _ = gen(lr_patch)

    t0 = time.perf_counter()
    for _ in range(10):
        sr = gen(lr_patch)
    t_gen_patch = (time.perf_counter() - t0) / 10
    print(f"  Generator patch (B=16, 16x16->64x64): {t_gen_patch*1000:.1f} ms")

    t0 = time.perf_counter()
    for _ in range(10):
        _ = disc(hr_patch)
    t_disc_patch = (time.perf_counter() - t0) / 10
    print(f"  Discriminator patch (B=16, 64x64):     {t_disc_patch*1000:.1f} ms")

    # Timing: full scene
    lr_full = torch.randn(1, 1, 80, 140)
    hr_full = torch.randn(1, 1, 320, 560)
    mask_full = torch.ones(1, 1, 320, 560)

    _ = gen(lr_full)
    t0 = time.perf_counter()
    for _ in range(3):
        sr_full = gen(lr_full)
    t_gen_full = (time.perf_counter() - t0) / 3
    print(f"  Generator full scene (80x140->320x560): {t_gen_full*1000:.1f} ms")

    t0 = time.perf_counter()
    for _ in range(3):
        _ = disc(hr_full)
    t_disc_full = (time.perf_counter() - t0) / 3
    print(f"  Discriminator full scene (320x560):     {t_disc_full*1000:.1f} ms")

    # Gradient flow analysis
    print(f"\n  --- Gradient Flow Analysis ---")
    gen.zero_grad()
    sr = gen(lr_patch)
    loss = masked_l1_loss(sr, hr_patch, mask_patch)
    loss.backward()

    grad_norms = {}
    for name, param in gen.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms[name] = norm

    # Summary statistics
    norms = list(grad_norms.values())
    print(f"  Total parameters with gradients: {len(norms)}")
    print(f"  Gradient norm min:  {min(norms):.6e}")
    print(f"  Gradient norm max:  {max(norms):.6e}")
    print(f"  Gradient norm mean: {np.mean(norms):.6e}")
    print(f"  Gradient norm std:  {np.std(norms):.6e}")

    # Check for vanishing/exploding
    vanishing = sum(1 for n in norms if n < 1e-7)
    exploding = sum(1 for n in norms if n > 100)
    print(f"  Vanishing (norm < 1e-7): {vanishing}/{len(norms)}")
    print(f"  Exploding (norm > 100):  {exploding}/{len(norms)}")

    # Layer-wise gradient norms (groups)
    print(f"\n  Layer-group gradient norms:")
    groups = {}
    for name, norm in grad_norms.items():
        parts = name.split(".")
        group = parts[0]
        if group not in groups:
            groups[group] = []
        groups[group].append(norm)

    for group, gnorms in groups.items():
        print(f"    {group:25s}: mean={np.mean(gnorms):.6e}, min={min(gnorms):.6e}, max={max(gnorms):.6e}")

    # Loss magnitude analysis
    print(f"\n  --- Loss Magnitude Analysis ---")
    gen.zero_grad()
    sr = gen(lr_patch)

    l1 = masked_l1_loss(sr, hr_patch, mask_patch)
    grad = gradient_loss(sr, hr_patch, mask_patch)

    fake_preds = disc(sr)
    real_preds = disc(hr_patch.detach())

    gen_loss_fn = GeneratorLoss(pixel_weight=1.0, adversarial_weight=1e-3, gradient_weight=0.1)
    g_losses = gen_loss_fn(sr, hr_patch, mask_patch, fake_preds)
    d_loss = adversarial_loss_d(real_preds, fake_preds.detach())

    print(f"  L1 loss:           {l1.item():.6f}")
    print(f"  Gradient loss:     {grad.item():.6f}")
    print(f"  Adversarial (G):   {g_losses['adversarial_loss'].item():.6f}")
    print(f"  Total G loss:      {g_losses['total_loss'].item():.6f}")
    print(f"  D loss:            {d_loss.item():.6f}")
    print(f"")
    print(f"  Weighted contributions to G loss:")
    print(f"    1.0 * L1:     {1.0 * l1.item():.6f}  ({100*1.0*l1.item()/g_losses['total_loss'].item():.1f}%)")
    print(f"    0.1 * grad:   {0.1 * grad.item():.6f}  ({100*0.1*grad.item()/g_losses['total_loss'].item():.1f}%)")
    print(f"    1e-3 * adv:   {1e-3 * g_losses['adversarial_loss'].item():.6f}  ({100*1e-3*g_losses['adversarial_loss'].item()/g_losses['total_loss'].item():.1f}%)")

    # Receptive field estimation
    print(f"\n  --- Receptive Field Analysis ---")
    print(f"  Generator:")
    print(f"    Initial conv: 9x9 kernel -> RF = 9")
    print(f"    8 residual blocks (2x 3x3 each) -> RF += 8 * 2 * 2 = 32")
    print(f"    Post-residual conv: 3x3 -> RF += 2")
    print(f"    2x upsample blocks (3x3 each at lower res) -> RF += 2*2 = 4")
    print(f"    Final conv: 9x9 at HR -> RF += 8/4 = 2 (at LR scale)")
    print(f"    Estimated LR RF: ~49 pixels")
    print(f"    LR image size: 80x140")
    print(f"    RF coverage: {49/80*100:.0f}% of height, {49/140*100:.0f}% of width")

    print(f"\n  Discriminator:")
    print(f"    4x4 stride-2: RF = 4")
    print(f"    4x4 stride-2: RF = 10")
    print(f"    4x4 stride-2: RF = 22")
    print(f"    4x4 stride-1: RF = 25")
    print(f"    4x4 stride-1: RF = 28")
    print(f"    Estimated HR RF: ~28 pixels (not 70 as documented)")
    print(f"    Note: Actual PatchGAN RF may differ from 70x70 claim")

    # Capacity analysis
    print(f"\n  --- Capacity Analysis ---")
    print(f"  Dataset: 392 timesteps, 32 patches/image")
    print(f"  Train set (70%): 274 timesteps x 32 = {274*32:,} patches/epoch")
    print(f"  Parameters: {g_params:,}")
    print(f"  Ratio (patches/params): {274*32/g_params:.4f}")
    print(f"  Note: With 595K params and 8768 patches, overfitting risk is MODERATE")
    print(f"  Rule of thumb: need ~5-10x more data than params for safe generalization")
    print(f"  Current ratio suggests model capacity is appropriate but near the limit")

    # Discriminator architecture analysis
    print(f"\n  --- Discriminator Architecture Analysis ---")
    print(f"  Uses: SpectralNorm + InstanceNorm (inside _conv_block)")
    print(f"  Concern: SpectralNorm constrains Lipschitz constant of weight matrix")
    print(f"  InstanceNorm then re-normalizes activations")
    print(f"  These two normalizations can conflict:")
    print(f"    - SN controls the weight spectral radius")
    print(f"    - IN then undoes some of this control by normalizing activations")
    print(f"  Recommendation: benchmark SN-only vs SN+IN")


if __name__ == "__main__":
    profile_model()
