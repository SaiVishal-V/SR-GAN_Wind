"""
Smoke tests for the SR-GAN Wind Speed pipeline.
Verifies all modules import correctly, shapes are correct, and losses compute properly.
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    from utils.seed import set_seed
    from utils.checkpoint import save_checkpoint, load_checkpoint, BestMetricTracker
    from utils.metrics import (
        compute_all_metrics, denormalize, masked_mae, masked_rmse,
        masked_psnr, masked_ssim, gradient_rmse, physical_metrics,
        NORM_MEAN, NORM_STD,
    )
    from utils.plotting import (
        plot_wind_field, plot_comparison, plot_scatter, plot_histogram,
    )
    from datasets.wind_dataset import WindSRDataset, get_temporal_split
    from models.generator import SRResNet, ResidualBlock, UpsampleBlock
    from models.discriminator import PatchGANDiscriminator
    from models.losses import (
        masked_l1_loss, masked_mse_loss, masked_rmse as loss_rmse,
        gradient_loss, adversarial_loss_g, adversarial_loss_d,
        GeneratorLoss,
    )
    
    print("  [PASS] All modules imported successfully")
    return True


def test_generator():
    print("\n" + "=" * 60)
    print("TEST 2: Generator (SRResNet)")
    print("=" * 60)
    
    from models.generator import SRResNet
    
    model = SRResNet(
        in_channels=1,
        num_features=64,
        num_residual_blocks=8,
        scale_factor=4,
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    # Test with patch input
    x = torch.randn(2, 1, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {y.shape}"
    print(f"  Patch input:  (2,1,16,16) -> {tuple(y.shape)} [PASS]")
    
    # Test with full image input
    x_full = torch.randn(1, 1, 80, 140)
    y_full = model(x_full)
    assert y_full.shape == (1, 1, 320, 560), f"Expected (1,1,320,560), got {y_full.shape}"
    print(f"  Full input:   (1,1,80,140) -> {tuple(y_full.shape)} [PASS]")
    
    # Verify no BatchNorm
    has_bn = any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))
                 for m in model.modules())
    assert not has_bn, "Generator should NOT have BatchNorm!"
    print(f"  No BatchNorm: [PASS]")
    
    print(f"  [PASS] Generator tests passed")
    return True


def test_discriminator():
    print("\n" + "=" * 60)
    print("TEST 3: Discriminator (PatchGAN)")
    print("=" * 60)
    
    from models.discriminator import PatchGANDiscriminator
    
    model = PatchGANDiscriminator(
        in_channels=1,
        base_channels=64,
        use_spectral_norm=True,
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    # Test with HR patch
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"  Patch input:  (2,1,64,64) -> {tuple(y.shape)}")
    assert y.dim() == 4 and y.shape[1] == 1, "Output should be (B,1,H',W')"
    print(f"  Output is patch map: [PASS]")
    
    print(f"  [PASS] Discriminator tests passed")
    return True


def test_losses():
    print("\n" + "=" * 60)
    print("TEST 4: Loss Functions")
    print("=" * 60)
    
    from models.losses import (
        masked_l1_loss, masked_mse_loss, masked_rmse,
        gradient_loss, adversarial_loss_g, adversarial_loss_d,
        GeneratorLoss,
    )
    
    B, C, H, W = 2, 1, 64, 64
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randn(B, C, H, W)
    
    # Create realistic mask (some ocean, some land)
    mask = torch.zeros(B, C, H, W)
    mask[:, :, :40, :] = 1.0  # Upper part is ocean
    
    # Masked L1
    l1 = masked_l1_loss(pred, target, mask)
    assert l1.requires_grad, "L1 loss should require grad"
    print(f"  Masked L1:     {l1.item():.6f} [PASS]")
    
    # Masked MSE
    mse = masked_mse_loss(pred, target, mask)
    print(f"  Masked MSE:    {mse.item():.6f} [PASS]")
    
    # Masked RMSE
    rmse = masked_rmse(pred, target, mask)
    print(f"  Masked RMSE:   {rmse.item():.6f} [PASS]")
    
    # Gradient loss
    grad = gradient_loss(pred, target, mask)
    print(f"  Gradient loss: {grad.item():.6f} [PASS]")
    
    # Verify loss is zero for identical inputs
    pred2 = pred.detach().requires_grad_(True)
    l1_zero = masked_l1_loss(pred2, pred2.detach(), mask)
    assert l1_zero.item() < 1e-6, f"L1(x,x) should be ~0, got {l1_zero.item()}"
    print(f"  L1(x,x) ~ 0:  {l1_zero.item():.8f} [PASS]")
    
    # Verify loss is zero when mask is all-land
    empty_mask = torch.zeros_like(mask)
    l1_empty = masked_l1_loss(pred, target, empty_mask)
    assert l1_empty.item() < 1e-6, f"L1 on empty mask should be 0, got {l1_empty.item()}"
    print(f"  L1(no ocean):  {l1_empty.item():.8f} [PASS]")
    
    # GeneratorLoss combined
    gen_loss = GeneratorLoss(pixel_weight=1.0, adversarial_weight=1e-3, gradient_weight=0.1)
    fake_preds = torch.randn(B, 1, 4, 4)
    losses = gen_loss(pred, target, mask, fake_preds)
    assert "total_loss" in losses
    assert "pixel_loss" in losses
    assert "gradient_loss" in losses
    assert "adversarial_loss" in losses
    print(f"  Generator combined loss: {losses['total_loss'].item():.6f} [PASS]")
    
    print(f"  [PASS] All loss tests passed")
    return True


def test_metrics():
    print("\n" + "=" * 60)
    print("TEST 5: Metrics")
    print("=" * 60)
    
    from utils.metrics import (
        compute_all_metrics, denormalize, gradient_rmse,
        physical_metrics, NORM_MEAN, NORM_STD,
    )
    
    # Verify normalization constants
    assert abs(NORM_MEAN - 6.336302810300043) < 1e-10, "Mean mismatch!"
    assert abs(NORM_STD - 2.9275431488456434) < 1e-10, "Std mismatch!"
    print(f"  Norm mean: {NORM_MEAN} [PASS]")
    print(f"  Norm std:  {NORM_STD} [PASS]")
    
    # Denormalization
    x = torch.tensor([0.0])
    x_mps = denormalize(x)
    assert abs(x_mps.item() - NORM_MEAN) < 1e-6
    print(f"  Denorm(0) = {x_mps.item():.4f} m/s [PASS]")
    
    # Full metrics
    pred = torch.randn(1, 1, 64, 64)
    target = torch.randn(1, 1, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    
    metrics = compute_all_metrics(pred, target, mask)
    required_keys = ["mae", "rmse", "psnr", "ssim", "gradient_rmse",
                     "rmse_mps", "mae_mps", "bias_mps", "correlation"]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        print(f"  {key:20s}: {metrics[key]:.6f}")
    
    print(f"  [PASS] All metric tests passed")
    return True


def test_temporal_split():
    print("\n" + "=" * 60)
    print("TEST 6: Temporal Split")
    print("=" * 60)
    
    from datasets.wind_dataset import get_temporal_split
    
    train, val, test = get_temporal_split(392, 0.70, 0.15, 0.15)
    
    print(f"  Train: {len(train)} timesteps (indices {train[0]}-{train[-1]})")
    print(f"  Val:   {len(val)} timesteps (indices {val[0]}-{val[-1]})")
    print(f"  Test:  {len(test)} timesteps (indices {test[0]}-{test[-1]})")
    
    # Verify no overlap
    assert len(set(train) & set(val)) == 0, "Train/val overlap!"
    assert len(set(val) & set(test)) == 0, "Val/test overlap!"
    assert len(set(train) & set(test)) == 0, "Train/test overlap!"
    print(f"  No overlap: [PASS]")
    
    # Verify temporal ordering (no random)
    assert train[-1] < val[0], "Not temporal split!"
    assert val[-1] < test[0], "Not temporal split!"
    print(f"  Temporal ordering: [PASS]")
    
    # Verify coverage
    assert len(train) + len(val) + len(test) == 392
    print(f"  Full coverage: [PASS]")
    
    print(f"  [PASS] Temporal split tests passed")
    return True


def test_dataset():
    print("\n" + "=" * 60)
    print("TEST 7: Dataset Loading")
    print("=" * 60)
    
    from datasets.wind_dataset import WindSRDataset, get_temporal_split
    
    nc_path = r"E:\SR-GAN\IR_wind_23_24_new_SRGAN_ready.nc"
    if not os.path.isfile(nc_path):
        print(f"  [SKIP] Dataset not found at {nc_path}")
        return True
    
    train_idx, val_idx, test_idx = get_temporal_split(392, 0.70, 0.15, 0.15)
    
    # Test patch mode
    dataset = WindSRDataset(
        nc_path=nc_path,
        time_indices=train_idx[:5],  # Only 5 timesteps for speed
        mode="patch",
        lr_patch_size=16,
        hr_patch_size=64,
        min_ocean_fraction=0.5,
        patches_per_image=4,
        cache=True,
    )
    
    sample = dataset[0]
    lr = sample["lr"]
    hr = sample["hr"]
    mask = sample["mask"]
    
    assert lr.shape == (1, 16, 16), f"LR shape: {lr.shape}"
    assert hr.shape == (1, 64, 64), f"HR shape: {hr.shape}"
    assert mask.shape == (1, 64, 64), f"Mask shape: {mask.shape}"
    print(f"  Patch LR:   {tuple(lr.shape)} [PASS]")
    print(f"  Patch HR:   {tuple(hr.shape)} [PASS]")
    print(f"  Patch Mask: {tuple(mask.shape)} [PASS]")
    
    # Verify no fill values in output
    assert (lr > -9990).all(), "Fill values in LR!"
    assert (hr > -9990).all(), "Fill values in HR!"
    print(f"  No fill values: [PASS]")
    
    # Test full mode
    full_dataset = WindSRDataset(
        nc_path=nc_path,
        time_indices=val_idx[:2],
        mode="full",
        cache=True,
    )
    
    sample_full = full_dataset[0]
    assert sample_full["lr"].shape == (1, 80, 140), f"Full LR: {sample_full['lr'].shape}"
    assert sample_full["hr"].shape == (1, 320, 560), f"Full HR: {sample_full['hr'].shape}"
    assert sample_full["mask"].shape == (1, 320, 560), f"Full Mask: {sample_full['mask'].shape}"
    print(f"  Full LR:    {tuple(sample_full['lr'].shape)} [PASS]")
    print(f"  Full HR:    {tuple(sample_full['hr'].shape)} [PASS]")
    print(f"  Full Mask:  {tuple(sample_full['mask'].shape)} [PASS]")
    
    dataset.close()
    full_dataset.close()
    print(f"  [PASS] Dataset tests passed")
    return True


def test_end_to_end():
    print("\n" + "=" * 60)
    print("TEST 8: End-to-End Forward Pass")
    print("=" * 60)
    
    from models.generator import SRResNet
    from models.discriminator import PatchGANDiscriminator
    from models.losses import GeneratorLoss, adversarial_loss_d
    
    generator = SRResNet(in_channels=1, num_features=64, num_residual_blocks=8)
    discriminator = PatchGANDiscriminator(in_channels=1, base_channels=64)
    gen_loss_fn = GeneratorLoss()
    
    # Simulate one training step
    lr = torch.randn(4, 1, 16, 16)
    hr = torch.randn(4, 1, 64, 64)
    mask = torch.ones(4, 1, 64, 64)
    
    # Generator forward
    sr = generator(lr)
    assert sr.shape == hr.shape
    
    # Discriminator forward
    real_preds = discriminator(hr)
    fake_preds = discriminator(sr.detach())
    
    # Losses
    d_loss = adversarial_loss_d(real_preds, fake_preds)
    
    fake_preds_for_g = discriminator(sr)
    g_losses = gen_loss_fn(sr, hr, mask, fake_preds_for_g)
    
    # Backward
    g_losses["total_loss"].backward()
    
    print(f"  Generator loss:     {g_losses['total_loss'].item():.6f}")
    print(f"  Discriminator loss: {d_loss.item():.6f}")
    print(f"  Gradient computed:  [PASS]")
    
    # Verify gradients exist
    has_grad = any(p.grad is not None for p in generator.parameters())
    assert has_grad, "No gradients on generator!"
    print(f"  Generator grads:    [PASS]")
    
    print(f"  [PASS] End-to-end test passed")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SR-GAN WIND SPEED -- SMOKE TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Generator", test_generator),
        ("Discriminator", test_discriminator),
        ("Losses", test_losses),
        ("Metrics", test_metrics),
        ("Temporal Split", test_temporal_split),
        ("Dataset", test_dataset),
        ("End-to-End", test_end_to_end),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "[PASS]" if passed else "[FAIL]"
        except Exception as e:
            results[name] = f"[ERROR]: {e}"
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name:20s}: {result}")
    
    all_passed = all("PASS" in v for v in results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
