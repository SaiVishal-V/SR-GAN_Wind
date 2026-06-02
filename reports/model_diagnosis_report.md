# SR-GAN Wind-Speed Super-Resolution — Model Diagnosis Report

**Generated**: Phase 1 Diagnostic Audit  
**Status**: Code-level + synthetic profiling analysis (no historical checkpoints available)

---

## 1. Executive Summary

The V1 SR-GAN pipeline is architecturally sound but has **five identified bottlenecks** that limit reconstruction quality. The highest-priority findings are: (1) the generator is 3× smaller than the discriminator, creating a capacity imbalance; (2) the adversarial loss contributes effectively 0% of the total generator loss, making GAN fine-tuning nearly ineffective; (3) the discriminator's actual receptive field is ~28 pixels, not 70 as documented; (4) the observed_mask (36.3% coverage) is not used during training, wasting the strongest quality signal; and (5) gradient norms show moderate imbalance across layer groups but no vanishing gradients.

---

## 2. Architecture Analysis

### Generator (SRResNet)
| Property | Value |
|----------|-------|
| Total parameters | 934,337 |
| Trainable parameters | 934,337 |
| Residual blocks | 8 |
| Feature channels | 64 |
| Upsampling | 2× PixelShuffle × 2 = 4× |
| BatchNorm | **None** (correct) |
| Activation | PReLU |
| Initial kernel | 9×9 |
| Final kernel | 9×9 |

**Assessment**: Architecture follows SRResNet best practices. No BatchNorm is correct for geophysical continuous fields. 8 residual blocks provide adequate depth. PReLU is appropriate.

### Discriminator (PatchGAN)
| Property | Value |
|----------|-------|
| Total parameters | 2,762,689 |
| Architecture | 4-layer PatchGAN |
| Normalization | SpectralNorm + InstanceNorm |
| Activation | LeakyReLU(0.2) |

**Finding: G/D Parameter Imbalance**
> Generator: 934K params  
> Discriminator: 2,763K params  
> **G/D ratio: 0.34** (Generator is 3× smaller)

This imbalance means the discriminator has significantly more capacity than the generator. In GAN training, this creates risk of the discriminator overpowering the generator, leading to:
- Generator gradient saturation
- Mode collapse
- Training instability

**Recommendation**: This is not necessarily harmful (SNGAN papers use similar ratios), but should be monitored during training via gradient norms and D/G loss ratio.

### Receptive Field

**Generator** (at LR scale):
- Estimated: ~49 pixels
- LR image: 80×140
- Coverage: 61% height, 35% width
- **Assessment**: Adequate but not full-scene coverage. The generator can see ~half the image context, which is sufficient for local structure but may miss large-scale coherence.

**Discriminator** (at HR scale):
- Estimated: ~28 pixels (not 70 as documented in code comments)
- HR image: 320×560
- Coverage: 8.75% height, 5% width
- **Assessment**: The discriminator evaluates very local texture realism. This is appropriate for PatchGAN but means it cannot enforce large-scale spatial coherence — that responsibility falls entirely on the pixel and gradient losses.

---

## 3. Loss Function Analysis

### Loss Magnitude Balance (Synthetic Data)

| Loss Component | Raw Value | Weight | Weighted Value | % of Total |
|----------------|-----------|--------|----------------|------------|
| Masked L1 | 34.17 | 1.0 | 34.17 | **86.7%** |
| Gradient Loss | 52.43 | 0.1 | 5.24 | **13.3%** |
| Adversarial (G) | 0.94 | 0.001 | 0.00094 | **0.0%** |

> [!WARNING]
> **Critical Finding**: The adversarial loss contributes effectively **0.0%** to the total generator loss. With weight 1e-3 and raw magnitude ~0.94, the adversarial gradient signal is completely drowned out by L1 and gradient losses. This means **GAN fine-tuning (Stage 2) may be nearly ineffective** — the generator receives virtually no adversarial learning signal.

**Impact**: The Stage 2 GAN training may produce minimal improvement over Stage 1 pretrained SRResNet, because the adversarial loss cannot meaningfully steer the generator.

**Recommendation for Tier 1**: After establishing baseline metrics, consider increasing `adversarial_weight` to 0.01–0.05 during GAN phase as a controlled experiment. Alternatively, use relativistic adversarial loss which naturally has higher magnitude.

### Missing Loss Components
The following losses are absent but would benefit wind-field reconstruction:
1. **Laplacian loss** — would improve sharpness/edge preservation (Tier 1: E3)
2. **Spectral loss** — would preserve spatial frequency content (Tier 2: E5)
3. **No observed-mask-aware training** — the `observed_mask` (36.3% coverage) could weight losses higher on satellite-observed pixels

---

## 4. Gradient Flow Analysis

### Per-Layer Group Gradient Norms (L1 backward pass)

| Layer Group | Mean Norm | Min Norm | Max Norm |
|-------------|-----------|----------|----------|
| initial | 3.95e+01 | 5.91e+00 | 6.14e+01 |
| residual_blocks | 3.31e+01 | 5.61e-01 | 9.22e+01 |
| post_residual | 4.76e+01 | 3.95e-01 | 9.48e+01 |
| upsample | 5.52e+01 | 4.69e-01 | **1.79e+02** |
| final | 2.01e+01 | 1.53e-01 | 4.01e+01 |

**Findings**:
- **No vanishing gradients**: All 53 parameter groups have norm > 0.15 ✓
- **Mild exploding risk**: 2/53 parameters have norm > 100 (both in upsample layers)
- **Upsample layers receive strongest gradients**: This is expected since PixelShuffle conv layers (3×3→256 channels) have the most parameters (147K each) and are closest to the output
- **Gradient clipping (max_norm=1.0)**: Already configured, which prevents exploding gradients ✓

**Assessment**: Gradient flow is healthy. The 9×9 initial/final kernels provide strong gradient paths. Global skip connection ensures deep residual blocks receive gradients. No intervention needed.

---

## 5. Overfitting Risk Assessment

| Factor | Value | Risk |
|--------|-------|------|
| Training samples | 274 timesteps × 32 patches = 8,768/epoch | |
| Total parameters | 934,337 | |
| Samples/params ratio | 0.0094 | **HIGH** |
| Data augmentation | None | Increases risk |
| Regularization | Weight decay 1e-4, gradient clip 1.0 | Moderate |
| Early stopping | Yes (patience=30, burn-in 80%) | Mitigates |

**Assessment**: The samples-to-parameters ratio of 0.0094 is **very low** (rule of thumb: 5–10× recommended). However:
1. Each patch is a 16×16=256 pixel spatial sample, so the effective "information" per sample is higher than a classification task
2. Patches are randomly positioned each epoch, providing implicit augmentation
3. EMA (Tier 1: E2) will help smooth overfitting
4. The temporal split prevents data leakage

**Risk Rating**: MODERATE — the model can fit the training data but generalization depends on regularization effectiveness. EMA implementation is high priority.

---

## 6. Mode Collapse Risk Assessment

| Factor | Risk Level | Rationale |
|--------|------------|-----------|
| Output channels | Low | Single channel (wind speed) — no color mode collapse |
| PatchGAN | Low | Patch-level discrimination is more stable than global |
| SpectralNorm | Low | Constrains discriminator, prevents D from overpowering |
| Adversarial weight | **Very Low** | At 0.0% contribution, the GAN barely influences G |
| L1 anchoring | Low | Strong L1 loss anchors output to ground truth |

**Assessment**: Mode collapse risk is **very low** precisely because the adversarial loss is so weak. The generator is effectively an L1-trained SRResNet. If adversarial weight is increased (recommended), mode collapse risk should be monitored.

---

## 7. Optimization Plateau Risk

| Factor | Assessment |
|--------|------------|
| Scheduler | CosineAnnealingLR with η_min=1e-6 |
| Pretrain epochs | 100 |
| GAN epochs | 200 |
| Early stopping | Patience=30, burn-in=80% |

**Concern**: The burn-in at 80% means early stopping cannot trigger until epoch 80 (pretrain) or epoch 160 (GAN). Combined with CosineAnnealingLR, the learning rate reaches η_min at the end of training. If the model plateaus early (e.g., epoch 30), it continues training for 50+ epochs at a decaying LR with no benefit.

**Recommendation**: Add plateau detection with automatic warm restart trigger (Tier 2: E7). For now, gradient monitoring (Phase 3) will detect whether the model is actually learning during the decay phase.

---

## 8. Discriminator Normalization Concern

The discriminator uses **both** SpectralNorm and InstanceNorm in conv blocks (except the first layer).

**SpectralNorm**: Constrains the spectral radius of weight matrices → Lipschitz continuity → stable GAN training.

**InstanceNorm**: Normalizes activations per-instance per-channel → removes mean/variance information.

**Conflict**: SpectralNorm carefully controls the weight magnitude to maintain Lipschitz bounds. InstanceNorm then normalizes the resulting activations, partially undoing the spectral normalization effect. This is not standard practice — most SNGAN implementations use SpectralNorm **only** in the discriminator.

**Recommendation**: Benchmark SN-only vs SN+IN as experiment E8 (Tier 2). Make configurable via `use_instance_norm` in config.

---

## 9. Ranked Recommendations

| Priority | Finding | Recommended Action | Experiment |
|----------|---------|-------------------|------------|
| 1 | Adversarial loss = 0% of total | Increase `adversarial_weight` or defer GAN phase assessment | Tier 1 awareness |
| 2 | No EMA | Implement EMA for generalization | E2 |
| 3 | No Laplacian loss | Add for sharpness | E3 |
| 4 | observed_mask unused | Integrate into evaluation | E1 |
| 5 | G/D parameter imbalance | Monitor via gradient norms | Phase 3 |
| 6 | SN+IN conflict in D | Benchmark SN-only | E8 |
| 7 | Plateau detection absent | Add warm restart trigger | E7 |
| 8 | No spectral evaluation | Add power spectrum metric | E4 |
| 9 | High overfitting risk ratio | EMA + gradient monitoring | E2 + Phase 3 |

---

## 10. Conclusion

The V1 architecture is fundamentally sound. The **primary bottleneck is not architectural but optimization-related**: the adversarial loss is too weak to provide meaningful GAN learning signal. Before any architecture changes, the priority should be:

1. **Establish baseline metrics** with current model (E1)
2. **Add EMA** for generalization (E2) 
3. **Add Laplacian loss** for sharpness (E3)
4. **Evaluate spectral quality** to determine if spectral loss is needed (E4)

The model is NOT trapped in local minima (no evidence of this). The model is likely underutilizing the GAN phase due to adversarial weight being too low.
