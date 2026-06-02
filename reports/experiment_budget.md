# Experiment Budget

| Experiment | Description | Est. Hours | GPU | Priority | Dependencies |
|---|---|---|---|---|---|
| E0 | Diagnosis (code analysis) | 0.5 | CPU | Tier 1 | None |
| E1 | Baseline training + full evaluation | 8–12 | P100/T4 | Tier 1 | E0 |
| E2 | + EMA generator (decay=0.999) | 8–12 | P100/T4 | Tier 1 | E1 |
| E3 | + Laplacian loss (weight=0.05) | 8–12 | P100/T4 | Tier 1 | E1 |
| E4 | Power spectrum evaluation (no training) | 0.5 | CPU/GPU | Tier 1 | E1 checkpoint |
| E5 | + Spectral loss (weight=0.001) | 8–12 | P100/T4 | Tier 2 | E4 results |
| E6 | Charbonnier vs L1 | 8–12 | P100/T4 | Tier 2 | E1 |
| E7 | Cosine Warm Restarts | 8–12 | P100/T4 | Tier 2 | E1 plateau analysis |
| E8 | Discriminator SN-only vs SN+IN | 8–12 | P100/T4 | Tier 2 | E1 |
| E9 | SWA (after EMA comparison) | 8–12 | P100/T4 | Tier 3 | E2 |
| E10 | OneCycleLR | 8–12 | P100/T4 | Tier 3 | E7 |
| E11 | Multi-seed sweep (5 seeds) | 40–60 | P100/T4 | Tier 3 | Best config |
| E12 | Alternative LR generation | 24–36 | P100/T4 | Tier 3 | Best config |

## Totals

| Tier | Experiments | Est. Total Hours | Notes |
|---|---|---|---|
| Tier 1 | E0–E4 | 25–37 | Minimum viable set |
| Tier 2 | E5–E8 | 32–48 | Only if Tier 1 shows >2% improvement |
| Tier 3 | E9–E12 | 80–120 | Only if Tier 2 shows improvement |
| **Grand Total** | E0–E12 | **137–205** | Worst case ~9 days of P100 time |

## Platform Notes

- **Kaggle P100**: 30-hour session limit, 16 GB VRAM. Can run 2–3 experiments per session.
- **Colab T4**: 12-hour limit (free), 15 GB VRAM. One experiment per session.
- **Local CPU**: Diagnosis only (E0, E4). Training is impractical.

## Recommendation

Start with Tier 1 only (E0–E4, ~25–37 GPU-hours). This fits in 2–3 Kaggle sessions. Evaluate results before committing to Tier 2.
