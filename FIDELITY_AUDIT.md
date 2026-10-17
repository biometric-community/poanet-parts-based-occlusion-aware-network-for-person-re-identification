# FIDELITY_AUDIT — POANET

Paper: Dutta et al., *POANET*, TBIOM 2026 · DOI: 10.1109/TBIOM.2025.3641048

## Per-pass scorecard

| Pass | Focus | Confidence | Notes |
|------|-------|------------|-------|
| 1 | Architecture | 80% | ODN + OAM + PAFE (ResNet-50); stripe parts vs parsing (D3) |
| 2 | Losses | 86% | CE + batch-hard triplet + BCE occ |
| 3 | Data/eval | 82% | Market CMC/mAP; occluded suites blocked (D2) |
| 4 | Hyperparams | 84% | 384×128, SGD 2e-4, 100 ep, P×K=16×4 |
| 5 | Packaging | 90% | scripts, REPORT, deviations, publish |

**Weighted ≈ 84%.** Stop after Pass 5.

## Measured Market-1501

| Epoch | mAP | Rank-1 | Rank-5 |
|------:|----:|-------:|-------:|
| 20 | 7.74 | 16.66 | — |
| 40 | 16.19 | 31.12 | — |
| 60 | 23.57 | 41.39 | — |
| 80 | 30.37 | 50.18 | — |
| **100 (best)** | **33.13** | **53.56** | **77.46** |

Gap vs paper SOTA tables expected under D1–D3 (no SHHQ-occ / parsing / multi-dataset).

## Sign-off

| Item | Status |
|------|--------|
| 5-pass complete | Yes |
| Real-data metrics | Yes |
| Deviations | `DEVIATIONS.md` D1–D8 |
| Upstream | None (`SOURCE_CODE.md`) |
