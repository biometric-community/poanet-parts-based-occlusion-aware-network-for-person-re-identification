# POANET: Parts-Based Occlusion Aware Network for Person Re-Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of Dutta et al. (TBIOM 2026): Occlusion Detection Network (ODN) + Part Aware Feature Extractor (PAFE) for person ReID on **Market-1501**.

**Repository:** https://github.com/biometric-community/poanet-parts-based-occlusion-aware-network-for-person-re-identification

**Paper:** [10.1109/TBIOM.2025.3641048](https://doi.org/10.1109/TBIOM.2025.3641048)

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/report.sh
```

## Results (Market-1501, this rebuild)

| | mAP | Rank-1 | Rank-5 |
|--|--:|--:|--:|
| **Ours** | 33.1 | 53.6 | 77.5 |

See `REPORT.md` and `DEVIATIONS.md` (Market-only; synthetic occlusion instead of SHHQ-occ).
