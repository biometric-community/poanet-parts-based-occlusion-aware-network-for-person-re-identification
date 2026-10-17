# IMPLEMENTATION_PLAN — POANET

Paper: Dutta, Jonnalagedda, Bhanu — *POANET: Parts-Based Occlusion Aware Network for Person Re-Identification*, TBIOM 2026.  
DOI: [10.1109/TBIOM.2025.3641048](https://doi.org/10.1109/TBIOM.2025.3641048)

## Method modules

| Module | Paper | Ours |
|--------|-------|------|
| ODN | ResNet-based occlusion map on SHHQ-occ | ResNet-18 encoder–decoder; synthetic rect masks on Market |
| OAM | Down-weight occluded regions | Per-stripe visibility × part/global embeddings |
| PAFE | Head / torso / legs + global | Horizontal 3-stripe pooling on ResNet-50 fmap + global |
| Loss | Triplet (+ ID) | Softmax CE + batch-hard triplet + BCE on occ map |
| Input | 384×128, ResNet-50 | Same |
| Optim | SGD mom 0.9, wd 5e-4, lr 2e-4, drop @90/95, 100 ep | Same |
| Batch | P×K mini-batch | 16×4 = 64 |

## Figure inventory (Results → our plots)

| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Table V / Market column | bars mAP/R1 | `outputs/figures/market_map_r1.svg` |
| Training dynamics | curves | `outputs/figures/train_curves.svg` |

## Scope

- **In:** Market-1501 train + query/gallery eval (size ≪ 5 GiB → full train).
- **Out / D:** SHHQ-occ, Occluded-Duke, MSMT17, CUHK03, Duke, super-resolution branch (see `DEVIATIONS.md`).

## Upstream

No official POANET ReID code found (`SOURCE_CODE.md`).
