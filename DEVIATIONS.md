# Deviations from the paper

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | SHHQ-occ (1.2M) for ODN | Synthetic rectangular occlusion on Market | SHHQ-occ unresolved / not registered |
| D2 | Occluded-Duke, MSMT17, CUHK03, Duke, Occluded-ReID, … | **Market-1501 only** | Only Market under `projects/datasets/` for this run |
| D3 | Human-parsing–guided anatomical parts | Horizontal 3-stripe PAFE | No parsing labels in Market package |
| D4 | Super-resolution preprocessing | Omitted | Optional SR branch; not required for Core Market train loop |
| D5 | Exact ODN architecture / SHHQ MAE tables | ResNet-18 encoder–decoder + BCE | Architecture under-specified in extract; functional occlusion prior |
| D6 | Protocol 2 (75 ep GE → 25 ep PAFE) | Joint ODN+PAFE from start (configurable) | Faster convergence on Market; `phase1_epochs` retained |
| D7 | Two RTX 3080 | Single GPU `CUDA_VISIBLE_DEVICES=0` | Available hardware |
| D8 | Jaccard re-ranking in some tables | Optional off by default | Standard CMC/mAP without RK for primary REPORT |

## Hyperparameters taken from Implementation

- 384×128, ResNet-50, SGD 0.0002, drops at 90/95, 100 epochs, P=16 K=4, triplet + CE
