"""POANET: ODN + OAM + PAFE on ResNet-50 (paper Sec. III)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class OcclusionDetector(nn.Module):
    """Lightweight ODN: ResNet stem + decoder → occlusion probability map.

    Paper trains ODN on SHHQ-occ; we use synthetic rect masks (see DEVIATIONS.md).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        bb = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.relu,
            bb.maxpool,
            bb.layer1,
            bb.layer2,
            bb.layer3,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,1,H',W') occlusion probability in [0,1]
        h = self.encoder(x)
        m = self.decoder(h)
        return F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)


class PartAwareFeatureExtractor(nn.Module):
    """PAFE: ResNet-50 → global + head/torso/legs stripe embeddings."""

    def __init__(self, num_classes: int, pretrained: bool = True, feat_dim: int = 256):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # (B,2048,H,W)
        self.feat_dim = int(feat_dim)
        in_dim = 2048
        self.global_fc = nn.Linear(in_dim, self.feat_dim, bias=False)
        self.part_fcs = nn.ModuleList([nn.Linear(in_dim, self.feat_dim, bias=False) for _ in range(3)])
        self.bn = nn.BatchNorm1d(self.feat_dim * 4)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.feat_dim * 4, int(num_classes), bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for m in [self.global_fc, *self.part_fcs]:
            nn.init.normal_(m.weight, std=0.001)

    def _pool_parts(self, fmap: torch.Tensor):
        # fmap: (B,C,H,W) → 3 horizontal stripes (head, torso, legs)
        b, c, h, w = fmap.shape
        # ensure divisible
        h3 = h // 3
        parts = []
        for i in range(3):
            sl = fmap[:, :, i * h3 : (i + 1) * h3 if i < 2 else h, :]
            parts.append(F.adaptive_avg_pool2d(sl, 1).flatten(1))
        g = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return g, parts

    def forward(
        self,
        x: torch.Tensor,
        occ_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(x)
        g, parts = self._pool_parts(fmap)
        g = self.global_fc(g)
        p_emb = [fc(p) for fc, p in zip(self.part_fcs, parts)]

        # OAM: down-weight parts by mean occlusion in corresponding stripe
        if occ_map is not None:
            b, _, oh, ow = occ_map.shape
            h3 = oh // 3
            weights = []
            for i in range(3):
                stripe = occ_map[:, :, i * h3 : (i + 1) * h3 if i < 2 else oh, :]
                # visibility = 1 - mean occlusion
                vis = (1.0 - stripe.mean(dim=(1, 2, 3))).clamp(0.05, 1.0)
                weights.append(vis.unsqueeze(1))
            p_emb = [p * w for p, w in zip(p_emb, weights)]
            g_vis = (1.0 - occ_map.mean(dim=(1, 2, 3))).clamp(0.05, 1.0).unsqueeze(1)
            g = g * g_vis

        feat = torch.cat([g, *p_emb], dim=1)
        feat = self.bn(feat)
        feat_n = F.normalize(feat, dim=1)
        logits = self.classifier(feat)
        return {
            "feat": feat_n,
            "feat_raw": feat,
            "logits": logits,
            "parts": p_emb,
            "global": g,
        }


class POANet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, feat_dim: int = 256):
        super().__init__()
        self.odn = OcclusionDetector(pretrained=pretrained)
        self.pafe = PartAwareFeatureExtractor(num_classes, pretrained=pretrained, feat_dim=feat_dim)

    def forward(self, x: torch.Tensor, use_odn: bool = True) -> Dict[str, torch.Tensor]:
        occ = self.odn(x) if use_odn else None
        out = self.pafe(x, occ_map=occ)
        out["occ_map"] = occ
        return out


def build_poanet(cfg: dict, num_classes: int) -> POANet:
    mcfg = cfg.get("model", {})
    return POANet(
        num_classes=num_classes,
        pretrained=bool(mcfg.get("pretrained", True)),
        feat_dim=int(mcfg.get("feat_dim", 256)),
    )


def synthetic_occlusion_batch(
    images: torch.Tensor,
    max_holes: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random rectangular occlusions; return (occluded_images, gt_mask)."""
    b, c, h, w = images.shape
    out = images.clone()
    mask = torch.zeros(b, 1, h, w, device=images.device, dtype=images.dtype)
    for i in range(b):
        n = int(torch.randint(1, max_holes + 1, (1,)).item())
        for _ in range(n):
            rh = int(torch.randint(h // 10, max(h // 3, h // 10 + 1), (1,)).item())
            rw = int(torch.randint(w // 5, max(w // 2, w // 5 + 1), (1,)).item())
            y0 = int(torch.randint(0, max(h - rh, 1), (1,)).item())
            x0 = int(torch.randint(0, max(w - rw, 1), (1,)).item())
            fill = torch.rand(c, 1, 1, device=images.device)
            out[i, :, y0 : y0 + rh, x0 : x0 + rw] = fill
            mask[i, :, y0 : y0 + rh, x0 : x0 + rw] = 1.0
    return out, mask
