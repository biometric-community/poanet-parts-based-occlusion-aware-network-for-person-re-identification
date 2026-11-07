"""POANET: ODN + OAM + PAFE on ResNet-50 (paper Sec. III)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class OcclusionDetector(nn.Module):
    """ODN, Eqs. (2)-(4): ResNet-50 to the third residual block, then one extent per body part.

    Paper trains ODN on SHHQ-occ; we use synthetic rect masks (see DEVIATIONS.md).
    """

    def __init__(self, pretrained: bool = True, input_size: tuple[int, int] = (384, 128)):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-3])  # (B,1024,H/16,W/16)
        self.conv = nn.Conv2d(1024, 2048, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        pooled = (input_size[0] // 32) * (input_size[1] // 32)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * pooled, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def regress(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.pool(self.conv(features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,3) occlusion extents for head, torso and legs, each in [0,1]
        return self.regress(self.features(x))


class PartAwareFeatureExtractor(nn.Module):
    """PAFE: ResNet-50 → global + head/torso/legs stripe embeddings."""

    def __init__(self, num_classes: int, pretrained: bool = True, feat_dim: int = 256):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-3])  # (B,1024,H,W)
        self.layer4 = net.layer4
        # Eqs. (7)-(8), starting from the identity on the ReID features
        self.fuse = nn.Conv2d(2048, 1024, 1, bias=False)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.weight[:, :1024, 0, 0] = torch.eye(1024)
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
        occ_features: Optional[torch.Tensor] = None,
        occ_extents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(x)
        if occ_features is not None:
            fmap = self.fuse(torch.cat([fmap, occ_features], dim=1))
        g, parts = self._pool_parts(self.layer4(fmap))
        g = self.global_fc(g)
        p_emb = [fc(p) for fc, p in zip(self.part_fcs, parts)]

        # Down-weight each part by its predicted occlusion extent
        if occ_extents is not None:
            vis = (1.0 - occ_extents).clamp(0.05, 1.0)
            p_emb = [p * vis[:, i : i + 1] for i, p in enumerate(p_emb)]
            g = g * (1.0 - occ_extents.mean(dim=1, keepdim=True)).clamp(0.05, 1.0)

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
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feat_dim: int = 256,
        input_size: tuple[int, int] = (384, 128),
    ):
        super().__init__()
        self.odn = OcclusionDetector(pretrained=pretrained, input_size=input_size)
        self.pafe = PartAwareFeatureExtractor(num_classes, pretrained=pretrained, feat_dim=feat_dim)
        self.odn_frozen = False

    def freeze_odn(self) -> None:
        """Sec. V-B: the ODN is trained separately and kept fixed during ReID training."""
        self.odn_frozen = True
        self.odn.requires_grad_(False)
        self.odn.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.odn_frozen:
            self.odn.eval()
        return self

    def forward(self, x: torch.Tensor, use_odn: bool = True) -> Dict[str, torch.Tensor]:
        if not use_odn:
            return self.pafe(x)
        with torch.no_grad():
            features = self.odn.features(x)
            extents = self.odn.regress(features)
        out = self.pafe(x, occ_features=features, occ_extents=extents)
        out["occ_extents"] = extents
        return out


def build_poanet(cfg: dict, num_classes: int) -> POANet:
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    return POANet(
        num_classes=num_classes,
        pretrained=bool(mcfg.get("pretrained", True)),
        feat_dim=int(mcfg.get("feat_dim", 256)),
        input_size=(int(dcfg.get("height", 384)), int(dcfg.get("width", 128))),
    )


def stripe_extents(mask: torch.Tensor) -> torch.Tensor:
    """Occluded fraction of the head, torso and leg stripes of a (B,1,H,W) mask."""
    h3 = mask.shape[2] // 3
    stripes = [mask[:, :, i * h3 : (i + 1) * h3 if i < 2 else mask.shape[2], :] for i in range(3)]
    return torch.stack([s.mean(dim=(1, 2, 3)) for s in stripes], dim=1)


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
