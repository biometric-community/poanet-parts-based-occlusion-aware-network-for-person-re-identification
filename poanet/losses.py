"""ID CE + batch-hard triplet (paper Sec. III / Implementation)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTriplet(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = float(margin)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=-1)
        dist = torch.cdist(feats, feats, p=2)
        labels = labels.view(-1)
        mask_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_neg = ~mask_pos
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        mask_pos = mask_pos & ~eye
        dist_ap = dist.clone()
        dist_ap[~mask_pos] = -1.0
        hardest_pos, _ = dist_ap.max(dim=1)
        dist_an = dist.clone()
        dist_an[~mask_neg] = 1e6
        hardest_neg, _ = dist_an.min(dim=1)
        valid = (hardest_pos > -0.5) & (hardest_neg < 1e5)
        if valid.sum() == 0:
            return dist.new_zeros(())
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin).mean()


def poanet_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    ce: nn.Module,
    triplet: BatchHardTriplet,
    lambda_tri: float = 1.0,
) -> Dict[str, torch.Tensor]:
    l_ce = ce(outputs["logits"], labels)
    l_tri = triplet(outputs["feat"], labels)
    loss = l_ce + lambda_tri * l_tri
    return {"loss": loss, "ce": l_ce.detach(), "tri": l_tri.detach()}


def occlusion_extent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. (5): mean absolute error over the three body-part extents."""
    return F.l1_loss(pred, target)
