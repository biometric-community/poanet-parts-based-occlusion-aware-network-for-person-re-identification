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
    occ_gt: torch.Tensor | None = None,
    lambda_tri: float = 1.0,
    lambda_occ: float = 0.5,
) -> Dict[str, torch.Tensor]:
    l_ce = ce(outputs["logits"], labels)
    l_tri = triplet(outputs["feat"], labels)
    loss = l_ce + lambda_tri * l_tri
    l_occ = outputs["feat"].new_zeros(())
    if occ_gt is not None and outputs.get("occ_map") is not None:
        pred = outputs["occ_map"]
        if pred.shape[-2:] != occ_gt.shape[-2:]:
            pred = F.interpolate(pred, size=occ_gt.shape[-2:], mode="bilinear", align_corners=False)
        # BCE unsafe under autocast — force fp32 context
        with torch.amp.autocast("cuda", enabled=False):
            l_occ = F.binary_cross_entropy(pred.float(), occ_gt.float())
        loss = loss + lambda_occ * l_occ
    return {"loss": loss, "ce": l_ce.detach(), "tri": l_tri.detach(), "occ": l_occ.detach()}
