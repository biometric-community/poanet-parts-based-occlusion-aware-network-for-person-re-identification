"""ODN and PAFE behaviour and the evaluation protocol, checked against the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from poanet.data.market1501 import Market1501, parse_market_name
from poanet.losses import BatchHardTriplet
from poanet.metrics import evaluate_rank
from poanet.models.poanet import (
    OcclusionDetector,
    PartAwareFeatureExtractor,
    POANet,
    stripe_extents,
    synthetic_occlusion_batch,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pafe():
    torch.manual_seed(0)
    return PartAwareFeatureExtractor(num_classes=4, pretrained=False, feat_dim=16).eval()


def occlusion(rows: slice) -> torch.Tensor:
    occ = torch.zeros(1, 1, 96, 32)
    occ[:, :, rows, :] = 1.0
    return occ


def part_norms(pafe, occ) -> list[float]:
    torch.manual_seed(1)
    with torch.no_grad():
        out = pafe(torch.rand(1, 3, 96, 32), occ_extents=None if occ is None else stripe_extents(occ))
    return [p.norm().item() for p in out["parts"]]


def test_there_are_head_torso_and_leg_parts(pafe):
    assert len(part_norms(pafe, None)) == 3


@pytest.mark.parametrize("part,rows", [(0, slice(0, 32)), (1, slice(32, 64)), (2, slice(64, 96))])
def test_occluded_part_is_down_weighted(pafe, part, rows):
    """Abstract: corrupted regions are down-weighted, visible regions emphasised."""
    clean, occluded = part_norms(pafe, occlusion(slice(0, 0))), part_norms(pafe, occlusion(rows))
    assert occluded[part] == pytest.approx(0.05 * clean[part], rel=1e-3)
    others = [i for i in range(3) if i != part]
    assert all(occluded[i] == pytest.approx(clean[i], rel=1e-3) for i in others)


def test_synthetic_mask_marks_exactly_the_changed_pixels():
    torch.manual_seed(2)
    images = torch.rand(4, 3, 96, 32)
    occluded, mask = synthetic_occlusion_batch(images)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0
    changed = (occluded != images).any(dim=1, keepdim=True).float()
    assert torch.equal(changed, mask)


def test_odn_predicts_one_occlusion_extent_per_body_part():
    """Eq. (4): the ODN outputs three values, for head, torso and legs."""
    odn = OcclusionDetector(pretrained=False, input_size=(96, 32)).eval()
    with torch.no_grad():
        out = odn(torch.rand(2, 3, 96, 32))
    assert out.shape == (2, 3)
    assert ((out >= 0) & (out <= 1)).all()


def test_ground_truth_extents_are_the_occluded_fraction_of_each_part():
    assert stripe_extents(occlusion(slice(0, 16))).tolist() == [[0.5, 0.0, 0.0]]


def test_odn_stays_fixed_during_reid_training():
    """Sec. V-B: the ODN is trained separately and kept fixed."""
    model = POANet(4, pretrained=False, feat_dim=16, input_size=(96, 32))
    model.freeze_odn()
    model.train()
    before = {k: v.clone() for k, v in model.odn.state_dict().items()}
    model(torch.rand(4, 3, 96, 32))
    assert not model.odn.training
    assert not any(p.requires_grad for p in model.odn.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.odn.state_dict().items())


def test_fusion_uses_the_occlusion_features():
    """Eqs. (7)-(8): the 1x1 convolution sees both backbones and starts from the ReID features."""
    torch.manual_seed(3)
    pafe = PartAwareFeatureExtractor(num_classes=4, pretrained=False, feat_dim=16).eval()
    x = torch.rand(2, 3, 96, 32)
    occ = torch.rand(2, 1024, 6, 2)
    with torch.no_grad():
        assert torch.allclose(pafe(x)["global"], pafe(x, occ_features=occ)["global"], atol=1e-5)
        pafe.fuse.weight[:, 1024:] = 0.01
        assert not torch.allclose(pafe(x)["global"], pafe(x, occ_features=occ)["global"], atol=1e-5)


@pytest.mark.parametrize("key", ["batch_size", "num_instances"])
def test_batch_composition_matches_the_paper(default_cfg, targets, key):
    assert default_cfg["train"][key] == targets["training"][key]


def test_batch_hard_triplet_is_zero_when_identities_are_separated():
    feats = torch.tensor([[1.0, 0.0], [1.0, 0.01], [-1.0, 0.0], [-1.0, 0.01]])
    assert BatchHardTriplet(0.3)(feats, torch.tensor([0, 0, 1, 1])).item() == pytest.approx(0.0)


def test_market_names_are_parsed():
    assert parse_market_name("0002_c1s1_000451_03.jpg") == (2, 1)


def rank(q_pid, q_cam, g_pids, g_cams, g_feats) -> dict:
    return evaluate_rank(
        torch.tensor([[1.0, 0.0]]), np.array([q_pid]), np.array([q_cam]), torch.tensor(g_feats),
        np.array(g_pids), np.array(g_cams), ranks=[1],
    )


def test_same_camera_matches_are_excluded():
    result = rank(1, 1, [1, 1, 2], [1, 2, 2], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    assert result["Rank-1"] == 100.0


def test_junk_images_do_not_count_against_a_query(tmp_path):
    """Market-1501 protocol: id -1 images are junk and are ignored, unlike id 0 distractors."""
    (tmp_path / "bounding_box_test").mkdir()
    for name in ("0001_c2s1_000001_01.jpg", "-1_c2s1_000002_01.jpg", "0000_c2s1_000003_01.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / "bounding_box_test" / name)
    gallery = Market1501(tmp_path, "gallery")
    feats = {-1: [1.0, 0.0], 1: [0.8, 0.6], 0: [0.0, 1.0]}
    assert rank(1, 1, gallery.pids, gallery.camids, [feats[p] for p in gallery.pids])["Rank-1"] == 100.0


def test_checkpoints_are_not_selected_on_query_and_gallery():
    """Keeping the best epoch by test mAP makes the reported test mAP optimistic."""
    source = (ROOT / "poanet" / "train.py").read_text(encoding="utf-8")
    assert 'metrics["mAP"] > best["mAP"]' not in source
