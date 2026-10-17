"""Extract embeddings for a folder of images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poanet.data import build_transforms, resolve_market_root, Market1501
from poanet.models import build_poanet
from poanet.train import load_config


class ImageFolder(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(list(Path(root).glob("*.jpg")) + list(Path(root).glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"image": self.transform(img), "path": str(self.paths[i])}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    root = resolve_market_root(cfg)
    nclass = Market1501(root, "train", transform=build_transforms(cfg, False)).num_classes
    model = build_poanet(cfg, num_classes=nclass).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    ds = ImageFolder(Path(args.input), build_transforms(cfg, False))
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            feat = model(batch["image"].to(device))["feat"].cpu()
            for p, f in zip(batch["path"], feat):
                rows.append({"path": p, "feat": f.tolist()})
    out = Path(cfg["paths"]["pred_dir"]) / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"Wrote {len(rows)} embeddings → {out}")


if __name__ == "__main__":
    main()
