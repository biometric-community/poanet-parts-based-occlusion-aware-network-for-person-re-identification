"""Generate REPORT.md + SVG figures from real logs (dev-plot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poanet.plot_style import (
    BAR_COLOR,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FIGSIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    apply_rcparams,
    apply_style,
)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def plot_market_bars(metrics: dict, figdir: Path):
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["mAP", "Rank-1"]
    ours = [float(metrics.get("mAP", 0)), float(metrics.get("Rank-1", 0))]
    x = np.arange(len(labels))
    ax.bar(x, ours, 0.5, color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Market-1501 POANET (Ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    out = figdir / "market_map_r1.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_train_curves(history: list, figdir: Path):
    apply_rcparams()
    ep = [h["epoch"] for h in history]
    loss = [h.get("loss", 0) for h in history]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ep, loss, color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="Train loss")
    maps = [(h["epoch"], h["mAP"]) for h in history if "mAP" in h]
    if maps:
        ax2 = ax.twinx()
        ax2.plot(
            [e for e, _ in maps],
            [m for _, m in maps],
            color=MULTI_SERIES_COLORS[1],
            linewidth=LINEWIDTH_MAIN,
            label="mAP",
        )
        ax2.set_ylabel("mAP (%)")
        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lb2, loc="best")
    else:
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    apply_style(ax)
    fig.tight_layout()
    out = figdir / "train_curves.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def write_report(cfg: dict, logs: Path, figdir: Path, report_path: Path):
    hist = load_json(logs / "train_history.json") or []
    summary = load_json(logs / "train_summary.json") or {}
    eval_res = load_json(logs / "eval_results.json") or {}
    size = load_json(logs / "dataset_size.json") or {}
    metrics = eval_res if eval_res.get("mAP") is not None else (summary.get("best") or {})
    figs = []
    if metrics:
        figs.append(plot_market_bars(metrics, figdir))
    if hist:
        figs.append(plot_train_curves(hist, figdir))

    lines = [
        "# Report: POANET",
        "",
        "Market-1501 rebuild of Dutta et al., TBIOM 2026 "
        "(DOI [10.1109/TBIOM.2025.3641048](https://doi.org/10.1109/TBIOM.2025.3641048)).",
        "",
        "## Setup",
        "",
        f"- Image: {cfg['data'].get('height')}×{cfg['data'].get('width')}, ResNet-50 + ODN/PAFE",
        f"- Epochs: {cfg['train']['epochs']} (phase-1 GE {cfg['train'].get('phase1_epochs', 75)})",
        f"- Dataset size gate: {size.get('decision', 'n/a')} ({size.get('total_gib', '?')} GiB)",
        "",
        "## Market-1501 results",
        "",
        "| | mAP | Rank-1 | Rank-5 |",
        "|--|--:|--:|--:|",
        f"| **Ours** | {float(metrics.get('mAP', float('nan'))):.2f} | "
        f"{float(metrics.get('Rank-1', float('nan'))):.2f} | "
        f"{float(metrics.get('Rank-5', float('nan'))):.2f} |",
        "",
        "SHHQ-occ / Occluded-Duke / MSMT17 not used in this rebuild — see `DEVIATIONS.md`.",
        "",
        "## Figures",
        "",
    ]
    for p in figs:
        if p is None:
            continue
        rel = p.relative_to(report_path.parent) if p.is_absolute() else p
        lines += [f"### `{rel}`", "", f"![]({rel})", ""]
    lines += [
        "## Regenerate",
        "",
        "```bash",
        "bash scripts/report.sh",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logs = Path(cfg["paths"]["log_dir"])
    figdir = Path(cfg["paths"]["figure_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    write_report(cfg, logs, figdir, _ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
