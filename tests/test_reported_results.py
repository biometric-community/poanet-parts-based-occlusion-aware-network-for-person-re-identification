"""REPORT.md and README.md, the committed outputs of the full run."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ours(text: str) -> list[float]:
    match = re.search(r"\| \*\*Ours\*\* \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", text)
    assert match, "results row missing"
    return [float(g) for g in match.groups()]


def test_readme_and_report_agree(report):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ours(readme) == pytest.approx(ours(report), abs=0.06)


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    expected = targets["published_values"]
    assert ours(report)[:2] == pytest.approx([expected["map"], expected["rank1"]], abs=1.0)
