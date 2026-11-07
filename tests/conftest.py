"""Shared test setup: repo root on sys.path, paper targets, REPORT.md text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def targets() -> dict:
    return yaml.safe_load((ROOT / "tests" / "paper_targets.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def report() -> str:
    return (ROOT / "REPORT.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    name = "default.yaml" if (ROOT / "configs" / "default.yaml").is_file() else "full.yaml"
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
