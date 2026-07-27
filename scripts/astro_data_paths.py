#!/usr/bin/env python3
"""Resolve large astro_navigation data paths outside the Git worktree.

Resolution order is environment, the ignored per-worktree local JSON file, then
portable repository fallbacks. This keeps a clone runnable while allowing real
datasets and benchmark artifacts to live on an external SSD.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO_ROOT / ".astro_navigation.local.json"


def _local_config() -> dict:
    if not LOCAL_CONFIG.exists():
        return {}
    with LOCAL_CONFIG.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{LOCAL_CONFIG} must contain a JSON object")
    return value


def data_root() -> Path:
    cfg = _local_config()
    return Path(
        os.environ.get(
            "ASTRO_NAV_DATA_ROOT",
            cfg.get("data_root", REPO_ROOT / "datasets"),
        )
    ).expanduser()


def output_root() -> Path:
    cfg = _local_config()
    return Path(
        os.environ.get(
            "ASTRO_NAV_OUTPUT_ROOT",
            cfg.get("output_root", REPO_ROOT / "outputs"),
        )
    ).expanduser()
