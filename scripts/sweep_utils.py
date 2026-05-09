"""Shared utilities for grasp parameter sweep scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+)\s+no_col=(?P<no_col>\d+)\s+valid=(?P<valid>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+time=(?P<sim_time>[0-9.]+)s\s+total_elapsed=(?P<total_elapsed>[0-9.]+)s\s+"
    r"stop_reason=(?P<stop_reason>\w+)"
)


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def parse_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        match = SUMMARY_RE.search(line)
        if match:
            return {
                "samples": int(match.group("samples")),
                "no_col": int(match.group("no_col")),
                "valid": int(match.group("valid")),
                "fail": int(match.group("fail")),
                "sim_time_sec": float(match.group("sim_time")),
                "total_elapsed_sec": float(match.group("total_elapsed")),
                "stop_reason": match.group("stop_reason"),
            }
    return {
        "samples": None,
        "no_col": None,
        "valid": None,
        "fail": None,
        "sim_time_sec": None,
        "total_elapsed_sec": None,
        "stop_reason": None,
    }
