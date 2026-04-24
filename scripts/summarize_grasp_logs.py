#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

SUMMARY_RE = re.compile(
    r"\bvalid=(?P<valid>\d+)\b.*?\btotal_elapsed=(?P<total_elapsed>[0-9]+(?:\.[0-9]+)?)s\b"
)

DEFAULT_LOG_DIRS = [
    "logs/run/graspdata_YCB_liberhand",
    "logs/run/graspdata_YCB_liberhand_right",
]


@dataclass
class DirSummary:
    directory: Path
    log_files: int = 0
    parsed_files: int = 0
    skipped_files: int = 0
    total_valid: int = 0
    total_elapsed_sec: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize total elapsed time and feasible grasp count from run logs."
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        default=DEFAULT_LOG_DIRS,
        help="Log directories to summarize.",
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern for log files inside each log directory.",
    )
    return parser.parse_args()


def parse_log(log_path: Path) -> tuple[int, float] | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match:
            valid = int(match.group("valid"))
            total_elapsed_sec = float(match.group("total_elapsed"))
            return valid, total_elapsed_sec
    return None


def summarize_directory(log_dir: Path, pattern: str) -> DirSummary:
    summary = DirSummary(directory=log_dir)
    files = sorted(log_dir.glob(pattern))
    summary.log_files = len(files)
    for log_path in files:
        parsed = parse_log(log_path)
        if parsed is None:
            summary.skipped_files += 1
            continue
        valid, elapsed = parsed
        summary.parsed_files += 1
        summary.total_valid += valid
        summary.total_elapsed_sec += elapsed
    return summary


def format_duration(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{seconds:.2f}s ({hours:.2f}h)"


def main() -> None:
    args = parse_args()
    summaries = [summarize_directory(Path(p), args.pattern) for p in args.log_dirs]

    total_valid = sum(x.total_valid for x in summaries)
    total_elapsed = sum(x.total_elapsed_sec for x in summaries)
    total_files = sum(x.log_files for x in summaries)
    total_parsed = sum(x.parsed_files for x in summaries)
    total_skipped = sum(x.skipped_files for x in summaries)

    for item in summaries:
        print(f"[{item.directory}]")
        print(f"  logs: {item.log_files} (parsed={item.parsed_files}, skipped={item.skipped_files})")
        print(f"  total_valid: {item.total_valid}")
        print(f"  total_elapsed: {format_duration(item.total_elapsed_sec)}")

    print("[ALL]")
    print(f"  logs: {total_files} (parsed={total_parsed}, skipped={total_skipped})")
    print(f"  total_valid: {total_valid}")
    print(f"  total_elapsed: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
