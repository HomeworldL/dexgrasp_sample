#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete pc_warp subdirectories under a dataset root."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/graspdata_YCB_liberhand",
        help="Dataset root containing per-object scale directories.",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="pc_warp",
        help="Name of the render output subdirectory to delete.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched directories. Without this flag the script is dry-run only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    targets = sorted(path for path in dataset_dir.glob(f"*/scale*/{args.subdir}") if path.is_dir())
    total = len(targets)
    total_bytes = 0
    for path in targets:
        total_bytes += sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())

    action = "delete" if args.execute else "dry-run"
    print(
        f"[delete_pc_warp_dirs] mode={action} dataset_dir={dataset_dir} "
        f"matched_dirs={total} total_bytes={total_bytes}"
    )
    for path in targets:
        print(path)
        if args.execute:
            shutil.rmtree(path)

    if args.execute:
        print("[delete_pc_warp_dirs] deletion finished")


if __name__ == "__main__":
    main()
