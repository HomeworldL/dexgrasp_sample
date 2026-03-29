#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete pc_warp contents except global_pc.npy under a dataset root."
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
        help="Name of the render output subdirectory to clean.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched files. Without this flag the script is dry-run only.",
    )
    return parser.parse_args()


def _collect_delete_targets(pc_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in pc_dir.rglob("*")
        if path.is_file() and path.name != "global_pc.npy"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    targets = sorted(path for path in dataset_dir.glob(f"*/scale*/{args.subdir}") if path.is_dir())
    total_dirs = len(targets)
    delete_targets: list[Path] = []
    total_bytes = 0
    for path in targets:
        dir_targets = _collect_delete_targets(path)
        delete_targets.extend(dir_targets)
        total_bytes += sum(file_path.stat().st_size for file_path in dir_targets)

    action = "delete" if args.execute else "dry-run"
    print(
        f"[delete_pc_warp_dirs] mode={action} dataset_dir={dataset_dir} "
        f"matched_dirs={total_dirs} matched_files={len(delete_targets)} total_bytes={total_bytes}"
    )
    for path in delete_targets:
        print(path)
        if args.execute:
            path.unlink()

    if args.execute:
        for pc_dir in targets:
            empty_dirs = sorted(
                (path for path in pc_dir.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            )
            for empty_dir in empty_dirs:
                if not any(empty_dir.iterdir()):
                    empty_dir.rmdir()
        print("[delete_pc_warp_dirs] deletion finished")


if __name__ == "__main__":
    main()
