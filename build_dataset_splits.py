#!/usr/bin/env python3
"""Build train/test split JSON files from completed grasp and render outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    dataset_tag_from_config,
    list_existing_files,
    load_config,
    relpath_str,
)
from utils.utils_sample import grasp_h5_nonempty


def _collect_entry_record(
    entry: Dict,
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    output_dir = Path(str(entry["output_dir_abs"])).resolve()
    grasp_h5_path = output_dir / "grasp.h5"
    grasp_npy_path = output_dir / "grasp.npy"
    if not grasp_h5_path.exists():
        return None, f"missing {grasp_h5_path.name}"
    if not grasp_npy_path.exists():
        return None, f"missing {grasp_npy_path.name}"

    render_dir = output_dir / render_subdir
    cam_in_path = render_dir / "cam_in.npy"
    if not cam_in_path.exists():
        return None, f"missing {render_subdir}/cam_in.npy"

    partial_pc_paths = list_existing_files(render_dir, "partial_pc_")
    partial_pc_cam_paths = list_existing_files(render_dir, "partial_pc_cam_")
    cam_ex_paths = list_existing_files(render_dir, "cam_ex_")

    partial_pc_paths = [
        path for path in partial_pc_paths if not path.name.startswith("partial_pc_cam_")
    ]

    if not partial_pc_paths:
        return None, f"missing {render_subdir}/partial_pc_*.npy"
    if not partial_pc_cam_paths:
        return None, f"missing {render_subdir}/partial_pc_cam_*.npy"
    if not cam_ex_paths:
        return None, f"missing {render_subdir}/cam_ex_*.npy"

    world_suffixes = [path.stem[len("partial_pc_") :] for path in partial_pc_paths]
    cam_suffixes = [path.stem[len("partial_pc_cam_") :] for path in partial_pc_cam_paths]
    ex_suffixes = [path.stem[len("cam_ex_") :] for path in cam_ex_paths]
    if world_suffixes != cam_suffixes or world_suffixes != ex_suffixes:
        return None, f"mismatched render view files under {render_subdir}"

    record = {
        "global_id": int(entry["global_id"]),
        "object_scale_key": str(entry["object_scale_key"]),
        "object_name": str(entry["object_name"]),
        "output_path": relpath_str(output_dir, dataset_dir),
        "coacd_path": relpath_str(Path(str(entry["coacd_abs"])), dataset_dir),
        "mjcf_path": relpath_str(Path(str(entry["mjcf_abs"])), dataset_dir),
        "grasp_h5_path": relpath_str(grasp_h5_path, dataset_dir),
        "grasp_npy_path": relpath_str(grasp_npy_path, dataset_dir),
        "partial_pc_path": [relpath_str(path, dataset_dir) for path in partial_pc_paths],
        "partial_pc_cam_path": [
            relpath_str(path, dataset_dir) for path in partial_pc_cam_paths
        ],
        "cam_ex_path": [relpath_str(path, dataset_dir) for path in cam_ex_paths],
        "cam_in": relpath_str(cam_in_path, dataset_dir),
        "scale": float(entry["scale"]),
    }
    return record, None


def build_split_records(
    entries: Sequence[Dict],
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    records: List[Dict] = []
    skipped: List[Tuple[str, str]] = []
    for entry in sorted(entries, key=lambda it: int(it["global_id"])):
        record, reason = _collect_entry_record(
            entry=entry,
            dataset_dir=dataset_dir,
            render_subdir=render_subdir,
        )
        if record is None:
            skipped.append((str(entry["object_scale_key"]), str(reason)))
            continue
        records.append(record)
    return records, skipped


def split_records_by_object(records: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    object_names = sorted({str(record["object_name"]) for record in records})
    if len(object_names) <= 1:
        test_objects = set()
    else:
        test_count = max(1, int(round(len(object_names) * 0.2)))
        test_objects = set(object_names[-test_count:])

    train_records = [
        record for record in records if str(record["object_name"]) not in test_objects
    ]
    test_records = [
        record for record in records if str(record["object_name"]) in test_objects
    ]
    return train_records, test_records


def filter_nonempty_grasp_records(
    records: Sequence[Dict],
    dataset_dir: Path,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    kept: List[Dict] = []
    removed: List[Tuple[str, str]] = []
    for record in records:
        h5_path = dataset_dir / str(record["grasp_h5_path"])
        ok, reason = grasp_h5_nonempty(h5_path)
        if ok:
            kept.append(record)
            continue
        removed.append((str(record["object_scale_key"]), reason))
    return kept, removed


def write_split_jsons(
    entries: Sequence[Dict],
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[Path, Path]:
    records, skipped = build_split_records(
        entries=entries,
        dataset_dir=dataset_dir,
        render_subdir=render_subdir,
    )
    train_records, test_records = split_records_by_object(records)
    train_records, empty_train = filter_nonempty_grasp_records(train_records, dataset_dir)
    test_records, empty_test = filter_nonempty_grasp_records(test_records, dataset_dir)
    empty_records = empty_train + empty_test

    train_path = dataset_dir / "train.json"
    test_path = dataset_dir / "test.json"
    train_path.write_text(
        json.dumps(train_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps(test_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote dataset splits under {dataset_dir}: "
        f"train={len(train_records)} test={len(test_records)} "
        f"skipped={len(skipped)} empty={len(empty_records)}"
    )
    for object_scale_key, reason in skipped:
        print(f"[SKIP] {object_scale_key}: {reason}")
    for object_scale_key, reason in empty_records:
        print(f"[EMPTY] {object_scale_key}: {reason}")
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/test split JSON files from completed grasp and render outputs."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_tag = dataset_tag_from_config(args.config)
    ds = DatasetObjects(
        cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )
    dataset_root = Path(cfg.get("output", {}).get("dataset_root", "datasets")).resolve()
    dataset_dir = dataset_root / dataset_tag
    write_split_jsons(
        entries=ds.get_entries(),
        dataset_dir=dataset_dir,
        render_subdir=str(cfg["warp_render"]["output_subdir"]),
    )


if __name__ == "__main__":
    main()
