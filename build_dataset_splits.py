#!/usr/bin/env python3
"""Build train/test split JSON files from completed grasp and render outputs."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_generated_dataset_root_cfg,
    data_run_scales_cfg,
    data_use_native_asset_cfg,
    data_verbose_cfg,
    graspdata_tag_cfg,
    list_existing_files,
    load_run_config,
    objdata_tag_cfg,
    relpath_str,
)
from utils.utils_sample import grasp_h5_nonempty


def _collect_entry_record(
    entry: Dict,
    dataset_dir: Path,
    render_subdir: str,
    grasp_h5_name: str,
    grasp_npy_name: str,
    grasp_fail_h5_name: str,
    grasp_fail_npy_name: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    output_dir = Path(str(entry["output_dir_abs"])).resolve()
    asset_dir = Path(str(entry.get("asset_dir_abs", entry["output_dir_abs"]))).resolve()
    grasp_h5_path = output_dir / str(grasp_h5_name)
    grasp_npy_path = output_dir / str(grasp_npy_name)
    grasp_fail_h5_path = output_dir / str(grasp_fail_h5_name)
    grasp_fail_npy_path = output_dir / str(grasp_fail_npy_name)
    if not grasp_h5_path.exists():
        return None, f"missing {grasp_h5_path.name}"
    if not grasp_npy_path.exists():
        return None, f"missing {grasp_npy_path.name}"
    if not grasp_fail_h5_path.exists():
        return None, f"missing {grasp_fail_h5_path.name}"
    if not grasp_fail_npy_path.exists():
        return None, f"missing {grasp_fail_npy_path.name}"

    render_dir = asset_dir / render_subdir
    cam_in_path = render_dir / "cam_in.npy"
    global_pc_path = render_dir / "global_pc.npy"
    if not cam_in_path.exists():
        return None, f"missing {render_subdir}/cam_in.npy"
    if not global_pc_path.exists():
        return None, f"missing {render_subdir}/global_pc.npy"

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
    cam_suffixes = [
        path.stem[len("partial_pc_cam_") :] for path in partial_pc_cam_paths
    ]
    ex_suffixes = [path.stem[len("cam_ex_") :] for path in cam_ex_paths]
    if world_suffixes != cam_suffixes or world_suffixes != ex_suffixes:
        return None, f"mismatched render view files under {render_subdir}"

    record = {
        "global_id": int(entry["global_id"]),
        "object_scale_key": str(entry["object_scale_key"]),
        "object_name": str(entry["object_name"]),
        "output_path": relpath_str(output_dir, dataset_dir),
        "asset_path": relpath_str(asset_dir, dataset_dir),
        "coacd_path": relpath_str(Path(str(entry["coacd_abs"])), dataset_dir),
        "mjcf_path": relpath_str(Path(str(entry["mjcf_abs"])), dataset_dir),
        "grasp_h5_path": relpath_str(grasp_h5_path, dataset_dir),
        "grasp_npy_path": relpath_str(grasp_npy_path, dataset_dir),
        "grasp_h5_fail_path": relpath_str(grasp_fail_h5_path, dataset_dir),
        "grasp_fail_npy_path": relpath_str(grasp_fail_npy_path, dataset_dir),
        "partial_pc_path": [
            relpath_str(path, dataset_dir) for path in partial_pc_paths
        ],
        "partial_pc_cam_path": [
            relpath_str(path, dataset_dir) for path in partial_pc_cam_paths
        ],
        "cam_ex_path": [relpath_str(path, dataset_dir) for path in cam_ex_paths],
        "cam_in": relpath_str(cam_in_path, dataset_dir),
        "global_pc_path": relpath_str(global_pc_path, dataset_dir),
        "scale_tag": str(entry.get("scale_tag", "")),
        "scale": None if entry.get("scale") is None else float(entry["scale"]),
    }
    return record, None


def build_split_records(
    entries: Sequence[Dict],
    dataset_dir: Path,
    render_subdir: str,
    grasp_h5_name: str = "grasp.h5",
    grasp_npy_name: str = "grasp.npy",
    grasp_fail_h5_name: str = "grasp_fail.h5",
    grasp_fail_npy_name: str = "grasp_fail.npy",
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    records: List[Dict] = []
    skipped: List[Tuple[str, str]] = []
    for entry in sorted(entries, key=lambda it: int(it["global_id"])):
        record, reason = _collect_entry_record(
            entry=entry,
            dataset_dir=dataset_dir,
            render_subdir=render_subdir,
            grasp_h5_name=grasp_h5_name,
            grasp_npy_name=grasp_npy_name,
            grasp_fail_h5_name=grasp_fail_h5_name,
            grasp_fail_npy_name=grasp_fail_npy_name,
        )
        if record is None:
            skipped.append((str(entry["object_scale_key"]), str(reason)))
            continue
        records.append(record)
    return records, skipped


def split_records_by_object(
    records: Sequence[Dict],
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    object_names = sorted({str(record["object_name"]) for record in records})
    random.Random(int(seed)).shuffle(object_names)
    if len(object_names) <= 1:
        test_objects = set()
    else:
        test_count = max(1, int(round(len(object_names) * 0.2)))
        test_objects = set(object_names[:test_count])

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
    split_seed: int,
    grasp_h5_name: str,
    grasp_npy_name: str,
    grasp_fail_h5_name: str,
    grasp_fail_npy_name: str,
) -> Tuple[Path, Path]:
    records, skipped = build_split_records(
        entries=entries,
        dataset_dir=dataset_dir,
        render_subdir=render_subdir,
        grasp_h5_name=grasp_h5_name,
        grasp_npy_name=grasp_npy_name,
        grasp_fail_h5_name=grasp_fail_h5_name,
        grasp_fail_npy_name=grasp_fail_npy_name,
    )
    train_records, test_records = split_records_by_object(records, seed=split_seed)
    train_records, empty_train = filter_nonempty_grasp_records(
        train_records, dataset_dir
    )
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

    cfg = load_run_config(args.config)
    objdata_tag = objdata_tag_cfg(cfg, args.config)
    graspdata_tag = graspdata_tag_cfg(cfg, args.config)
    ds = DatasetObjects(
        scales=data_run_scales_cfg(cfg),
        objdata_tag=objdata_tag,
        include_native=data_use_native_asset_cfg(cfg),
        graspdata_tag=graspdata_tag,
        generated_dataset_root=data_generated_dataset_root_cfg(cfg),
        verbose=data_verbose_cfg(cfg),
    )
    dataset_root = Path(data_generated_dataset_root_cfg(cfg)).resolve()
    dataset_dir = dataset_root / graspdata_tag
    write_split_jsons(
        entries=ds.get_entries(),
        dataset_dir=dataset_dir,
        render_subdir=str(cfg["sampling"]["pc_subdir"]),
        split_seed=int(cfg["seed"]),
        grasp_h5_name=str(cfg["data"]["h5_name"]),
        grasp_npy_name=str(cfg["data"]["npy_name"]),
        grasp_fail_h5_name=str(cfg["data"]["fail_h5_name"]),
        grasp_fail_npy_name=str(cfg["data"]["fail_npy_name"]),
    )


if __name__ == "__main__":
    main()
