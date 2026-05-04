#!/usr/bin/env python3
"""Build RL-oriented train/test split JSON files from objdata and shape-cluster metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.dataset_objects import DatasetObjects
from src.shape_cluster import build_cluster_tag
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    asset_scales_from_config,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    load_asset_config,
    objdata_tag_from_config,
    raw_dataset_name_from_config,
    raw_dataset_root_from_config,
    relpath_str,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RL train/test split JSON files from objdata and shape-cluster outputs."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--cluster-tag", type=str, default="")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_shape_cluster_cfg(cfg: Dict) -> Dict:
    shape_cfg = cfg.get("shape_cluster", {})
    if shape_cfg is None:
        shape_cfg = {}
    if not isinstance(shape_cfg, dict):
        raise ValueError("Config field shape_cluster must be an object when provided.")
    return {
        "scale_tag": str(shape_cfg.get("scale_tag", "scale120")),
        "feature_dim": int(shape_cfg.get("feature_dim", 128)),
        "kmeans_k": int(shape_cfg.get("kmeans_k", 24)),
        "version": str(shape_cfg.get("version", "v1")),
    }


def resolve_cluster_tag(cfg: Dict, cluster_tag_override: str) -> str:
    if cluster_tag_override:
        return cluster_tag_override
    shape_cfg = resolve_shape_cluster_cfg(cfg)
    return build_cluster_tag(
        version=shape_cfg["version"],
        feature_dim=shape_cfg["feature_dim"],
        k=shape_cfg["kmeans_k"],
        seed=int(cfg["seed"]),
    )


def build_rl_split_tag(cluster_tag: str, split_seed: int, test_ratio: float) -> str:
    test_percent = int(round(float(test_ratio) * 100.0))
    return f"{cluster_tag}_test{test_percent:02d}_seed{int(split_seed)}"


def load_object_cluster_payload(cluster_dir: Path) -> Dict:
    object_cluster_path = cluster_dir / "object_cluster.json"
    cluster_index_path = cluster_dir / "cluster_index.json"
    if not object_cluster_path.exists():
        raise FileNotFoundError(f"Missing shape-cluster file: {object_cluster_path}")
    if not cluster_index_path.exists():
        raise FileNotFoundError(f"Missing shape-cluster file: {cluster_index_path}")

    object_cluster_payload = json.loads(object_cluster_path.read_text(encoding="utf-8"))
    cluster_index_payload = json.loads(cluster_index_path.read_text(encoding="utf-8"))
    if not isinstance(object_cluster_payload, dict) or "objects" not in object_cluster_payload:
        raise ValueError(f"Invalid object_cluster.json payload: {object_cluster_path}")
    if not isinstance(cluster_index_payload, dict) or "clusters" not in cluster_index_payload:
        raise ValueError(f"Invalid cluster_index.json payload: {cluster_index_path}")
    return {
        "cluster_tag": str(object_cluster_payload.get("cluster_tag", "")),
        "scale_tag": str(object_cluster_payload.get("scale_tag", "")),
        "objects": dict(object_cluster_payload["objects"]),
        "clusters": dict(cluster_index_payload["clusters"]),
    }


def build_rl_record(
    entry: Dict,
    objdata_root: Path,
    cluster_tag: str,
    cluster_scale_tag: str,
    object_cluster_info: Dict,
) -> Dict:
    asset_dir = Path(str(entry["asset_dir_abs"])).resolve()
    coacd_path = Path(str(entry["coacd_abs"])).resolve()
    mjcf_path = Path(str(entry["mjcf_abs"])).resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Missing asset_dir_abs: {asset_dir}")
    if not coacd_path.exists():
        raise FileNotFoundError(f"Missing coacd_abs: {coacd_path}")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing mjcf_abs: {mjcf_path}")

    object_name = str(entry["object_name"])
    return {
        "global_id": int(entry["global_id"]),
        "object_scale_key": str(entry["object_scale_key"]),
        "object_name": object_name,
        "asset_path": relpath_str(asset_dir, objdata_root),
        "coacd_path": relpath_str(coacd_path, objdata_root),
        "mjcf_path": relpath_str(mjcf_path, objdata_root),
        "scale_tag": str(entry.get("scale_tag", "")),
        "scale": None if entry.get("scale") is None else float(entry["scale"]),
        "is_native": bool(entry.get("is_native", False)),
        "cluster_tag": cluster_tag,
        "cluster_scale_tag": cluster_scale_tag,
        "cluster_id": int(object_cluster_info["cluster_id"]),
        "cluster_distance": float(object_cluster_info["distance_to_center"]),
        "cluster_rank": int(object_cluster_info["rank_in_cluster"]),
        "cluster_feature_index": int(object_cluster_info["feature_index"]),
        "cluster_center_object_name": str(object_cluster_info["center_object_name"]),
        "cluster_center_object_scale_key": str(object_cluster_info["center_object_scale_key"]),
    }


def split_records_by_object(
    records: Sequence[Dict],
    seed: int,
    test_ratio: float,
) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 <= float(test_ratio) < 1.0:
        raise ValueError("test_ratio must satisfy 0 <= test_ratio < 1.")

    object_names = sorted({str(record["object_name"]) for record in records})
    random.Random(int(seed)).shuffle(object_names)
    if len(object_names) <= 1 or test_ratio <= 0.0:
        test_objects = set()
    else:
        test_count = max(1, int(round(len(object_names) * float(test_ratio))))
        test_count = min(test_count, len(object_names) - 1)
        test_objects = set(object_names[:test_count])

    train_records = [
        record for record in records if str(record["object_name"]) not in test_objects
    ]
    test_records = [
        record for record in records if str(record["object_name"]) in test_objects
    ]
    return train_records, test_records


def build_split_cluster_index(records: Sequence[Dict]) -> Dict:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(str(record["cluster_id"]), []).append(record)

    clusters: Dict[str, Dict] = {}
    for cluster_id in sorted(grouped.keys(), key=int):
        unique_objects: Dict[str, Dict] = {}
        for record in grouped[cluster_id]:
            object_name = str(record["object_name"])
            candidate = {
                "object_name": object_name,
                "object_scale_key": f"{object_name}__{record['cluster_scale_tag']}",
                "distance_to_center": float(record["cluster_distance"]),
                "rank_in_cluster": int(record["cluster_rank"]),
            }
            current = unique_objects.get(object_name)
            if current is None or candidate["rank_in_cluster"] < current["rank_in_cluster"]:
                unique_objects[object_name] = candidate

        members = sorted(
            unique_objects.values(),
            key=lambda item: (float(item["distance_to_center"]), int(item["rank_in_cluster"])),
        )
        split_center = members[0] if members else None
        clusters[cluster_id] = {
            "cluster_id": int(cluster_id),
            "member_count": len(members),
            "global_center_object_name": str(grouped[cluster_id][0]["cluster_center_object_name"]),
            "global_center_object_scale_key": str(
                grouped[cluster_id][0]["cluster_center_object_scale_key"]
            ),
            "split_center_object_name": None if split_center is None else split_center["object_name"],
            "split_center_object_scale_key": None
            if split_center is None
            else split_center["object_scale_key"],
            "members": members,
        }
    return {"clusters": clusters}


def write_json(path: Path, payload: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_asset_config(args.config)

    split_seed = int(cfg["seed"]) if args.split_seed is None else int(args.split_seed)
    cluster_tag = resolve_cluster_tag(cfg, args.cluster_tag)
    cluster_cfg = resolve_shape_cluster_cfg(cfg)

    generated_root = Path(generated_dataset_root_from_config(cfg)).resolve()
    objdata_tag = objdata_tag_from_config(cfg, args.config)
    objdata_root = generated_root / objdata_tag
    if not objdata_root.is_dir():
        raise FileNotFoundError(f"objdata root not found: {objdata_root}")

    cluster_dir = objdata_root / "_meta" / "shape_cluster" / cluster_tag
    cluster_payload = load_object_cluster_payload(cluster_dir)
    object_cluster_map = dict(cluster_payload["objects"])
    cluster_scale_tag = str(cluster_payload["scale_tag"])

    ds = DatasetObjects(
        raw_dataset_root=raw_dataset_root_from_config(cfg),
        raw_dataset_name=raw_dataset_name_from_config(cfg),
        scales=asset_scales_from_config(cfg),
        objdata_tag=objdata_tag,
        include_native=False,
        graspdata_tag=objdata_tag,
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        verbose=data_verbose_from_config(cfg),
    )

    records: List[Dict] = []
    for entry in sorted(ds.get_entries(), key=lambda item: int(item["global_id"])):
        object_name = str(entry["object_name"])
        if object_name not in object_cluster_map:
            raise KeyError(
                f"Object '{object_name}' not found in shape-cluster payload under {cluster_dir}"
            )
        records.append(
            build_rl_record(
                entry=entry,
                objdata_root=objdata_root,
                cluster_tag=cluster_tag,
                cluster_scale_tag=cluster_scale_tag,
                object_cluster_info=object_cluster_map[object_name],
            )
        )

    train_records, test_records = split_records_by_object(
        records=records,
        seed=split_seed,
        test_ratio=float(args.test_ratio),
    )
    train_cluster_index = build_split_cluster_index(train_records)
    test_cluster_index = build_split_cluster_index(test_records)

    split_tag = build_rl_split_tag(
        cluster_tag=cluster_tag,
        split_seed=split_seed,
        test_ratio=float(args.test_ratio),
    )
    output_dir = objdata_root / "_meta" / "rl_split" / split_tag
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_dir}. Use --force to overwrite.")
    if output_dir.exists() and args.force:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()

    write_json(output_dir / "train_rl.json", train_records)
    write_json(output_dir / "test_rl.json", test_records)
    write_json(output_dir / "train_cluster_index.json", train_cluster_index)
    write_json(output_dir / "test_cluster_index.json", test_cluster_index)
    write_json(
        output_dir / "meta.json",
        {
            "split_tag": split_tag,
            "cluster_tag": cluster_tag,
            "cluster_scale_tag": cluster_scale_tag,
            "shape_cluster_dir": relpath_str(cluster_dir, objdata_root),
            "cluster_version": cluster_cfg["version"],
            "split_seed": split_seed,
            "test_ratio": float(args.test_ratio),
            "num_records": len(records),
            "num_train_records": len(train_records),
            "num_test_records": len(test_records),
            "num_train_objects": len({record["object_name"] for record in train_records}),
            "num_test_objects": len({record["object_name"] for record in test_records}),
            "include_native": False,
            "files": {
                "train_rl": "train_rl.json",
                "test_rl": "test_rl.json",
                "train_cluster_index": "train_cluster_index.json",
                "test_cluster_index": "test_cluster_index.json",
            },
        },
    )

    print(
        f"[build_dataset_splits_rl] objdata_tag={objdata_tag} cluster_tag={cluster_tag} "
        f"records={len(records)} train={len(train_records)} test={len(test_records)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
