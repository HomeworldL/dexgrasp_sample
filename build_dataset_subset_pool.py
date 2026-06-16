#!/usr/bin/env python3
"""Build fixed-size pool subsets from objdata and shape-cluster metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.dataset_objects import DatasetObjects
from src.shape_cluster import build_cluster_tag
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    data_asset_scales_cfg,
    data_generated_dataset_root_cfg,
    data_verbose_cfg,
    load_asset_config,
    objdata_tag_cfg,
    relpath_str,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-size train/test pool subsets from objdata and shape-cluster outputs."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--cluster-tag", type=str, default="")
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
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


def resolve_pool_filter_cfg(cfg: Dict) -> Dict:
    pool_cfg = cfg.get("pool_filter", {})
    if pool_cfg is None:
        pool_cfg = {}
    if not isinstance(pool_cfg, dict):
        raise ValueError("Config field pool_filter must be an object when provided.")
    return {
        "enabled": bool(pool_cfg.get("enabled", True)),
        "version": str(pool_cfg.get("version", "v1")).strip() or "v1",
        "mesh_kind": str(pool_cfg.get("mesh_kind", "coacd")).strip().lower(),
        "thin_ratio_min": float(pool_cfg.get("thin_ratio_min", 0.12)),
        "flat_ratio_min": float(pool_cfg.get("flat_ratio_min", 0.12)),
    }


def build_pool_subset_tag(
    cluster_tag: str,
    filter_version: str,
    train_count: int,
    test_count: int,
    seed: int,
) -> str:
    return (
        f"{cluster_tag}_{filter_version}_train{int(train_count)}"
        f"_test{int(test_count)}_seed{int(seed)}"
    )


def load_object_cluster_payload(cluster_dir: Path) -> Dict:
    object_labels_path = cluster_dir / "object_labels.json"
    cluster_labels_path = cluster_dir / "cluster_labels.json"
    if not object_labels_path.exists():
        raise FileNotFoundError(f"Missing shape-cluster file: {object_labels_path}")
    if not cluster_labels_path.exists():
        raise FileNotFoundError(f"Missing shape-cluster file: {cluster_labels_path}")

    object_labels_payload = json.loads(object_labels_path.read_text(encoding="utf-8"))
    cluster_labels_payload = json.loads(cluster_labels_path.read_text(encoding="utf-8"))
    if (
        not isinstance(object_labels_payload, dict)
        or "objects" not in object_labels_payload
    ):
        raise ValueError(f"Invalid object_labels.json payload: {object_labels_path}")
    if (
        not isinstance(cluster_labels_payload, dict)
        or "clusters" not in cluster_labels_payload
    ):
        raise ValueError(f"Invalid cluster_labels.json payload: {cluster_labels_path}")
    return {
        "cluster_tag": str(object_labels_payload.get("cluster_tag", "")),
        "scale_tag": str(object_labels_payload.get("scale_tag", "")),
        "objects": dict(object_labels_payload["objects"]),
        "clusters": dict(cluster_labels_payload["clusters"]),
    }


def build_scale_record(
    entry: Dict,
    objdata_root: Path,
) -> Dict:
    asset_dir = Path(str(entry["asset_dir_abs"])).resolve()
    mjcf_path = Path(str(entry["mjcf_abs"])).resolve()
    usd_path = asset_dir / "object.usd"
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Missing asset_dir_abs: {asset_dir}")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing mjcf_abs: {mjcf_path}")

    return {
        "object_scale_key": str(entry["object_scale_key"]),
        "asset_path": relpath_str(asset_dir, objdata_root),
        "mjcf_path": relpath_str(mjcf_path, objdata_root),
        "usd_path": relpath_str(usd_path, objdata_root) if usd_path.exists() else None,
        "scale_tag": str(entry.get("scale_tag", "")),
        "scale": None if entry.get("scale") is None else float(entry["scale"]),
    }


def build_rl_object_record(
    *,
    object_name: str,
    cluster_scale_tag: str,
    object_cluster_info: Dict,
    scale_records: Sequence[Dict],
) -> Dict:
    ordered_scales = sorted(
        scale_records,
        key=lambda item: (
            str(item["scale_tag"]),
            "" if item["scale"] is None else f"{float(item['scale']):.8f}",
            str(item["object_scale_key"]),
        ),
    )
    return {
        "object_name": object_name,
        "cluster_scale_tag": cluster_scale_tag,
        "cluster_id": int(object_cluster_info["cluster_id"]),
        "cluster_rank": int(object_cluster_info["rank_in_cluster"]),
        "cluster_distance": float(object_cluster_info["distance_to_center"]),
        "distance_to_global_center": float(
            object_cluster_info["distance_to_global_center"]
        ),
        "num_scales": len(ordered_scales),
        "scale_tags": [str(item["scale_tag"]) for item in ordered_scales],
        "scales": list(ordered_scales),
    }


def order_records_within_cluster(records: Dict[str, Dict]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records.values():
        grouped.setdefault(str(record["cluster_id"]), []).append(record)

    ordered: Dict[str, List[str]] = {}
    for cluster_id in sorted(grouped.keys(), key=int):
        members = sorted(
            grouped[cluster_id],
            key=lambda record: (
                int(record["cluster_rank"]),
                float(record["cluster_distance"]),
                str(record["object_name"]),
            ),
        )
        ordered[cluster_id] = [str(member["object_name"]) for member in members]
    return ordered


def split_cluster_members_for_pool(
    ordered_cluster_members: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    train_members: Dict[str, List[str]] = {}
    test_members: Dict[str, List[str]] = {}
    for cluster_id, members in ordered_cluster_members.items():
        train_members[cluster_id] = list(members[0::2])
        test_members[cluster_id] = list(members[1::2])
    return train_members, test_members


def interleave_cluster_members(cluster_members: Dict[str, List[str]]) -> List[str]:
    ordered_cluster_ids = sorted(cluster_members.keys(), key=int)
    max_len = max(
        (len(cluster_members[cluster_id]) for cluster_id in ordered_cluster_ids),
        default=0,
    )
    ordered_names: List[str] = []
    for index in range(max_len):
        for cluster_id in ordered_cluster_ids:
            members = cluster_members[cluster_id]
            if index < len(members):
                ordered_names.append(members[index])
    return ordered_names


def select_prefix_records(
    ordered_names: Sequence[str],
    records: Dict[str, Dict],
    count: int,
    split_name: str,
) -> Dict[str, Dict]:
    if count < 0:
        raise ValueError(f"{split_name}_count must be >= 0.")
    if count > len(ordered_names):
        raise ValueError(
            f"Requested {split_name}_count={count} but only {len(ordered_names)} "
            f"candidates are available in the deterministic {split_name} pool."
        )
    return {object_name: records[object_name] for object_name in ordered_names[:count]}


def build_split_cluster_index(
    records: Dict[str, Dict], cluster_labels_map: Dict[str, Dict]
) -> Dict:
    grouped: Dict[str, List[Dict]] = {}
    for record in records.values():
        grouped.setdefault(str(record["cluster_id"]), []).append(record)

    clusters: Dict[str, Dict] = {}
    for cluster_id in sorted(grouped.keys(), key=int):
        members = sorted(
            grouped[cluster_id],
            key=lambda item: (
                int(item["cluster_rank"]),
                float(item["cluster_distance"]),
                str(item["object_name"]),
            ),
        )
        split_center = members[0] if members else None
        clusters[cluster_id] = {
            "cluster_id": int(cluster_id),
            "member_count": len(members),
            "total_scale_count": sum(int(member["num_scales"]) for member in members),
            "distance_to_global_center": float(
                cluster_labels_map[cluster_id]["distance_to_global_center"]
            ),
            "split_center_object_name": (
                None if split_center is None else split_center["object_name"]
            ),
            "members": list(members),
        }
    return {"clusters": clusters}


def collect_rl_records(
    *,
    ds: DatasetObjects,
    objdata_root: Path,
    cluster_scale_tag: str,
    object_cluster_map: Dict[str, Dict],
    cluster_dir: Path,
) -> Dict[str, Dict]:
    grouped_scale_records: Dict[str, List[Dict]] = {}
    for entry in sorted(ds.get_entries(), key=lambda item: int(item["global_id"])):
        object_name = str(entry["object_name"])
        if object_name not in object_cluster_map:
            raise KeyError(
                f"Object '{object_name}' not found in shape-cluster payload under {cluster_dir}"
            )
        grouped_scale_records.setdefault(object_name, []).append(
            build_scale_record(entry=entry, objdata_root=objdata_root)
        )

    records: Dict[str, Dict] = {}
    for object_name in sorted(grouped_scale_records.keys()):
        records[object_name] = build_rl_object_record(
            object_name=object_name,
            cluster_scale_tag=cluster_scale_tag,
            object_cluster_info=object_cluster_map[object_name],
            scale_records=grouped_scale_records[object_name],
        )
    return records


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --force to overwrite."
        )
    if output_dir.exists() and force:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()


def write_split_outputs(
    *,
    output_dir: Path,
    subset_tag: str,
    cluster_tag: str,
    cluster_scale_tag: str,
    cluster_dir: Path,
    objdata_root: Path,
    cluster_cfg: Dict,
    pool_filter_cfg: Dict,
    train_count: int,
    test_count: int,
    records: Dict[str, Dict],
    train_records: Dict[str, Dict],
    test_records: Dict[str, Dict],
    train_cluster_index: Dict,
    test_cluster_index: Dict,
) -> None:
    write_json(output_dir / "train_object.json", {"objects": train_records})
    write_json(output_dir / "test_object.json", {"objects": test_records})
    write_json(output_dir / "train_cluster.json", train_cluster_index)
    write_json(output_dir / "test_cluster.json", test_cluster_index)
    write_json(
        output_dir / "meta.json",
        {
            "subset_tag": subset_tag,
            "cluster_tag": cluster_tag,
            "cluster_scale_tag": cluster_scale_tag,
            "shape_cluster_dir": relpath_str(cluster_dir, objdata_root),
            "cluster_version": cluster_cfg["version"],
            "pool_filter": pool_filter_cfg,
            "train_count": int(train_count),
            "test_count": int(test_count),
            "num_objects": len(records),
            "num_train_objects": len(train_records),
            "num_test_objects": len(test_records),
            "num_scales": sum(int(record["num_scales"]) for record in records.values()),
            "num_train_scales": sum(
                int(record["num_scales"]) for record in train_records.values()
            ),
            "num_test_scales": sum(
                int(record["num_scales"]) for record in test_records.values()
            ),
            "include_native": False,
            "selection_rule": (
                "Within each cluster, objects are sorted by ascending cluster_rank, "
                "ascending cluster_distance, then object_name. Even-ranked objects feed "
                "the train candidate pool, odd-ranked objects feed the test candidate pool, "
                "and each pool is interleaved "
                "cluster-by-cluster in ascending cluster id order before fixed-count prefix truncation."
            ),
            "files": {
                "train_object": "train_object.json",
                "test_object": "test_object.json",
                "train_cluster": "train_cluster.json",
                "test_cluster": "test_cluster.json",
            },
        },
    )


def write_json(path: Path, payload: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    cfg = load_asset_config(args.config)

    selection_seed = int(cfg["seed"])
    cluster_tag = resolve_cluster_tag(cfg, args.cluster_tag)
    cluster_cfg = resolve_shape_cluster_cfg(cfg)
    pool_filter_cfg = resolve_pool_filter_cfg(cfg)

    generated_root = Path(data_generated_dataset_root_cfg(cfg)).resolve()
    objdata_tag = objdata_tag_cfg(cfg, args.config)
    objdata_root = generated_root / objdata_tag
    if not objdata_root.is_dir():
        raise FileNotFoundError(f"objdata root not found: {objdata_root}")

    cluster_dir = objdata_root / "_meta" / "shape_cluster" / cluster_tag
    cluster_payload = load_object_cluster_payload(cluster_dir)
    object_cluster_map = dict(cluster_payload["objects"])
    cluster_scale_tag = str(cluster_payload["scale_tag"])

    ds = DatasetObjects(
        scales=data_asset_scales_cfg(cfg),
        objdata_tag=objdata_tag,
        include_native=False,
        graspdata_tag=objdata_tag,
        generated_dataset_root=data_generated_dataset_root_cfg(cfg),
        verbose=data_verbose_cfg(cfg),
    )

    records = collect_rl_records(
        ds=ds,
        objdata_root=objdata_root,
        cluster_scale_tag=cluster_scale_tag,
        object_cluster_map=object_cluster_map,
        cluster_dir=cluster_dir,
    )
    ordered_cluster_members = order_records_within_cluster(records)
    train_cluster_members, test_cluster_members = split_cluster_members_for_pool(
        ordered_cluster_members
    )
    ordered_train_names = interleave_cluster_members(train_cluster_members)
    ordered_test_names = interleave_cluster_members(test_cluster_members)

    train_records = select_prefix_records(
        ordered_names=ordered_train_names,
        records=records,
        count=int(args.train_count),
        split_name="train",
    )
    test_records = select_prefix_records(
        ordered_names=ordered_test_names,
        records=records,
        count=int(args.test_count),
        split_name="test",
    )
    train_cluster_index = build_split_cluster_index(
        train_records, cluster_payload["clusters"]
    )
    test_cluster_index = build_split_cluster_index(
        test_records, cluster_payload["clusters"]
    )

    subset_tag = build_pool_subset_tag(
        cluster_tag=cluster_tag,
        filter_version=str(pool_filter_cfg["version"]),
        train_count=int(args.train_count),
        test_count=int(args.test_count),
        seed=selection_seed,
    )
    output_dir = objdata_root / "_meta" / "pool_subset" / subset_tag
    prepare_output_dir(output_dir, args.force)
    write_split_outputs(
        output_dir=output_dir,
        subset_tag=subset_tag,
        cluster_tag=cluster_tag,
        cluster_scale_tag=cluster_scale_tag,
        cluster_dir=cluster_dir,
        objdata_root=objdata_root,
        cluster_cfg=cluster_cfg,
        pool_filter_cfg=pool_filter_cfg,
        train_count=int(args.train_count),
        test_count=int(args.test_count),
        records=records,
        train_records=train_records,
        test_records=test_records,
        train_cluster_index=train_cluster_index,
        test_cluster_index=test_cluster_index,
    )

    print(
        f"[build_dataset_subset_pool] objdata_tag={objdata_tag} cluster_tag={cluster_tag} "
        f"records={len(records)} train={len(train_records)} test={len(test_records)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
