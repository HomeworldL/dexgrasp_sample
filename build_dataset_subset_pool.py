#!/usr/bin/env python3
"""Build fixed-size in-domain and OOD pool subsets from shape-cluster metadata."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

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
        description=(
            "Build fixed-size train/test/subtest and OOD/subOOD pool subsets from "
            "objdata and shape-cluster outputs."
        )
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--cluster-tag", type=str, default="")
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--subtest-count", type=int, required=True)
    parser.add_argument("--ood-count", type=int, required=True)
    parser.add_argument("--subood-count", type=int, required=True)
    parser.add_argument("--in-domain-cluster-max-id", type=int, default=19)
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
    subtest_count: int,
    ood_count: int,
    subood_count: int,
    in_domain_cluster_max_id: int,
    seed: int,
) -> str:
    return f"{cluster_tag}_{filter_version}_pool_seed{int(seed)}"


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


def build_pool_object_record(
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


def split_cluster_ids(
    all_cluster_ids: Sequence[int], in_domain_cluster_max_id: int
) -> tuple[List[int], List[int]]:
    in_domain_ids = [
        cluster_id
        for cluster_id in sorted(all_cluster_ids)
        if cluster_id <= in_domain_cluster_max_id
    ]
    ood_ids = [
        cluster_id
        for cluster_id in sorted(all_cluster_ids)
        if cluster_id > in_domain_cluster_max_id
    ]
    if not in_domain_ids:
        raise ValueError("No in-domain clusters selected.")
    if not ood_ids:
        raise ValueError("No OOD clusters selected.")
    return in_domain_ids, ood_ids


def build_cluster_member_lists(
    ordered_cluster_members: Dict[str, List[str]],
    selected_cluster_ids: Sequence[int],
) -> Dict[str, List[str]]:
    return {
        str(cluster_id): list(ordered_cluster_members[str(cluster_id)])
        for cluster_id in selected_cluster_ids
    }


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


def split_even_odd_cluster_members(
    cluster_members: Dict[str, List[str]],
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    odd_members: Dict[str, List[str]] = {}
    even_members: Dict[str, List[str]] = {}
    for cluster_id, members in cluster_members.items():
        odd_members[cluster_id] = list(members[0::2])
        even_members[cluster_id] = list(members[1::2])
    return odd_members, even_members


def select_balanced_cluster_members(
    cluster_members: Dict[str, List[str]],
    count: int,
    split_name: str,
    seed: int,
) -> Dict[str, List[str]]:
    if count < 0:
        raise ValueError(f"{split_name}_count must be >= 0.")

    cluster_ids = sorted(cluster_members.keys(), key=int)
    if not cluster_ids:
        raise ValueError(f"No clusters are available for {split_name}.")

    total_candidates = sum(
        len(cluster_members[cluster_id]) for cluster_id in cluster_ids
    )
    if count > total_candidates:
        raise ValueError(
            f"Requested {split_name}_count={count} but only {total_candidates} "
            f"candidates are available in the deterministic {split_name} pool."
        )

    base_quota, remainder = divmod(count, len(cluster_ids))
    quota_order = list(cluster_ids)
    random.Random(int(seed)).shuffle(quota_order)
    quotas = {cluster_id: base_quota for cluster_id in cluster_ids}
    for cluster_id in quota_order[:remainder]:
        quotas[cluster_id] += 1

    selected: Dict[str, List[str]] = {}
    remaining: Dict[str, List[str]] = {}
    for cluster_id in cluster_ids:
        members = list(cluster_members[cluster_id])
        take_count = min(quotas[cluster_id], len(members))
        selected[cluster_id] = members[:take_count]
        remaining[cluster_id] = members[take_count:]

    deficit = count - sum(len(members) for members in selected.values())
    refill_order = list(cluster_ids)
    random.Random(int(seed) + 1).shuffle(refill_order)
    while deficit > 0:
        made_progress = False
        for cluster_id in refill_order:
            if not remaining[cluster_id]:
                continue
            selected[cluster_id].append(remaining[cluster_id].pop(0))
            deficit -= 1
            made_progress = True
            if deficit == 0:
                break
        if not made_progress:
            raise RuntimeError(f"Failed to refill balanced {split_name} selection.")

    return selected


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


def shuffle_cluster_members(
    cluster_members: Dict[str, List[str]],
    seed: int,
) -> Dict[str, List[str]]:
    shuffled: Dict[str, List[str]] = {}
    for cluster_id in sorted(cluster_members.keys(), key=int):
        members = list(cluster_members[cluster_id])
        rng = random.Random(int(seed) * 1000 + int(cluster_id))
        rng.shuffle(members)
        shuffled[cluster_id] = members
    return shuffled


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


def collect_pool_records(
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
        records[object_name] = build_pool_object_record(
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


def write_json(path: Path, payload: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_subset_outputs(
    *,
    output_dir: Path,
    subset_tag: str,
    cluster_tag: str,
    cluster_scale_tag: str,
    cluster_dir: Path,
    objdata_root: Path,
    cluster_cfg: Dict,
    pool_filter_cfg: Dict,
    selection_seed: int,
    in_domain_cluster_max_id: int,
    train_records: Dict[str, Dict],
    test_records: Dict[str, Dict],
    subtest_records: Dict[str, Dict],
    ood_records: Dict[str, Dict],
    subood_records: Dict[str, Dict],
    train_cluster_index: Dict,
    test_cluster_index: Dict,
    subtest_cluster_index: Dict,
    ood_cluster_index: Dict,
    subood_cluster_index: Dict,
    all_records: Dict[str, Dict],
) -> None:
    write_json(output_dir / "train_object.json", {"objects": train_records})
    write_json(output_dir / "test_object.json", {"objects": test_records})
    write_json(output_dir / "subtest_object.json", {"objects": subtest_records})
    write_json(output_dir / "ood_object.json", {"objects": ood_records})
    write_json(output_dir / "subood_object.json", {"objects": subood_records})

    write_json(output_dir / "train_cluster.json", train_cluster_index)
    write_json(output_dir / "test_cluster.json", test_cluster_index)
    write_json(output_dir / "subtest_cluster.json", subtest_cluster_index)
    write_json(output_dir / "ood_cluster.json", ood_cluster_index)
    write_json(output_dir / "subood_cluster.json", subood_cluster_index)

    write_json(
        output_dir / "meta.json",
        {
            "subset_tag": subset_tag,
            "cluster_tag": cluster_tag,
            "cluster_scale_tag": cluster_scale_tag,
            "shape_cluster_dir": relpath_str(cluster_dir, objdata_root),
            "cluster_version": cluster_cfg["version"],
            "pool_filter": pool_filter_cfg,
            "selection_seed": int(selection_seed),
            "in_domain_cluster_max_id": int(in_domain_cluster_max_id),
            "ood_cluster_min_id": int(in_domain_cluster_max_id) + 1,
            "num_objects": len(all_records),
            "num_train_objects": len(train_records),
            "num_test_objects": len(test_records),
            "num_subtest_objects": len(subtest_records),
            "num_ood_objects": len(ood_records),
            "num_subood_objects": len(subood_records),
            "include_native": False,
            "selection_rule": {
                "train": (
                    "Use odd-indexed in-domain candidates from each cluster after sorting "
                    "by ascending cluster_rank, ascending cluster_distance, then object_name. "
                    "Select a capacity-aware balanced quota per cluster, refill deficits "
                    "from remaining clusters with the selection seed, then interleave "
                    "clusters in ascending cluster id order."
                ),
                "test": (
                    "Use even-indexed in-domain candidates from each cluster after the same "
                    "center ordering used by train. Shuffle each cluster deterministically, "
                    "select a capacity-aware balanced quota per cluster, refill deficits "
                    "from remaining clusters with the selection seed, then interleave "
                    "clusters in ascending cluster id order."
                ),
                "subtest": "Take the fixed-count prefix of the deterministic test ordering.",
                "ood": (
                    "Use OOD clusters only. Shuffle each cluster deterministically, select "
                    "a capacity-aware balanced quota per cluster, refill deficits from "
                    "remaining OOD clusters with the selection seed, then interleave "
                    "clusters in ascending cluster id order."
                ),
                "subood": "Take the fixed-count prefix of the deterministic OOD ordering.",
            },
            "files": {
                "train_object": "train_object.json",
                "test_object": "test_object.json",
                "subtest_object": "subtest_object.json",
                "ood_object": "ood_object.json",
                "subood_object": "subood_object.json",
                "train_cluster": "train_cluster.json",
                "test_cluster": "test_cluster.json",
                "subtest_cluster": "subtest_cluster.json",
                "ood_cluster": "ood_cluster.json",
                "subood_cluster": "subood_cluster.json",
            },
        },
    )


def main() -> None:
    args = parse_args()
    cfg = load_asset_config(args.config)

    selection_seed = int(cfg["seed"])
    cluster_tag = resolve_cluster_tag(cfg, args.cluster_tag)
    cluster_cfg = resolve_shape_cluster_cfg(cfg)
    pool_filter_cfg = resolve_pool_filter_cfg(cfg)

    if int(args.subtest_count) > int(args.test_count):
        raise ValueError("subtest_count must be <= test_count.")
    if int(args.subood_count) > int(args.ood_count):
        raise ValueError("subood_count must be <= ood_count.")

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

    records = collect_pool_records(
        ds=ds,
        objdata_root=objdata_root,
        cluster_scale_tag=cluster_scale_tag,
        object_cluster_map=object_cluster_map,
        cluster_dir=cluster_dir,
    )
    ordered_cluster_members = order_records_within_cluster(records)
    all_cluster_ids = sorted(int(cluster_id) for cluster_id in ordered_cluster_members)
    in_domain_cluster_ids, ood_cluster_ids = split_cluster_ids(
        all_cluster_ids=all_cluster_ids,
        in_domain_cluster_max_id=int(args.in_domain_cluster_max_id),
    )

    in_domain_members = build_cluster_member_lists(
        ordered_cluster_members, in_domain_cluster_ids
    )
    train_candidates, test_candidates = split_even_odd_cluster_members(
        in_domain_members
    )
    train_selected_members = select_balanced_cluster_members(
        train_candidates,
        count=int(args.train_count),
        split_name="train",
        seed=selection_seed + 11,
    )
    ordered_train_names = interleave_cluster_members(train_selected_members)
    train_records = select_prefix_records(
        ordered_names=ordered_train_names,
        records=records,
        count=int(args.train_count),
        split_name="train",
    )

    test_shuffled_members = shuffle_cluster_members(
        test_candidates, seed=selection_seed + 101
    )
    test_selected_members = select_balanced_cluster_members(
        test_shuffled_members,
        count=int(args.test_count),
        split_name="test",
        seed=selection_seed + 111,
    )
    ordered_test_names = interleave_cluster_members(test_selected_members)
    test_records = select_prefix_records(
        ordered_names=ordered_test_names,
        records=records,
        count=int(args.test_count),
        split_name="test",
    )
    subtest_records = select_prefix_records(
        ordered_names=ordered_test_names,
        records=records,
        count=int(args.subtest_count),
        split_name="subtest",
    )

    ood_members = build_cluster_member_lists(ordered_cluster_members, ood_cluster_ids)
    ood_shuffled_members = shuffle_cluster_members(
        ood_members, seed=selection_seed + 202
    )
    ood_selected_members = select_balanced_cluster_members(
        ood_shuffled_members,
        count=int(args.ood_count),
        split_name="ood",
        seed=selection_seed + 222,
    )
    ordered_ood_names = interleave_cluster_members(ood_selected_members)
    ood_records = select_prefix_records(
        ordered_names=ordered_ood_names,
        records=records,
        count=int(args.ood_count),
        split_name="ood",
    )
    subood_records = select_prefix_records(
        ordered_names=ordered_ood_names,
        records=records,
        count=int(args.subood_count),
        split_name="subood",
    )

    train_cluster_index = build_split_cluster_index(
        train_records, cluster_payload["clusters"]
    )
    test_cluster_index = build_split_cluster_index(
        test_records, cluster_payload["clusters"]
    )
    subtest_cluster_index = build_split_cluster_index(
        subtest_records, cluster_payload["clusters"]
    )
    ood_cluster_index = build_split_cluster_index(
        ood_records, cluster_payload["clusters"]
    )
    subood_cluster_index = build_split_cluster_index(
        subood_records, cluster_payload["clusters"]
    )

    subset_tag = build_pool_subset_tag(
        cluster_tag=cluster_tag,
        filter_version=str(pool_filter_cfg["version"]),
        train_count=int(args.train_count),
        test_count=int(args.test_count),
        subtest_count=int(args.subtest_count),
        ood_count=int(args.ood_count),
        subood_count=int(args.subood_count),
        in_domain_cluster_max_id=int(args.in_domain_cluster_max_id),
        seed=selection_seed,
    )
    output_dir = objdata_root / "_meta" / "pool_subset" / subset_tag
    prepare_output_dir(output_dir, args.force)
    write_subset_outputs(
        output_dir=output_dir,
        subset_tag=subset_tag,
        cluster_tag=cluster_tag,
        cluster_scale_tag=cluster_scale_tag,
        cluster_dir=cluster_dir,
        objdata_root=objdata_root,
        cluster_cfg=cluster_cfg,
        pool_filter_cfg=pool_filter_cfg,
        selection_seed=selection_seed,
        in_domain_cluster_max_id=int(args.in_domain_cluster_max_id),
        train_records=train_records,
        test_records=test_records,
        subtest_records=subtest_records,
        ood_records=ood_records,
        subood_records=subood_records,
        train_cluster_index=train_cluster_index,
        test_cluster_index=test_cluster_index,
        subtest_cluster_index=subtest_cluster_index,
        ood_cluster_index=ood_cluster_index,
        subood_cluster_index=subood_cluster_index,
        all_records=records,
    )

    print(
        f"[build_dataset_subset_pool] objdata_tag={objdata_tag} cluster_tag={cluster_tag} "
        f"in_domain_clusters={len(in_domain_cluster_ids)} ood_clusters={len(ood_cluster_ids)} "
        f"train={len(train_records)} test={len(test_records)} subtest={len(subtest_records)} "
        f"ood={len(ood_records)} subood={len(subood_records)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
