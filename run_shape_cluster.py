#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

from src.shape_cluster import (
    build_cluster_tag,
    extract_embeddings,
    load_object_point_clouds,
    reorder_clusters_by_global_center_distance,
    run_kmeans,
    save_cluster_artifacts,
    standardize_features,
    train_autoencoder,
)
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    generated_dataset_root_from_config,
    load_asset_config,
    objdata_tag_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shape autoencoder and cluster objdata objects.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--scale-tag", type=str, default=None)
    parser.add_argument("--fps-points", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--kmeans-k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--kmeans-n-init", type=int, default=None)
    parser.add_argument("--kmeans-max-iter", type=int, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_shape_cluster_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    shape_cfg = cfg.get("shape_cluster", {})
    if shape_cfg is None:
        shape_cfg = {}
    if not isinstance(shape_cfg, dict):
        raise ValueError("Config field shape_cluster must be an object when provided.")

    defaults = {
        "scale_tag": "scale120",
        "fps_points": 1024,
        "feature_dim": 128,
        "kmeans_k": 24,
        "epochs": 300,
        "batch_size": 16,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "scheduler_eta_min": 1e-7,
        "kmeans_n_init": 10,
        "kmeans_max_iter": 100,
        "version": "v1",
    }
    resolved = {}
    for key, default_value in defaults.items():
        cli_value = getattr(args, key)
        if cli_value is not None:
            resolved[key] = cli_value
        elif key in shape_cfg:
            resolved[key] = shape_cfg[key]
        else:
            resolved[key] = default_value
    return resolved


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    cfg = load_asset_config(args.config)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generated_root = Path(generated_dataset_root_from_config(cfg)).resolve()
    objdata_tag = objdata_tag_from_config(cfg, args.config)
    objdata_root = generated_root / objdata_tag
    if not objdata_root.is_dir():
        raise FileNotFoundError(f"objdata root not found: {objdata_root}")

    pc_subdir = str(cfg["warp_render"]["output_subdir"])
    shape_cluster_cfg = resolve_shape_cluster_cfg(cfg, args)
    cluster_tag = build_cluster_tag(
        version=str(shape_cluster_cfg["version"]),
        feature_dim=int(shape_cluster_cfg["feature_dim"]),
        k=int(shape_cluster_cfg["kmeans_k"]),
        seed=seed,
    )
    output_dir = objdata_root / "_meta" / "shape_cluster" / cluster_tag
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_dir}. Use --force to overwrite.")
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    device = resolve_device(args.device)

    object_names, object_dirs, point_clouds = load_object_point_clouds(
        objdata_root=objdata_root,
        scale_tag=str(shape_cluster_cfg["scale_tag"]),
        pc_subdir=pc_subdir,
        fps_points=int(shape_cluster_cfg["fps_points"]),
        seed=seed,
    )

    training = train_autoencoder(
        point_clouds=point_clouds,
        feature_dim=int(shape_cluster_cfg["feature_dim"]),
        epochs=int(shape_cluster_cfg["epochs"]),
        batch_size=int(shape_cluster_cfg["batch_size"]),
        lr=float(shape_cluster_cfg["lr"]),
        weight_decay=float(shape_cluster_cfg["weight_decay"]),
        scheduler_eta_min=float(shape_cluster_cfg["scheduler_eta_min"]),
        device=device,
        seed=seed,
    )
    embeddings = extract_embeddings(
        model=training.model,
        point_clouds=point_clouds,
        batch_size=int(shape_cluster_cfg["batch_size"]),
        device=device,
    )
    normalized_embeddings, feature_mean, feature_std = standardize_features(embeddings)
    kmeans = run_kmeans(
        features=normalized_embeddings,
        k=int(shape_cluster_cfg["kmeans_k"]),
        seed=seed,
        n_init=int(shape_cluster_cfg["kmeans_n_init"]),
        max_iter=int(shape_cluster_cfg["kmeans_max_iter"]),
    )
    reordered_labels, reordered_centers, reordered_center_global_distances = (
        reorder_clusters_by_global_center_distance(
            labels=kmeans.labels,
            centers=kmeans.centers,
            features=normalized_embeddings,
        )
    )

    save_cluster_artifacts(
        output_dir=output_dir,
        object_names=object_names,
        labels=reordered_labels,
        normalized_embeddings=normalized_embeddings,
        centers=reordered_centers,
        center_global_distances=reordered_center_global_distances,
        train_history=training.history,
        cluster_tag=cluster_tag,
        scale_tag=str(shape_cluster_cfg["scale_tag"]),
        pc_subdir=pc_subdir,
        fps_points=int(shape_cluster_cfg["fps_points"]),
        feature_dim=int(shape_cluster_cfg["feature_dim"]),
        kmeans_k=int(shape_cluster_cfg["kmeans_k"]),
        seed=seed,
        model_state_dict=training.model.state_dict(),
        extra_meta={
            "epochs": int(shape_cluster_cfg["epochs"]),
            "batch_size": int(shape_cluster_cfg["batch_size"]),
            "lr": float(shape_cluster_cfg["lr"]),
            "weight_decay": float(shape_cluster_cfg["weight_decay"]),
            "scheduler_eta_min": float(shape_cluster_cfg["scheduler_eta_min"]),
            "kmeans_n_init": int(shape_cluster_cfg["kmeans_n_init"]),
            "kmeans_max_iter": int(shape_cluster_cfg["kmeans_max_iter"]),
        },
    )

    print(
        f"[run_shape_cluster] objdata_tag={objdata_tag} objects={len(object_names)} "
        f"scale_tag={shape_cluster_cfg['scale_tag']} device={device} cluster_tag={cluster_tag} "
        f"output={output_dir} total_sec={time.perf_counter() - start:.3f}"
    )


if __name__ == "__main__":
    main()
