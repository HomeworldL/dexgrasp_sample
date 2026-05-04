#!/usr/bin/env python3
"""Visualize shape-cluster members as top-view thumbnail grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from src.shape_cluster import build_cluster_tag
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    generated_dataset_root_from_config,
    load_asset_config,
    objdata_tag_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustered object thumbnails by cluster.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--cluster-tag", type=str, default="")
    parser.add_argument("--cluster-id", type=int, default=None)
    parser.add_argument("--members-per-page", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--point-size", type=float, default=0.8)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
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


def _load_mesh_points(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Empty mesh scene: {mesh_path}")
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"No trimesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid vertices in mesh: {mesh_path}")
    return vertices


def _draw_top_view(ax: plt.Axes, points: np.ndarray, point_size: float, highlight: bool) -> None:
    xy = points[:, :2]
    center = xy.mean(axis=0, keepdims=True)
    xy = xy - center
    radius = np.abs(xy).max()
    if radius <= 1e-8:
        radius = 1.0
    ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c="black", alpha=0.85, linewidths=0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0 if highlight else 0.8)
        spine.set_color("tab:red" if highlight else "#999999")


def _short_object_name(name: str, max_len: int = 22) -> str:
    if len(name) <= max_len:
        return name
    return f"{name[: max_len - 1]}…"


def _render_cluster_pages(
    objdata_root: Path,
    output_dir: Path,
    cluster_id: str,
    cluster_payload: Dict,
    scale_tag: str,
    members_per_page: int,
    columns: int,
    point_size: float,
    dpi: int,
) -> List[Path]:
    members = list(cluster_payload["members"])
    if members_per_page <= 0:
        raise ValueError("members_per_page must be > 0.")
    if columns <= 0:
        raise ValueError("columns must be > 0.")
    rows = int(math.ceil(members_per_page / columns))
    total_pages = int(math.ceil(len(members) / members_per_page))
    center_object_name = str(cluster_payload["center_object_name"])

    saved_paths: List[Path] = []
    for page_idx in range(total_pages):
        start = page_idx * members_per_page
        end = min(len(members), start + members_per_page)
        page_members = members[start:end]

        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.4))
        axes_arr = np.atleast_1d(axes).reshape(rows, columns)
        for flat_idx, ax in enumerate(axes_arr.flat):
            if flat_idx >= len(page_members):
                ax.axis("off")
                continue
            member = page_members[flat_idx]
            object_name = str(member["object_name"])
            mesh_path = objdata_root / object_name / scale_tag / "coacd.obj"
            points = _load_mesh_points(mesh_path)
            is_center = object_name == center_object_name
            _draw_top_view(ax, points=points, point_size=point_size, highlight=is_center)
            title = (
                f"{member['rank_in_cluster']:02d} {_short_object_name(object_name)}\n"
                f"d={float(member['distance_to_center']):.3f}"
            )
            ax.set_title(title, fontsize=8)

        fig.suptitle(
            f"cluster {cluster_id} | center={center_object_name} | page {page_idx + 1}/{total_pages}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"cluster_{int(cluster_id):02d}_page_{page_idx + 1:02d}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    cfg = load_asset_config(args.config)
    cluster_tag = resolve_cluster_tag(cfg, args.cluster_tag)

    generated_root = Path(generated_dataset_root_from_config(cfg)).resolve()
    objdata_tag = objdata_tag_from_config(cfg, args.config)
    objdata_root = generated_root / objdata_tag
    cluster_dir = objdata_root / "_meta" / "shape_cluster" / cluster_tag
    cluster_index_path = cluster_dir / "cluster_index.json"
    if not cluster_index_path.exists():
        raise FileNotFoundError(f"Missing cluster index: {cluster_index_path}")

    cluster_payload = json.loads(cluster_index_path.read_text(encoding="utf-8"))
    if not isinstance(cluster_payload, dict) or "clusters" not in cluster_payload:
        raise ValueError(f"Invalid cluster index payload: {cluster_index_path}")
    clusters = dict(cluster_payload["clusters"])
    scale_tag = str(cluster_payload.get("scale_tag", resolve_shape_cluster_cfg(cfg)["scale_tag"]))

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (cluster_dir / "vis").resolve()
    )

    target_cluster_ids = sorted(clusters.keys(), key=int)
    if args.cluster_id is not None:
        target_cluster_ids = [str(int(args.cluster_id))]
        if target_cluster_ids[0] not in clusters:
            raise KeyError(f"Cluster id {args.cluster_id} not found in {cluster_index_path}")

    total_pages = 0
    for cluster_id in target_cluster_ids:
        saved_paths = _render_cluster_pages(
            objdata_root=objdata_root,
            output_dir=output_dir,
            cluster_id=cluster_id,
            cluster_payload=clusters[cluster_id],
            scale_tag=scale_tag,
            members_per_page=int(args.members_per_page),
            columns=int(args.columns),
            point_size=float(args.point_size),
            dpi=int(args.dpi),
        )
        total_pages += len(saved_paths)
        print(
            f"[vis_shape_cluster] cluster={cluster_id} members={clusters[cluster_id]['member_count']} "
            f"pages={len(saved_paths)} center={clusters[cluster_id]['center_object_name']}"
        )
    print(
        f"[vis_shape_cluster] cluster_tag={cluster_tag} scale_tag={scale_tag} "
        f"clusters={len(target_cluster_ids)} output_dir={output_dir} pages={total_pages}"
    )


if __name__ == "__main__":
    main()
