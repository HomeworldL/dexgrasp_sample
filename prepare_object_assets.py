#!/usr/bin/env python3
"""Prepare hand-independent object-scale assets."""

import argparse
import time
from pathlib import Path

import numpy as np

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    asset_scales_from_config,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    load_asset_config,
    objdata_tag_from_config,
    raw_dataset_name_from_config,
    raw_dataset_root_from_config,
)
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_sample import (
    global_normals_path,
    global_pc_path,
    load_global_pc_and_normals,
    write_global_normals,
    write_global_pc,
)
from utils.utils_seed import set_seed, stable_seed


def _global_pc_ready(asset_dir: str, render_subdir: str, n_points: int) -> bool:
    pc_path = global_pc_path(asset_dir, render_subdir)
    normals_path = global_normals_path(asset_dir, render_subdir)
    if not pc_path.exists() or not normals_path.exists():
        return False
    try:
        points, normals = load_global_pc_and_normals(asset_dir, render_subdir)
    except Exception:
        return False
    return (
        points.shape == (int(n_points), 3)
        and normals.shape == (int(n_points), 3)
        and points.dtype == np.float32
        and normals.dtype == np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or rebuild hand-independent objdata assets, then prepare "
            "deterministic global point clouds."
        )
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clean and fully rebuild all objdata scale directories described by the "
            "config before writing outputs. Use this for first-time objdata creation "
            "or any rebuild."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    total_start = time.perf_counter()
    cfg = load_asset_config(args.config)
    base_seed = int(cfg["seed"])
    set_seed(base_seed)

    objdata_tag = objdata_tag_from_config(cfg, args.config)
    render_subdir = str(cfg["warp_render"]["output_subdir"])
    n_points = int(cfg["sampling"]["n_points"])

    if args.verbose:
        if args.force:
            print(
                "[prepare_object_assets] mode=force_rebuild "
                "will clean and rebuild all objdata scale directories from config."
            )
        else:
            print(
                "[prepare_object_assets] mode=validate_existing "
                "expects all objdata scale assets from config to already exist; "
                "missing assets will raise an error."
            )

    ds = DatasetObjects(
        raw_dataset_root=raw_dataset_root_from_config(cfg),
        raw_dataset_name=raw_dataset_name_from_config(cfg),
        scales=asset_scales_from_config(cfg),
        objdata_tag=objdata_tag,
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        rebuild_existing_assets=bool(args.force),
        verbose=bool(args.verbose or data_verbose_from_config(cfg)),
    )
    entries = sorted(ds.get_entries(), key=lambda item: int(item["global_id"]))

    prepared = 0
    skipped = 0
    for entry in entries:
        asset_dir = str(entry["asset_dir_abs"])
        object_scale_key = str(entry["object_scale_key"])
        if (not args.force) and _global_pc_ready(asset_dir, render_subdir, n_points):
            skipped += 1
            if args.verbose:
                print(
                    "[prepare_object_assets] skip existing global_pc/global_normals: "
                    f"{object_scale_key}"
                )
            continue

        entry_seed = stable_seed(base_seed, object_scale_key, "global_pc")
        set_seed(entry_seed)
        points, normals = sample_surface_o3d(
            str(entry["coacd_abs"]),
            n_points=n_points,
            method="poisson",
        )
        pc_path = write_global_pc(points, asset_dir, render_subdir)
        normals_path = write_global_normals(normals, asset_dir, render_subdir)
        prepared += 1
        if args.verbose:
            print(
                f"[prepare_object_assets] wrote {object_scale_key}: "
                f"{Path(pc_path).name}, {Path(normals_path).name}"
            )

    print(
        f"[prepare_object_assets] objdata_tag={objdata_tag} entries={len(entries)} "
        f"prepared_pc={prepared} skipped_pc={skipped} "
        f"total_sec={time.perf_counter() - total_start:.3f}"
    )


if __name__ == "__main__":
    main()
