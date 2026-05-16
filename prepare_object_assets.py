#!/usr/bin/env python3
"""Prepare hand-independent object-scale assets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.scale_dataset_builder import ScaleDatasetBuilder
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    data_asset_scales_cfg,
    data_build_native_asset_cfg,
    data_generated_dataset_root_cfg,
    data_raw_dataset_name_cfg,
    data_raw_dataset_root_cfg,
    load_asset_config,
    objdata_tag_cfg,
    parse_scale_tag,
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

MANIFEST_NAME = "manifest.process_meshes.json"
POINTCLOUD_SAMPLE_METHOD = "poisson"


def _to_abs_path(path_str: str) -> Path:
    """Resolve a manifest path to an absolute filesystem path."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _global_pc_ready(asset_dir: str, render_subdir: str, n_points: int) -> bool:
    """Check whether the cached global point cloud pair is complete and usable."""
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


def _load_source_manifest(cfg: dict) -> tuple[Path, dict]:
    """Load the raw mesh-process manifest that gates which objects are eligible."""
    raw_dataset_root = Path(data_raw_dataset_root_cfg(cfg)).resolve()
    raw_dataset_name = data_raw_dataset_name_cfg(cfg)
    manifest_path = raw_dataset_root / raw_dataset_name / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"Invalid manifest objects list: {manifest_path}")
    return manifest_path, manifest


def _load_rebuild_tasks(cfg: dict, config_path: str) -> list[dict]:
    """Expand the raw manifest into per-object rebuild tasks.

    Each task carries the source object directory plus the scale/native policy
    needed by the downstream asset builder.
    """
    _, source_manifest = _load_source_manifest(cfg)
    objects = source_manifest["objects"]

    default_mass = float(source_manifest.get("summary", {}).get("default_mass_kg", 0.1))
    generated_root = data_generated_dataset_root_cfg(cfg)
    objdata_tag = objdata_tag_cfg(cfg, config_path)
    scales = data_asset_scales_cfg(cfg)
    build_native = data_build_native_asset_cfg(cfg)

    tasks: list[dict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("process_status", "")).lower() != "success":
            continue

        object_name = str(obj.get("object_id") or obj.get("name") or "").strip()
        mesh_path_raw = obj.get("mesh_path")
        if not object_name or not isinstance(mesh_path_raw, str) or not mesh_path_raw:
            continue

        object_dir = _to_abs_path(mesh_path_raw).parent

        pm = obj.get("principal_moments")
        if not isinstance(pm, list) or len(pm) != 3:
            pm = [1e-6, 1e-6, 1e-6]

        tasks.append(
            {
                "generated_dataset_root": generated_root,
                "objdata_tag": objdata_tag,
                "object_name": object_name,
                "source_dir_abs": str(object_dir.resolve()),
                "mass_kg": float(obj.get("mass_kg", default_mass)),
                "principal_moments": [float(pm[0]), float(pm[1]), float(pm[2])],
                "scales": [float(s) for s in scales],
                "build_native": bool(build_native),
            }
        )

    if not tasks:
        raise RuntimeError("No valid source objects found in raw manifest.")
    return tasks


def _asset_complete(asset_dir: Path) -> bool:
    """Validate the minimum scale/native asset payload expected by objdata."""
    required = [
        ScaleDatasetBuilder.object_xml_path(asset_dir),
        ScaleDatasetBuilder.object_urdf_path(asset_dir),
    ]
    return all(path.exists() for path in required)


def _shared_mesh_root_for_scale(object_root: Path, scale_tag: str) -> Path:
    mesh_subdir = (
        ScaleDatasetBuilder.RAW_MESH_SUBDIR
        if scale_tag == ScaleDatasetBuilder.native_tag()
        else ScaleDatasetBuilder.NORMALIZED_MESH_SUBDIR
    )
    return object_root / mesh_subdir


def _make_entry(
    *,
    object_name: str,
    scale_tag: str,
    asset_dir: Path,
    coacd_path: Path,
) -> dict:
    is_native = scale_tag == ScaleDatasetBuilder.native_tag()
    return {
        "object_name": object_name,
        "object_scale_key": f"{object_name}__{scale_tag}",
        "asset_dir_abs": str(asset_dir.resolve()),
        "coacd_abs": str(coacd_path.resolve()),
        "urdf_abs": str(ScaleDatasetBuilder.object_urdf_path(asset_dir).resolve()),
        "scale_tag": scale_tag,
        "scale": None if is_native else parse_scale_tag(scale_tag),
        "is_native": is_native,
    }


def _delete_object_assets(
    generated_root: str, objdata_tag: str, object_name: str
) -> None:
    """Remove one object's prepared assets so a rebuild starts from a clean root."""
    object_root = Path(generated_root).resolve() / objdata_tag / object_name
    if object_root.exists():
        shutil.rmtree(object_root)


def _rebuild_object_assets_task(task: dict) -> dict:
    """Rebuild one object's shared meshes and all requested scale/native assets.

    Failure policy is intentionally asymmetric:
    - missing shared sources skip the whole object
    - a bad scale/native build only drops that specific asset
    """
    generated_root = str(task["generated_dataset_root"])
    objdata_tag = str(task["objdata_tag"])
    object_name = str(task["object_name"])

    # object-level source guard: skip whole object when source files are missing
    source_dir = Path(str(task["source_dir_abs"])).resolve()
    coacd_path = ScaleDatasetBuilder.coacd_path(source_dir)
    manifold_path = ScaleDatasetBuilder.manifold_path(source_dir)
    if (not coacd_path.exists()) or (not manifold_path.exists()):
        _delete_object_assets(generated_root, objdata_tag, object_name)
        return {
            "object_name": object_name,
            "entries": [],
            "status": "skipped_object_missing_source",
            "skip_reasons": [
                {
                    "scope": "object",
                    "reason": "missing_source",
                    "source_dir_abs": str(source_dir),
                    "coacd_abs": str(coacd_path),
                    "manifold_abs": str(manifold_path),
                }
            ],
        }

    _delete_object_assets(generated_root, objdata_tag, object_name)
    builder = ScaleDatasetBuilder(generated_root)

    object_info = {
        "object_name": object_name,
        "source_dir_abs": str(source_dir),
    }

    entries: list[dict] = []
    skip_reasons: list[dict] = []

    # First build the object-level shared mesh roots. If this fails there is no
    # point attempting any scale-specific MJCF/URDF assets for the object.
    try:
        builder.build_shared_mesh_assets(
            config_stem=objdata_tag,
            object_info=object_info,
            overwrite=False,
        )
    except Exception as exc:
        _delete_object_assets(generated_root, objdata_tag, object_name)
        return {
            "object_name": object_name,
            "entries": [],
            "status": "skipped_object_shared_build_failed",
            "skip_reasons": [
                {
                    "scope": "object",
                    "reason": "shared_build_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    # Then build per-scale assets. These are thin directories that reference the
    # shared meshes, so one failed scale should not poison the whole object.
    for scale in task["scales"]:
        scale_value = float(scale)
        scale_tag = builder.scale_tag(scale_value)
        try:
            rec = builder.build_scale_assets(
                config_stem=objdata_tag,
                object_info=object_info,
                scale=scale_value,
                mass_kg=float(task["mass_kg"]),
                principal_moments=list(task["principal_moments"]),
                overwrite=False,
            )
            asset_dir = Path(rec["xml_abs"]).resolve().parent
            if not _asset_complete(asset_dir):
                skip_reasons.append(
                    {
                        "scope": "scale",
                        "scale_tag": scale_tag,
                        "reason": "incomplete_output",
                    }
                )
                if asset_dir.exists():
                    shutil.rmtree(asset_dir)
                continue
            entries.append(
                _make_entry(
                    object_name=object_name,
                    scale_tag=scale_tag,
                    asset_dir=asset_dir,
                    coacd_path=Path(str(rec["coacd_abs"])),
                )
            )
        except Exception as exc:
            skip_reasons.append(
                {
                    "scope": "scale",
                    "scale_tag": scale_tag,
                    "reason": "build_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            scale_dir = (
                Path(generated_root).resolve() / objdata_tag / object_name / scale_tag
            )
            if scale_dir.exists():
                shutil.rmtree(scale_dir)

    # Native uses the raw shared meshes rather than the normalized shared meshes,
    # but its lifecycle is otherwise the same as a scale-specific asset.
    if bool(task["build_native"]):
        native_tag = builder.native_tag()
        try:
            rec = builder.build_native_assets(
                config_stem=objdata_tag,
                object_info=object_info,
                mass_kg=float(task["mass_kg"]),
                principal_moments=list(task["principal_moments"]),
                overwrite=False,
            )
            asset_dir = Path(rec["xml_abs"]).resolve().parent
            if not _asset_complete(asset_dir):
                skip_reasons.append(
                    {
                        "scope": "scale",
                        "scale_tag": native_tag,
                        "reason": "incomplete_output",
                    }
                )
                if asset_dir.exists():
                    shutil.rmtree(asset_dir)
            else:
                entries.append(
                    _make_entry(
                        object_name=object_name,
                        scale_tag=native_tag,
                        asset_dir=asset_dir,
                        coacd_path=Path(str(rec["coacd_abs"])),
                    )
                )
        except Exception as exc:
            skip_reasons.append(
                {
                    "scope": "scale",
                    "scale_tag": native_tag,
                    "reason": "build_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            native_dir = (
                Path(generated_root).resolve() / objdata_tag / object_name / native_tag
            )
            if native_dir.exists():
                shutil.rmtree(native_dir)

    if not entries:
        _delete_object_assets(generated_root, objdata_tag, object_name)
        status = "skipped_object_no_valid_assets"
    else:
        status = "ok"

    return {
        "object_name": object_name,
        "entries": sorted(entries, key=lambda item: item["object_scale_key"]),
        "status": status,
        "skip_reasons": skip_reasons,
    }


def _write_objdata_manifest(
    *,
    generated_root: str,
    objdata_tag: str,
    source_manifest_path: Path,
    object_results: list[dict],
    scales: list[float],
    build_native: bool,
) -> Path:
    """Write the compact objdata manifest consumed by DatasetObjects.

    The manifest records only object-level availability. Per-entry file paths
    remain implicit from the directory layout.
    """
    dataset_dir = Path(generated_root).resolve() / objdata_tag
    dataset_dir.mkdir(parents=True, exist_ok=True)

    objects: list[dict] = []
    for result in sorted(object_results, key=lambda item: str(item["object_name"])):
        object_entries = list(result["entries"])
        if not object_entries:
            continue
        scale_tags = sorted(str(entry["scale_tag"]) for entry in object_entries)
        objects.append(
            {
                "name": result["object_name"],
                "scales_available": scale_tags,
            }
        )

    out_manifest = {
        "dataset": objdata_tag,
        "summary": {
            "source_manifest": str(source_manifest_path.resolve()),
            "object_count": len(objects),
            "asset_scales": [float(s) for s in scales],
            "build_native_asset": bool(build_native),
        },
        "objects": objects,
    }
    manifest_path = dataset_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def _load_existing_entries(cfg: dict, config_path: str) -> list[dict]:
    """Scan an existing objdata directory back into flat object-scale entries.

    This is the validate-only path used when `--force` is not requested.
    """
    generated_root = data_generated_dataset_root_cfg(cfg)
    objdata_tag = objdata_tag_cfg(cfg, config_path)
    manifest_path = Path(generated_root).resolve() / objdata_tag / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Objdata manifest not found: {manifest_path}. Run prepare_object_assets.py --force first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"Invalid objdata manifest objects list: {manifest_path}")

    entries: list[dict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_name = str(obj.get("name") or "").strip()
        if not object_name:
            continue
        scales_available = obj.get("scales_available", [])
        if not isinstance(scales_available, list):
            continue
        for scale_tag_raw in scales_available:
            scale_tag = str(scale_tag_raw).strip()
            if not scale_tag:
                continue

            # Reconstruct the entry from the new shared-mesh layout instead of
            # relying on per-scale mesh copies.
            asset_dir = (
                Path(generated_root).resolve() / objdata_tag / object_name / scale_tag
            ).resolve()
            if not _asset_complete(asset_dir):
                continue
            shared_root = _shared_mesh_root_for_scale(asset_dir.parent, scale_tag)
            coacd_path = ScaleDatasetBuilder.coacd_path(shared_root)
            manifold_path = ScaleDatasetBuilder.manifold_path(shared_root)
            convex_dir = ScaleDatasetBuilder.convex_parts_dir(shared_root)
            if (
                (not coacd_path.exists())
                or (not manifold_path.exists())
                or (not convex_dir.is_dir())
                or (
                    not any(
                        path.is_file()
                        for path in ScaleDatasetBuilder.convex_part_paths(shared_root)
                    )
                )
            ):
                continue
            entries.append(
                _make_entry(
                    object_name=object_name,
                    scale_tag=scale_tag,
                    asset_dir=asset_dir,
                    coacd_path=coacd_path,
                )
            )
    if not entries:
        raise RuntimeError(
            f"No valid entries found from objdata manifest: {manifest_path}"
        )
    return sorted(entries, key=lambda item: item["object_scale_key"])


def _prepare_global_pc_task(task: dict) -> str:
    """Build or skip one asset's deterministic global point cloud cache."""
    asset_dir = str(task["asset_dir_abs"])
    render_subdir = str(task["render_subdir"])
    n_points = int(task["n_points"])
    if (not bool(task["force"])) and _global_pc_ready(
        asset_dir, render_subdir, n_points
    ):
        return "skipped"

    set_seed(int(task["seed"]))
    points, normals = sample_surface_o3d(
        str(task["coacd_abs"]),
        n_points=n_points,
        method=POINTCLOUD_SAMPLE_METHOD,
        scale=float(task["mesh_scale"]),
    )
    write_global_pc(points, asset_dir, render_subdir)
    write_global_normals(normals, asset_dir, render_subdir)
    return "prepared"


def _run_tasks(
    *,
    tasks: list[dict],
    worker: Callable[[dict], object],
    jobs: int,
    desc: str,
    chunksize: int,
) -> list:
    if jobs == 1:
        return [
            worker(task)
            for task in tqdm(tasks, desc=desc, total=len(tasks), leave=True)
        ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        return list(
            tqdm(
                pool.imap_unordered(worker, tasks, chunksize=chunksize),
                desc=desc,
                total=len(tasks),
                leave=True,
            )
        )


def main() -> None:
    """Entry point for objdata preparation.

    High-level flow:
    1. load and validate the asset config
    2. rebuild objdata assets when `--force` is set, otherwise only validate
    3. ensure every discovered asset has its global point cloud cache
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate or rebuild hand-independent objdata assets, then prepare "
            "deterministic global point clouds."
        )
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be >= 1.")

    total_start = time.perf_counter()
    cfg = load_asset_config(args.config)
    base_seed = int(cfg["seed"])
    set_seed(base_seed)

    generated_root = data_generated_dataset_root_cfg(cfg)
    objdata_tag = objdata_tag_cfg(cfg, args.config)
    scales = data_asset_scales_cfg(cfg)
    build_native = data_build_native_asset_cfg(cfg)

    render_subdir = str(cfg["warp_render"]["output_subdir"])
    n_points = int(cfg["sampling"]["n_points"])

    # Phase 1: materialize objdata assets from the raw processed-mesh manifest,
    # or just rescan the existing objdata tree when running in validate mode.
    if args.force:
        source_manifest_path, _ = _load_source_manifest(cfg)
        object_tasks = _load_rebuild_tasks(cfg, args.config)
        object_results = _run_tasks(
            tasks=object_tasks,
            worker=_rebuild_object_assets_task,
            jobs=args.jobs,
            desc=f"rebuild:{objdata_tag}",
            chunksize=1,
        )

        entries: list[dict] = []
        skipped_object_count = 0
        skipped_scale_count = 0
        for result in object_results:
            entries.extend(result["entries"])
            skipped_scale_count += len(result["skip_reasons"])
            if result["status"] != "ok":
                skipped_object_count += 1

        entries = sorted(entries, key=lambda item: item["object_scale_key"])
        manifest_path = _write_objdata_manifest(
            generated_root=generated_root,
            objdata_tag=objdata_tag,
            source_manifest_path=source_manifest_path,
            object_results=object_results,
            scales=scales,
            build_native=build_native,
        )
        print(
            f"[prepare_object_assets] rebuild objects={len(object_tasks)} "
            f"skipped_objects={skipped_object_count} skipped_scale_events={skipped_scale_count} "
            f"entries={len(entries)} manifest={manifest_path}"
        )
    else:
        entries = _load_existing_entries(cfg, args.config)
        print(
            f"[prepare_object_assets] validate entries={len(entries)} objdata_tag={objdata_tag}"
        )

    # Phase 2: build deterministic global point clouds from the entry-resolved
    # COACD mesh. This runs after the asset scan so both rebuild and validate
    # modes share the same point-cloud preparation path.
    pc_tasks = [
        {
            "asset_dir_abs": str(entry["asset_dir_abs"]),
            "object_scale_key": str(entry["object_scale_key"]),
            "coacd_abs": str(entry["coacd_abs"]),
            "mesh_scale": (
                1.0 if bool(entry.get("is_native", False)) else float(entry["scale"])
            ),
            "render_subdir": render_subdir,
            "n_points": n_points,
            "seed": stable_seed(base_seed, str(entry["object_scale_key"]), "global_pc"),
            "force": bool(args.force),
        }
        for entry in entries
    ]

    pc_statuses = _run_tasks(
        tasks=pc_tasks,
        worker=_prepare_global_pc_task,
        jobs=args.jobs,
        desc=f"prepare:{objdata_tag}",
        chunksize=8,
    )
    prepared = sum(1 for status in pc_statuses if status == "prepared")
    skipped = len(pc_statuses) - prepared

    print(
        f"[prepare_object_assets] objdata_tag={objdata_tag} entries={len(entries)} "
        f"prepared_pc={prepared} skipped_pc={skipped} total_sec={time.perf_counter() - total_start:.3f}"
    )


if __name__ == "__main__":
    main()
