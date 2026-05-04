#!/usr/bin/env python3
"""Prepare hand-independent object-scale assets."""

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.scale_dataset_builder import ScaleDatasetBuilder
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    asset_scales_from_config,
    build_native_asset_from_config,
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


def _to_abs_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


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


def _load_rebuild_tasks(cfg: dict, config_path: str) -> list[dict]:
    raw_dataset_root = Path(raw_dataset_root_from_config(cfg)).resolve()
    raw_dataset_name = raw_dataset_name_from_config(cfg)
    manifest_path = raw_dataset_root / raw_dataset_name / "manifest.process_meshes.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"Invalid manifest objects list: {manifest_path}")

    default_mass = float(manifest.get("summary", {}).get("default_mass_kg", 0.1))
    generated_root = generated_dataset_root_from_config(cfg)
    objdata_tag = objdata_tag_from_config(cfg, config_path)
    scales = asset_scales_from_config(cfg)
    build_native = build_native_asset_from_config(cfg)

    tasks: list[dict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("process_status", "")).lower() != "success":
            continue

        object_name = str(obj.get("object_id") or obj.get("name") or "").strip()
        if not object_name:
            continue

        mesh_path_raw = obj.get("mesh_path")
        if not isinstance(mesh_path_raw, str) or not mesh_path_raw:
            continue

        object_dir = _to_abs_path(mesh_path_raw).parent
        coacd_abs = str((object_dir / "coacd.obj").resolve())
        pm = obj.get("principal_moments")
        if not isinstance(pm, list) or len(pm) != 3:
            pm = [1e-6, 1e-6, 1e-6]

        tasks.append(
            {
                "generated_dataset_root": generated_root,
                "objdata_tag": objdata_tag,
                "object_name": object_name,
                "coacd_abs": coacd_abs,
                "mass_kg": float(obj.get("mass_kg", default_mass)),
                "principal_moments": [float(pm[0]), float(pm[1]), float(pm[2])],
                "scales": [float(s) for s in scales],
                "build_native": bool(build_native),
            }
        )

    if not tasks:
        raise RuntimeError(
            f"No valid manifest objects found under root={raw_dataset_root} dataset={raw_dataset_name}."
        )
    return tasks


def _load_existing_entries(cfg: dict, config_path: str) -> list[dict]:
    builder = ScaleDatasetBuilder(generated_dataset_root_from_config(cfg))
    objdata_tag = objdata_tag_from_config(cfg, config_path)
    entries: list[dict] = []

    for task in _load_rebuild_tasks(cfg, config_path):
        object_name = str(task["object_name"])
        for scale in task["scales"]:
            scale_tag = builder.scale_tag(float(scale))
            scale_dir = builder.base_output_dir / objdata_tag / object_name / scale_tag
            convex_dir = scale_dir / "convex_parts"
            coacd_path = scale_dir / "coacd.obj"
            xml_path = scale_dir / "object.xml"

            if (not xml_path.exists()) or (not coacd_path.exists()) or (not convex_dir.is_dir()):
                raise FileNotFoundError(
                    f"Missing objdata asset for {object_name} scale={float(scale):.6f} "
                    f"under {objdata_tag}. First-time creation or rebuild must use "
                    "prepare_object_assets.py --force."
                )

            convex_parts = [p for p in sorted(convex_dir.glob("*.obj")) if p.is_file()]
            if not convex_parts:
                raise FileNotFoundError(
                    f"Missing convex parts for {object_name} scale={float(scale):.6f} "
                    f"under {objdata_tag}: {convex_dir}"
                )

            entries.append(
                {
                    "object_name": object_name,
                    "object_scale_key": f"{object_name}__{scale_tag}",
                    "asset_dir_abs": str(scale_dir.resolve()),
                    "coacd_abs": str(coacd_path.resolve()),
                }
            )

        if bool(task["build_native"]):
            native_tag = builder.native_tag()
            native_dir = builder.base_output_dir / objdata_tag / object_name / native_tag
            convex_dir = native_dir / "convex_parts"
            coacd_path = native_dir / "coacd.obj"
            xml_path = native_dir / "object.xml"

            if (not xml_path.exists()) or (not coacd_path.exists()) or (not convex_dir.is_dir()):
                raise FileNotFoundError(
                    f"Missing objdata asset for {object_name} scale=native under {objdata_tag}. "
                    "First-time creation or rebuild must use prepare_object_assets.py --force."
                )

            convex_parts = [p for p in sorted(convex_dir.glob("*.obj")) if p.is_file()]
            if not convex_parts:
                raise FileNotFoundError(
                    f"Missing convex parts for {object_name} scale=native under {objdata_tag}: {convex_dir}"
                )

            entries.append(
                {
                    "object_name": object_name,
                    "object_scale_key": f"{object_name}__{native_tag}",
                    "asset_dir_abs": str(native_dir.resolve()),
                    "coacd_abs": str(coacd_path.resolve()),
                }
            )

    return sorted(entries, key=lambda item: item["object_scale_key"])


def _rebuild_object_assets_task(task: dict) -> list[dict]:
    builder = ScaleDatasetBuilder(task["generated_dataset_root"])
    object_name = str(task["object_name"])
    obj_root = Path(task["generated_dataset_root"]).resolve() / str(task["objdata_tag"]) / object_name
    if obj_root.exists():
        shutil.rmtree(obj_root)

    recs = builder.build_multi_scale_assets(
        config_stem=str(task["objdata_tag"]),
        object_info={"object_name": object_name, "coacd_abs": str(task["coacd_abs"])},
        scales=list(task["scales"]),
        mass_kg=float(task["mass_kg"]),
        principal_moments=list(task["principal_moments"]),
        overwrite=False,
    )

    entries: list[dict] = []
    for scale_key, rec in sorted(recs.items()):
        asset_dir_abs = Path(rec["xml_abs"]).resolve().parent
        entries.append(
            {
                "object_name": object_name,
                "object_scale_key": f"{object_name}__{scale_key}",
                "asset_dir_abs": str(asset_dir_abs),
                "coacd_abs": str(rec["coacd_abs"]),
            }
        )

    if bool(task["build_native"]):
        native_rec = builder.build_native_assets(
            config_stem=str(task["objdata_tag"]),
            object_info={"object_name": object_name, "coacd_abs": str(task["coacd_abs"])},
            mass_kg=float(task["mass_kg"]),
            principal_moments=list(task["principal_moments"]),
            overwrite=False,
        )
        asset_dir_abs = Path(native_rec["xml_abs"]).resolve().parent
        entries.append(
            {
                "object_name": object_name,
                "object_scale_key": f"{object_name}__{native_rec['scale_tag']}",
                "asset_dir_abs": str(asset_dir_abs),
                "coacd_abs": str(native_rec["coacd_abs"]),
            }
        )
    return entries


def _prepare_global_pc_task(task: dict) -> str:
    asset_dir = str(task["asset_dir_abs"])
    render_subdir = str(task["render_subdir"])
    n_points = int(task["n_points"])
    if (not bool(task["force"])) and _global_pc_ready(asset_dir, render_subdir, n_points):
        return "skipped"

    set_seed(int(task["seed"]))
    points, normals = sample_surface_o3d(
        str(task["coacd_abs"]),
        n_points=n_points,
        method="poisson",
    )
    write_global_pc(points, asset_dir, render_subdir)
    write_global_normals(normals, asset_dir, render_subdir)
    return "prepared"


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for rebuilding objdata and writing global point clouds.",
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be >= 1.")

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

    if args.force:
        rebuild_start = time.perf_counter()
        object_tasks = _load_rebuild_tasks(cfg, args.config)
        entries: list[dict] = []
        if args.jobs == 1:
            obj_iter = tqdm(
                object_tasks,
                desc=f"rebuild:{objdata_tag}",
                total=len(object_tasks),
                dynamic_ncols=True,
                leave=True,
            )
            for task in obj_iter:
                entries.extend(_rebuild_object_assets_task(task))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.jobs) as pool:
                obj_iter = tqdm(
                    pool.imap_unordered(_rebuild_object_assets_task, object_tasks, chunksize=1),
                    desc=f"rebuild:{objdata_tag}",
                    total=len(object_tasks),
                    dynamic_ncols=True,
                    leave=True,
                )
                for object_entries in obj_iter:
                    entries.extend(object_entries)

        entries = sorted(entries, key=lambda item: item["object_scale_key"])
        print(
            f"[prepare_object_assets] rebuild_assets_sec={time.perf_counter() - rebuild_start:.3f} "
            f"objects={len(object_tasks)} entries={len(entries)} objdata_tag={objdata_tag} jobs={args.jobs}"
        )
    else:
        index_start = time.perf_counter()
        entries = _load_existing_entries(cfg, args.config)
        ds_sec = time.perf_counter() - index_start
        print(
            f"[prepare_object_assets] dataset_index_sec={ds_sec:.3f} "
            f"entries={len(entries)} objdata_tag={objdata_tag}"
        )

    prepared = 0
    skipped = 0
    loop_start = time.perf_counter()
    pc_tasks = [
        {
            "asset_dir_abs": str(entry["asset_dir_abs"]),
            "object_scale_key": str(entry["object_scale_key"]),
            "coacd_abs": str(entry["coacd_abs"]),
            "render_subdir": render_subdir,
            "n_points": n_points,
            "seed": stable_seed(base_seed, str(entry["object_scale_key"]), "global_pc"),
            "force": bool(args.force),
        }
        for entry in entries
    ]
    if args.jobs == 1:
        entry_iter = tqdm(
            pc_tasks,
            desc=f"prepare:{objdata_tag}",
            total=len(pc_tasks),
            dynamic_ncols=True,
            leave=True,
        )
        for task in entry_iter:
            status = _prepare_global_pc_task(task)
            if status == "prepared":
                prepared += 1
                if args.verbose:
                    tqdm.write(
                        f"[prepare_object_assets] wrote {task['object_scale_key']}: "
                        "global_pc.npy, global_normals.npy"
                    )
            else:
                skipped += 1
                if args.verbose:
                    tqdm.write(
                        "[prepare_object_assets] skip existing global_pc/global_normals: "
                        f"{task['object_scale_key']}"
                    )
            entry_iter.set_postfix(prepared=prepared, skipped=skipped)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            entry_iter = tqdm(
                pool.imap_unordered(_prepare_global_pc_task, pc_tasks, chunksize=8),
                desc=f"prepare:{objdata_tag}",
                total=len(pc_tasks),
                dynamic_ncols=True,
                leave=True,
            )
            for status in entry_iter:
                if status == "prepared":
                    prepared += 1
                else:
                    skipped += 1
                entry_iter.set_postfix(prepared=prepared, skipped=skipped)

    print(
        f"[prepare_object_assets] objdata_tag={objdata_tag} entries={len(entries)} "
        f"prepared_pc={prepared} skipped_pc={skipped} "
        f"prepare_loop_sec={time.perf_counter() - loop_start:.3f} "
        f"total_sec={time.perf_counter() - total_start:.3f}"
    )


if __name__ == "__main__":
    main()
