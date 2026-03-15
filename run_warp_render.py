import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

from src.dataset_objects import DatasetObjects
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config
from utils.utils_warp_render import (
    WarpPointCloudRenderer,
    get_camera_matrix,
    intrinsics_from_config,
    mesh_from_path,
    wp,
)


def _require(cfg: Dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[key]
    return cur


def _validate_render_config(cfg: Dict) -> Dict:
    wr = _require(cfg, "warp_render")
    for path in [
        "gpu_lst",
        "thread_per_gpu",
        "output_subdir",
        "max_point_num",
        "save_pc",
        "save_depth",
        "save_rgb",
        "skip_existing",
        "depth_max",
        "tile_width",
        "tile_height",
        "n_cols",
        "n_rows",
        "z_near",
        "z_far",
        "camera",
        "camera.type",
        "camera.radius",
        "camera.pos_noise",
        "camera.lookat",
        "camera.lookat_noise",
        "camera.up_noise",
        "intrinsics",
        "intrinsics.preset",
    ]:
        _require(wr, path)

    if not wr["gpu_lst"]:
        raise ValueError("warp_render.gpu_lst must be non-empty.")
    if int(wr["thread_per_gpu"]) <= 0:
        raise ValueError("warp_render.thread_per_gpu must be > 0.")
    if int(wr["n_cols"]) <= 0 or int(wr["n_rows"]) <= 0:
        raise ValueError("warp_render.n_cols and warp_render.n_rows must be > 0.")
    if int(wr["max_point_num"]) <= 0:
        raise ValueError("warp_render.max_point_num must be > 0.")
    return wr


def _parse_device_token(token: Union[str, int]) -> Union[str, int]:
    if isinstance(token, int):
        return token
    s = str(token).strip().lower()
    if s == "cpu":
        return "cpu"
    return int(s)


def _parse_device_list(values: Sequence[Union[str, int]]) -> List[Union[str, int]]:
    out: List[Union[str, int]] = []
    for value in values:
        out.append(_parse_device_token(value))
    return out


def _device_alias(device_token: Union[str, int]) -> str:
    if device_token == "cpu":
        return "cpu"
    return f"cuda:{int(device_token)}"


def _ensure_devices_available(device_tokens: Sequence[Union[str, int]]) -> None:
    if wp is None:
        raise RuntimeError("warp-lang is not available. Install warp-lang first.")
    wp.init()
    available = set(d.alias for d in wp.get_devices())
    for token in device_tokens:
        alias = _device_alias(token)
        if alias not in available:
            raise RuntimeError(
                f"Requested Warp device '{alias}' is unavailable. "
                f"Available devices: {sorted(available)}. "
                "Use --gpu-lst cpu to run in CPU mode."
            )


def _all_pc_exist(folder: Path, batch: int) -> bool:
    for b in range(batch):
        world_path = folder / f"partial_pc_{str(b).zfill(2)}.npy"
        cam_path = folder / f"partial_pc_cam_{str(b).zfill(2)}.npy"
        if not world_path.exists() or not cam_path.exists():
            return False
        arr = np.load(world_path, allow_pickle=True)
        arr_cam = np.load(cam_path, allow_pickle=True)
        if arr.size == 0:
            return False
        if arr_cam.size == 0:
            return False
    return True


def _all_cam_ex_exist(folder: Path, batch: int) -> bool:
    for b in range(batch):
        if not (folder / f"cam_ex_{str(b).zfill(2)}.npy").exists():
            return False
    return True


def _render_entry(
    renderer: WarpPointCloudRenderer,
    entry: Dict,
    render_cfg: Dict,
    rng: np.random.Generator,
) -> str:
    out_dir = Path(entry["output_dir_abs"]).resolve() / str(render_cfg["output_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = renderer.num_tiles
    if bool(render_cfg["skip_existing"]):
        if bool(render_cfg["save_pc"]) and _all_pc_exist(out_dir, batch):
            return f"skip {entry['object_scale_key']}"
        if (not bool(render_cfg["save_pc"])) and _all_cam_ex_exist(out_dir, batch):
            return f"skip {entry['object_scale_key']}"

    mesh = mesh_from_path(str(entry["coacd_abs"]))
    mesh_radius = float(np.max(np.linalg.norm(np.asarray(mesh.vertices, dtype=np.float64), axis=1)))
    camera_view = get_camera_matrix(
        render_cfg["camera"],
        sample_num=batch,
        rng=rng,
        min_radius=mesh_radius,
    )
    view_matrix = renderer.render_mesh(mesh=mesh, view_matrix=camera_view)

    cam_in_path = out_dir / "cam_in.npy"
    if not cam_in_path.exists():
        np.save(cam_in_path, renderer.projection_matrices[0])

    if bool(render_cfg["save_rgb"]):
        rgb = renderer.get_image(mode="rgb")
    else:
        rgb = None

    if bool(render_cfg["save_depth"]) or bool(render_cfg["save_pc"]):
        depth = renderer.get_image(mode="depth")
    else:
        depth = None

    if bool(render_cfg["save_pc"]):
        all_pc_world = renderer.depth_to_world_point_cloud(depth)
        all_pc_cam = renderer.depth_to_camera_point_cloud(depth)
        depth_mask = (depth.reshape(batch, -1) > 0) & (
            depth.reshape(batch, -1) < float(render_cfg["depth_max"])
        )
    else:
        all_pc_world = None
        all_pc_cam = None
        depth_mask = None

    for b in range(batch):
        data_id = str(b).zfill(2)
        # Convert row-vector C2W matrix to standard homogeneous matrix:
        # p_w = T_c2w @ p_c (column-vector), with [0,0,0,1] as last row.
        cam_ex_row = view_matrix[b].detach().cpu().numpy()
        cam_ex_h = cam_ex_row.T.astype(np.float32, copy=True)
        cam_ex_h[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        np.save(out_dir / f"cam_ex_{data_id}.npy", cam_ex_h)

        if rgb is not None:
            np.save(out_dir / f"rgb_{data_id}.npy", rgb[b].detach().cpu().numpy().astype(np.float32))

        if bool(render_cfg["save_depth"]) and depth is not None:
            np.save(out_dir / f"depth_{data_id}.npy", depth[b].detach().cpu().numpy().astype(np.float32))

        if bool(render_cfg["save_pc"]) and all_pc_world is not None and all_pc_cam is not None and depth_mask is not None:
            pc_world = all_pc_world[b, depth_mask[b]]
            pc_cam = all_pc_cam[b, depth_mask[b]]
            if pc_world.shape[0] > int(render_cfg["max_point_num"]):
                idx = torch.randperm(pc_world.shape[0], device=pc_world.device)[: int(render_cfg["max_point_num"])]
                pc_world = pc_world[idx]
                pc_cam = pc_cam[idx]
            np.save(out_dir / f"partial_pc_{data_id}.npy", pc_world.detach().cpu().numpy().astype(np.float16))
            np.save(out_dir / f"partial_pc_cam_{data_id}.npy", pc_cam.detach().cpu().numpy().astype(np.float16))

    return f"done {entry['object_scale_key']}"


def _batch_worker(
    worker_idx: int,
    device_token: Union[str, int],
    entries: List[Dict],
    render_cfg: Dict,
    seed: int,
    verbose: bool,
) -> None:
    if wp is None:
        raise RuntimeError("warp-lang is not available. Install warp-lang first.")

    device = _device_alias(device_token)
    rng = np.random.default_rng(seed + worker_idx)

    intr = intrinsics_from_config(
        cfg=render_cfg["intrinsics"],
        tile_width=int(render_cfg["tile_width"]),
        tile_height=int(render_cfg["tile_height"]),
    )

    with wp.ScopedDevice(device):
        try:
            renderer = WarpPointCloudRenderer(
                device=device,
                tile_width=int(render_cfg["tile_width"]),
                tile_height=int(render_cfg["tile_height"]),
                n_cols=int(render_cfg["n_cols"]),
                n_rows=int(render_cfg["n_rows"]),
                z_near=float(render_cfg["z_near"]),
                z_far=float(render_cfg["z_far"]),
                intrinsics=intr,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Warp OpenGL renderer on device '{device}'. "
                "Check GPU/EGL/driver availability and headless OpenGL permissions."
            ) from exc

        iterator = entries
        if verbose:
            iterator = tqdm(entries, desc=f"warp-{device}-w{worker_idx}", leave=False)
        for entry in iterator:
            _render_entry(renderer=renderer, entry=entry, render_cfg=render_cfg, rng=rng)


def _split_entries(entries: List[Dict], parts: int) -> List[List[Dict]]:
    if parts <= 1:
        return [entries]
    chunks = [[] for _ in range(parts)]
    for i, entry in enumerate(entries):
        chunks[i % parts].append(entry)
    return [chunk for chunk in chunks if chunk]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render partial point clouds with NVIDIA Warp for each object-scale entry.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument("-j", "--max-parallel", type=int, default=None, help="Limit worker process count.")
    parser.add_argument("--gpu-lst", type=str, default=None, help="Override gpu list, e.g. '0,1'.")
    parser.add_argument("--force", action="store_true", help="Disable skip_existing and re-render outputs.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    render_cfg = _validate_render_config(cfg)

    if args.force:
        render_cfg = dict(render_cfg)
        render_cfg["skip_existing"] = False

    if args.gpu_lst:
        render_cfg = dict(render_cfg)
        render_cfg["gpu_lst"] = [_parse_device_token(x) for x in args.gpu_lst.split(",") if x.strip()]

    config_stem = dataset_tag_from_config(args.config)
    ds = DatasetObjects(
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=config_stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )

    if args.obj_key:
        entries = [ds.get_obj_info_by_scale_key(args.obj_key)]
    elif args.obj_id is not None:
        entries = [ds.get_obj_info_by_index(int(args.obj_id))]
    else:
        entries = sorted(ds.get_entries(), key=lambda x: int(x["global_id"]))

    if not entries:
        print("No object-scale entries to render.")
        return

    configured_devices = _parse_device_list(list(render_cfg["gpu_lst"]))
    _ensure_devices_available(configured_devices)

    worker_devices: List[Union[str, int]] = configured_devices * int(render_cfg["thread_per_gpu"])
    if args.max_parallel is not None:
        worker_devices = worker_devices[: max(1, int(args.max_parallel))]
    if not worker_devices:
        raise RuntimeError("No workers configured. Check warp_render.gpu_lst/thread_per_gpu.")

    chunks = _split_entries(entries, len(worker_devices))
    worker_devices = worker_devices[: len(chunks)]

    print(f"[run_warp_render] entries={len(entries)} workers={len(worker_devices)}")
    print(
        f"[run_warp_render] views_per_entry={int(render_cfg['n_cols']) * int(render_cfg['n_rows'])} "
        f"max_point_num={int(render_cfg['max_point_num'])} output_subdir={render_cfg['output_subdir']}"
    )

    mp.set_start_method("spawn", force=True)
    processes: List[mp.Process] = []
    seed = int(cfg["seed"])

    for worker_idx, (device_token, chunk) in enumerate(zip(worker_devices, chunks)):
        p = mp.Process(
            target=_batch_worker,
            args=(
                worker_idx,
                device_token,
                chunk,
                render_cfg,
                seed,
                bool(args.verbose),
            ),
        )
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError("One or more warp rendering worker processes failed.")

    print("[run_warp_render] all jobs finished")


if __name__ == "__main__":
    main()
