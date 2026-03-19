import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from src.mj_ho import MjHO
from src.sample import downsample_fps
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_sample import (
    build_pose_candidates,
    encode_h5_str,
    grasp_outputs_exist,
    make_qpos_triplets,
    parse_object_scale_key,
    sample_frames_from_points,
    write_grasp_npy_from_h5,
)
from utils.utils_seed import set_seed


def run_sampling(
    cfg: Dict,
    object_scale_key: str,
    hand_name: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    output_dir_abs: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
    total_stage_start: float,
) -> str:
    object_name, parsed_scale = parse_object_scale_key(object_scale_key)
    scale = parsed_scale
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    target_body_params = cfg["hand"]["target_body_params"]

    mjho = MjHO(obj_info, hand_xml_path, target_body_params=target_body_params)
    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    mjho_valid = MjHO(obj_info, hand_xml_path, target_body_params=target_body_params, object_fixed=False)

    ts = time.time()
    transforms_np = sample_frames_from_points(cfg, points, normals)
    if verbose:
        print(f"[{object_scale_key}] frame sampling time: {time.time() - ts:.3f}s, N={len(transforms_np)}")

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    out_dir = Path(output_dir_abs)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "grasp.h5"
    npy_path = out_dir / "grasp.npy"

    d = qpos_prepared.shape[1]
    max_cap = int(cfg["output"]["max_cap"])
    max_time_sec = float(cfg["output"]["max_time_sec"])
    flush_every = int(cfg.get("output", {}).get("flush_every", 0) or 0)
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))
    sim_grasp_cfg.pop("visualize", None)
    extforce_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    ts = time.time()
    stop_reason = "depleted"

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("object_name", data=encode_h5_str(object_name))
        hf.create_dataset("scale", data=np.float32(scale if scale is not None else np.nan))
        hf.create_dataset("hand_name", data=encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))

        ds_init = hf.create_dataset("qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_approach = hf.create_dataset("qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype="f4")

        for i in tqdm(
            range(qpos_prepared.shape[0]),
            desc=f"sampling-{object_scale_key}",
            miniters=50,
            disable=not verbose,
        ):
            if num_valid >= max_cap:
                num_samples = i
                stop_reason = "cap"
                break
            if (time.perf_counter() - total_stage_start) >= max_time_sec:
                num_samples = i
                stop_reason = "timeout"
                break

            mjho.set_hand_qpos(qpos_prepared[i])
            if mjho.is_contact():
                continue
            mjho.set_hand_qpos(qpos_approach[i])
            if mjho.is_contact():
                continue
            mjho.set_hand_qpos(qpos_init[i])
            if mjho.is_contact():
                continue

            num_no_col += 1
            mjho.set_hand_qpos(qpos_prepared[i])
            qpos_grasp, _ = mjho.sim_grasp(visualize=False, **sim_grasp_cfg)
            ho_contact, _ = mjho.get_contact_info(obj_margin=0.00)

            if len(ho_contact) >= contact_min_count:
                is_valid, _, _ = mjho_valid.sim_under_extforce(
                    qpos_grasp.copy(),
                    visualize=False,
                    **extforce_cfg,
                )
                if is_valid:
                    ds_init[num_valid] = qpos_init[i].astype(np.float32, copy=False)
                    ds_approach[num_valid] = qpos_approach[i].astype(np.float32, copy=False)
                    ds_prepared[num_valid] = qpos_prepared[i].astype(np.float32, copy=False)
                    ds_grasp[num_valid] = qpos_grasp.astype(np.float32, copy=False)
                    num_valid += 1

            if flush_every > 0 and (i + 1) % flush_every == 0:
                hf.flush()

        final_size = num_valid
        ds_init.resize((final_size, d))
        ds_approach.resize((final_size, d))
        ds_prepared.resize((final_size, d))
        ds_grasp.resize((final_size, d))
        hf.flush()

    duration = time.time() - ts
    total_elapsed = time.perf_counter() - total_stage_start
    print(
        f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
        f"valid={num_valid} time={duration:.2f}s total_elapsed={total_elapsed:.2f}s "
        f"stop_reason={stop_reason} out={h5_path}"
    )
    write_grasp_npy_from_h5(h5_path, npy_path)
    print(f"[{object_scale_key}] converted {h5_path.name} -> {npy_path.name}")
    return str(h5_path)


def main():
    p = argparse.ArgumentParser(description="Sample grasps for one object-scale entry.")
    p.add_argument("--object-scale-key", type=str, required=True, help="Unique object-scale key.")
    p.add_argument("--coacd-path", type=str, required=True, help="Path to scaled COACD mesh OBJ.")
    p.add_argument("--mjcf-path", type=str, required=True, help="Path to scaled object MJCF.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for grasp artifacts.")
    p.add_argument("--force", action="store_true", help="Re-run even if grasp.h5 and grasp.npy already exist.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    total_stage_start = time.perf_counter()
    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")
    if (not args.force) and grasp_outputs_exist(args.output_dir):
        if verbose:
            print(f"[{args.object_scale_key}] skip existing grasp.h5 and grasp.npy in {args.output_dir}")
        return

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = Path(hand_xml_path).stem
    n_points = int(cfg["sampling"]["n_points"])
    pts, norms = sample_surface_o3d(args.coacd_path, n_points=n_points, method="poisson")
    run_sampling(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_name=hand_name,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        output_dir_abs=args.output_dir,
        points=pts,
        normals=norms,
        verbose=verbose,
        total_stage_start=total_stage_start,
    )


if __name__ == "__main__":
    main()
