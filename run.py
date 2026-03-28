import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from src.mj_ho import MjHO
from src.sample import downsample_fps
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_sample import (
    ARRAY_DTYPE,
    H5_DTYPE,
    as_array,
    build_pose_candidates,
    encode_h5_str,
    grasp_outputs_exist,
    make_qpos_triplets,
    parse_object_scale_key,
    sample_frames_from_points,
    write_fail_npy_from_h5,
    write_grasp_npy_from_h5,
)
from utils.utils_seed import set_seed


def _select_fail_samples(
    fail_qpos_rows: List[np.ndarray],
    fail_stages: List[str],
    valid_count: int,
    min_valid_count: int,
    fail_keep_ratio: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[str]]:
    if valid_count < min_valid_count:
        return [], []
    keep_count = min(
        len(fail_qpos_rows),
        int(np.floor(float(fail_keep_ratio) * float(valid_count))),
    )
    if keep_count <= 0:
        return [], []
    indices = np.random.default_rng(int(seed)).permutation(len(fail_qpos_rows))[:keep_count]
    selected_qpos = [fail_qpos_rows[int(idx)] for idx in indices]
    selected_stages = [fail_stages[int(idx)] for idx in indices]
    return selected_qpos, selected_stages


def _write_fail_h5(
    fail_h5_path: Path,
    object_name: str,
    scale: Optional[float],
    hand_name: str,
    qpos_dim: int,
    qpos_fail: List[np.ndarray],
    failure_stages: List[str],
) -> None:
    if qpos_fail:
        qpos_fail_np = np.asarray(qpos_fail, dtype=ARRAY_DTYPE)
        failure_stage_np = np.asarray(failure_stages, dtype=object)
    else:
        qpos_fail_np = np.zeros((0, qpos_dim), dtype=ARRAY_DTYPE)
        failure_stage_np = np.asarray([], dtype=object)
    failure_stage_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(fail_h5_path, "w") as hf:
        hf.create_dataset("object_name", data=encode_h5_str(object_name))
        hf.create_dataset("scale", data=np.float32(scale if scale is not None else np.nan))
        hf.create_dataset("hand_name", data=encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))
        hf.create_dataset("qpos_fail", data=qpos_fail_np, dtype=H5_DTYPE)
        hf.create_dataset(
            "failure_stage",
            data=failure_stage_np,
            dtype=failure_stage_dtype,
        )


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
    fail_h5_path = out_dir / str(cfg["output"]["fail_h5_name"])
    fail_npy_path = out_dir / str(cfg["output"]["fail_npy_name"])

    d = qpos_prepared.shape[1]
    max_cap = int(cfg["output"]["max_cap"])
    max_time_sec = float(cfg["output"]["max_time_sec"])
    min_valid_count = int(cfg["output"]["min_valid_count"])
    flush_every = int(cfg["output"]["flush_every"])
    fail_keep_ratio = float(cfg["output"]["fail_keep_ratio"])
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_sim_cfg = dict(extforce_cfg)
    sim_grasp_cfg.pop("visualize", None)
    extforce_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("grip_delta", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    ts = time.time()
    stop_reason = "depleted"
    fail_qpos_rows: List[np.ndarray] = []
    fail_stages: List[str] = []

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("object_name", data=encode_h5_str(object_name))
        hf.create_dataset("scale", data=np.float32(scale if scale is not None else np.nan))
        hf.create_dataset("hand_name", data=encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))

        ds_init = hf.create_dataset("qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE)
        ds_approach = hf.create_dataset("qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE)
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE)
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE)
        ds_squeeze = hf.create_dataset("qpos_squeeze", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE)

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
                fail_qpos_rows.append(as_array(qpos_prepared[i]))
                fail_stages.append("prepared_contact")
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
            ho_contact_num = mjho.get_contact_num(obj_margin=0.00)

            if ho_contact_num >= contact_min_count:
                qpos_grasp = as_array(qpos_grasp)
                qpos_squeeze = mjho_valid.build_squeeze_qpos(
                    qpos_grasp,
                    grip_delta=float(extforce_cfg.get("grip_delta", 0.0)),
                )
                qpos_prepared_valid = mjho_valid.build_pregrasp_qpos(
                    qpos_squeeze,
                    qpos_prepared[i][7:],
                )
                qpos_squeeze = as_array(qpos_squeeze)
                qpos_prepared_valid = as_array(qpos_prepared_valid)
                is_valid, _, _ = mjho_valid.sim_under_extforce(
                    qpos_squeeze.copy(),
                    qpos_prepared_valid.copy(),
                    visualize=False,
                    **extforce_sim_cfg,
                )
                if is_valid:
                    ds_init[num_valid] = qpos_init[i].astype(ARRAY_DTYPE, copy=False)
                    ds_approach[num_valid] = qpos_approach[i].astype(ARRAY_DTYPE, copy=False)
                    ds_prepared[num_valid] = qpos_prepared[i].astype(ARRAY_DTYPE, copy=False)
                    ds_grasp[num_valid] = qpos_grasp.astype(ARRAY_DTYPE, copy=False)
                    ds_squeeze[num_valid] = qpos_squeeze.astype(ARRAY_DTYPE, copy=False)
                    num_valid += 1
                else:
                    fail_qpos_rows.append(qpos_squeeze.astype(ARRAY_DTYPE, copy=False))
                    fail_stages.append("extforce_failure")
            else:
                fail_qpos_rows.append(qpos_grasp.astype(ARRAY_DTYPE, copy=False))
                fail_stages.append("insufficient_contact")

            if flush_every > 0 and (i + 1) % flush_every == 0:
                hf.flush()

        final_size = num_valid if num_valid >= min_valid_count else 0
        ds_init.resize((final_size, d))
        ds_approach.resize((final_size, d))
        ds_prepared.resize((final_size, d))
        ds_grasp.resize((final_size, d))
        ds_squeeze.resize((final_size, d))
        hf.flush()

    num_valid = final_size
    kept_fail_qpos, kept_fail_stages = _select_fail_samples(
        fail_qpos_rows=fail_qpos_rows,
        fail_stages=fail_stages,
        valid_count=num_valid,
        min_valid_count=min_valid_count,
        fail_keep_ratio=fail_keep_ratio,
        seed=int(cfg["seed"]),
    )
    _write_fail_h5(
        fail_h5_path=fail_h5_path,
        object_name=object_name,
        scale=scale,
        hand_name=hand_name,
        qpos_dim=d,
        qpos_fail=kept_fail_qpos,
        failure_stages=kept_fail_stages,
    )
    write_fail_npy_from_h5(fail_h5_path, fail_npy_path)

    duration = time.time() - ts
    total_elapsed = time.perf_counter() - total_stage_start
    print(
        f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
        f"valid={num_valid} fail={len(kept_fail_qpos)} "
        f"time={duration:.2f}s total_elapsed={total_elapsed:.2f}s "
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
