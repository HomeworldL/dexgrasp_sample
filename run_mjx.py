import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.mj_ho_mjx import MjHOMJX
from src.sample_mjx import downsample_fps, sample_grasp_frames
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_pointcloud import sample_surface_o3d


def set_seed(random_seed: int):
    np.random.seed(random_seed)
    random.seed(random_seed)


def object_name_from_scale_key(object_scale_key: str) -> str:
    if "__" in object_scale_key:
        return object_scale_key.split("__", 1)[0]
    return object_scale_key


def compose_rot_grasp_to_palm(cfg: Dict) -> np.ndarray:
    base = np.asarray(cfg["transform"]["base_rot_grasp_to_palm"], dtype=float)
    extra = cfg["transform"]["extra_euler"]
    extra_rot = R.from_euler(extra["axis"], float(extra["degrees"]), degrees=True).as_matrix()
    return (base @ extra_rot).T


def sample_frames_from_points(cfg: Dict, pts: np.ndarray, norms: np.ndarray) -> np.ndarray:
    sampling_cfg = cfg["sampling"]
    return sample_grasp_frames(
        pts,
        norms,
        Nd=int(sampling_cfg["Nd"]),
        d_min=float(sampling_cfg["d_min"]),
        d_max=float(sampling_cfg["d_max"]),
        max_points=sampling_cfg["max_points"],
        seed=int(cfg["seed"]),
    )


def build_pose_candidates(cfg: Dict, transforms_np: np.ndarray) -> np.ndarray:
    rot_grasp_to_palm = compose_rot_grasp_to_palm(cfg)
    rotation_matrices = transforms_np[:, :3, :3] @ rot_grasp_to_palm
    positions = transforms_np[:, :3, 3]
    quaternions = R.from_matrix(rotation_matrices).as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)
    return np.concatenate([positions, quaternions], axis=1).astype(np.float32)


def make_qpos_triplets(cfg: Dict, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float32)
    approach_joints = np.asarray(cfg["hand"]["approach_joints"], dtype=np.float32)

    q_expanded = np.tile(prepared_joints, (pose.shape[0], 1)).astype(np.float32)
    qpos_prepared = np.concatenate([pose, q_expanded], axis=1).astype(np.float32)

    n = qpos_prepared.shape[0]
    qpos_approach = qpos_prepared.copy()
    qpos_approach[:, 7:] = np.tile(approach_joints, (n, 1))

    shift_local = np.asarray(cfg["hand"]["shift_local"], dtype=float)
    positions = qpos_prepared[:, :3]
    quats_wxyz = qpos_prepared[:, 3:7]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    offset_world = R.from_quat(quats_xyzw).apply(shift_local)

    qpos_init = qpos_prepared.copy()
    qpos_init[:, :3] = positions + offset_world
    qpos_init[:, 7:] = np.tile(approach_joints, (n, 1))

    return qpos_init, qpos_approach, qpos_prepared


def _append_pool(pool: Dict[str, np.ndarray], values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not pool:
        return {k: v.copy() for k, v in values.items()}
    out = {}
    for k in values.keys():
        out[k] = np.concatenate([pool[k], values[k]], axis=0)
    return out


def _pop_batch(pool: Dict[str, np.ndarray], batch_size: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    head = {k: v[:batch_size] for k, v in pool.items()}
    tail = {k: v[batch_size:] for k, v in pool.items()}
    return head, tail


def _pool_size(pool: Dict[str, np.ndarray]) -> int:
    if not pool:
        return 0
    k0 = next(iter(pool.keys()))
    return int(pool[k0].shape[0])


def run_sampling_mjx(
    cfg: Dict,
    object_scale_key: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    output_dir_abs: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
) -> str:
    object_name = object_name_from_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}

    target_body_params = cfg["hand"]["target_body_params"]
    sampling_cfg = cfg["sampling"]
    contact_min_count = int(cfg["validation"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))

    mjx_cfg = dict(cfg.get("mjx", {}))
    mjx_impl = str(mjx_cfg.get("impl", "jax"))
    mjx_device = mjx_cfg.get("device", None)
    mjx_naconmax = mjx_cfg.get("naconmax", None)

    batch_cfg = dict(mjx_cfg.get("batch", {}))
    precheck_batch_size = int(batch_cfg.get("precheck_batch_size", 256))
    sim_grasp_batch_size = int(batch_cfg.get("sim_grasp_batch_size", 128))
    extforce_batch_size = int(batch_cfg.get("extforce_batch_size", 64))
    drop_tail = bool(dict(mjx_cfg.get("tail", {})).get("drop_tail", False))

    runner = MjHOMJX(
        obj_info,
        hand_xml_path,
        target_body_params=target_body_params,
        object_fixed=True,
        mjx_impl=mjx_impl,
        mjx_device=mjx_device,
        mjx_naconmax=mjx_naconmax,
    )
    runner_valid = MjHOMJX(
        obj_info,
        hand_xml_path,
        target_body_params=target_body_params,
        object_fixed=False,
        mjx_impl=mjx_impl,
        mjx_device=mjx_device,
        mjx_naconmax=mjx_naconmax,
    )

    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    runner.set_object_points(pts_for_sim, norms_for_sim)
    runner_valid.set_object_points(pts_for_sim, norms_for_sim)

    ts = time.time()
    transforms_np = sample_frames_from_points(cfg, points, normals)
    if verbose:
        print(f"[{object_scale_key}] frame sampling time: {time.time() - ts:.3f}s, N={len(transforms_np)}")

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init_all, qpos_approach_all, qpos_prepared_all = make_qpos_triplets(cfg, pose)

    out_dir = Path(output_dir_abs)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "grasp.h5"

    max_cap = int(cfg["output"]["max_cap"])
    flush_every = int(cfg.get("output", {}).get("flush_every", 0) or 0)
    d = qpos_prepared_all.shape[1]

    num_samples = int(qpos_prepared_all.shape[0])
    num_no_col = 0
    num_sim_grasp_ok = 0
    num_valid = 0

    sim_pool: Dict[str, np.ndarray] = {}
    ext_pool: Dict[str, np.ndarray] = {}

    ts = time.time()
    with h5py.File(h5_path, "w") as hf:
        ds_init = hf.create_dataset("qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_approach = hf.create_dataset("qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype="f4")

        pbar = tqdm(
            range(0, num_samples, precheck_batch_size),
            disable=not verbose,
            desc=f"mjx-sampling-{object_scale_key}",
        )
        for start in pbar:
            if num_valid >= max_cap:
                break

            end = min(start + precheck_batch_size, num_samples)
            q_init = qpos_init_all[start:end]
            q_app = qpos_approach_all[start:end]
            q_pre = qpos_prepared_all[start:end]

            mask_precheck = runner.precheck_batch(q_init, q_app, q_pre)
            if mask_precheck.any():
                num_no_col += int(mask_precheck.sum())
                print(f"Precheck passed {mask_precheck.sum()}/{len(mask_precheck)} samples in batch {start}-{end}.")
    #             passed = {
    #                 "qpos_init": q_init[mask_precheck],
    #                 "qpos_approach": q_app[mask_precheck],
    #                 "qpos_prepared": q_pre[mask_precheck],
    #             }
    #             sim_pool = _append_pool(sim_pool, passed)

    #         while _pool_size(sim_pool) >= sim_grasp_batch_size and num_valid < max_cap:
    #             sim_head, sim_pool = _pop_batch(sim_pool, sim_grasp_batch_size)
    #             q_grasp_batch, contact_count_batch = runner.sim_grasp_batch(
    #                 sim_head["qpos_prepared"],
    #                 sim_grasp_cfg=sim_grasp_cfg,
    #             )
    #             mask_contact = contact_count_batch >= contact_min_count
    #             if mask_contact.any():
    #                 num_sim_grasp_ok += int(mask_contact.sum())
    #                 ext_ready = {
    #                     "qpos_init": sim_head["qpos_init"][mask_contact],
    #                     "qpos_approach": sim_head["qpos_approach"][mask_contact],
    #                     "qpos_prepared": sim_head["qpos_prepared"][mask_contact],
    #                     "qpos_grasp": q_grasp_batch[mask_contact],
    #                 }
    #                 ext_pool = _append_pool(ext_pool, ext_ready)

    #             while _pool_size(ext_pool) >= extforce_batch_size and num_valid < max_cap:
    #                 ext_head, ext_pool = _pop_batch(ext_pool, extforce_batch_size)
    #                 valid_mask, _, _ = runner_valid.extforce_validate_batch(
    #                     ext_head["qpos_grasp"],
    #                     extforce_cfg=extforce_cfg,
    #                 )
    #                 if not valid_mask.any():
    #                     continue

    #                 keep = np.where(valid_mask)[0]
    #                 remain = min(max_cap - num_valid, len(keep))
    #                 keep = keep[:remain]

    #                 ds_init[num_valid : num_valid + remain] = ext_head["qpos_init"][keep].astype(np.float32, copy=False)
    #                 ds_approach[num_valid : num_valid + remain] = ext_head["qpos_approach"][keep].astype(np.float32, copy=False)
    #                 ds_prepared[num_valid : num_valid + remain] = ext_head["qpos_prepared"][keep].astype(np.float32, copy=False)
    #                 ds_grasp[num_valid : num_valid + remain] = ext_head["qpos_grasp"][keep].astype(np.float32, copy=False)
    #                 num_valid += int(remain)

    #                 if flush_every > 0 and num_valid % flush_every == 0:
    #                     hf.flush()

    #     if not drop_tail and num_valid < max_cap:
    #         if _pool_size(sim_pool) > 0:
    #             q_grasp_tail, contact_tail = runner.sim_grasp_batch(sim_pool["qpos_prepared"], sim_grasp_cfg=sim_grasp_cfg)
    #             mask_tail_contact = contact_tail >= contact_min_count
    #             if mask_tail_contact.any():
    #                 ext_ready = {
    #                     "qpos_init": sim_pool["qpos_init"][mask_tail_contact],
    #                     "qpos_approach": sim_pool["qpos_approach"][mask_tail_contact],
    #                     "qpos_prepared": sim_pool["qpos_prepared"][mask_tail_contact],
    #                     "qpos_grasp": q_grasp_tail[mask_tail_contact],
    #                 }
    #                 ext_pool = _append_pool(ext_pool, ext_ready)

    #         if _pool_size(ext_pool) > 0:
    #             valid_tail, _, _ = runner_valid.extforce_validate_batch(
    #                 ext_pool["qpos_grasp"],
    #                 extforce_cfg=extforce_cfg,
    #             )
    #             if valid_tail.any() and num_valid < max_cap:
    #                 keep = np.where(valid_tail)[0]
    #                 remain = min(max_cap - num_valid, len(keep))
    #                 keep = keep[:remain]
    #                 ds_init[num_valid : num_valid + remain] = ext_pool["qpos_init"][keep].astype(np.float32, copy=False)
    #                 ds_approach[num_valid : num_valid + remain] = ext_pool["qpos_approach"][keep].astype(np.float32, copy=False)
    #                 ds_prepared[num_valid : num_valid + remain] = ext_pool["qpos_prepared"][keep].astype(np.float32, copy=False)
    #                 ds_grasp[num_valid : num_valid + remain] = ext_pool["qpos_grasp"][keep].astype(np.float32, copy=False)
    #                 num_valid += int(remain)

    #     ds_init.resize((num_valid, d))
    #     ds_approach.resize((num_valid, d))
    #     ds_prepared.resize((num_valid, d))
    #     ds_grasp.resize((num_valid, d))
    #     hf.flush()

    # duration = time.time() - ts
    # if verbose:
    #     print(
    #         f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
    #         f"sim_ok={num_sim_grasp_ok} valid={num_valid} time={duration:.2f}s out={h5_path}"
    #     )

    # return str(h5_path)


def main():
    p = argparse.ArgumentParser(description="MJX sampling for one object-scale entry.")
    p.add_argument("--object-scale-key", type=str, required=True, help="Unique object-scale key.")
    p.add_argument("--coacd-path", type=str, required=True, help="Path to scaled COACD mesh OBJ.")
    p.add_argument("--mjcf-path", type=str, required=True, help="Path to scaled object MJCF.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for grasp artifacts.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    verbose = bool(args.verbose)
    if verbose:
        print(f"[MJX] Using object-scale key: {args.object_scale_key}")

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    n_points = int(cfg["sampling"]["n_points"])
    pts, norms = sample_surface_o3d(args.coacd_path, n_points=n_points, method="poisson")

    run_sampling_mjx(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        output_dir_abs=args.output_dir,
        points=pts,
        normals=norms,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
