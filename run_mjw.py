import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.mjw_ho import MjWarpHO
from src.sample import downsample_fps, sample_grasp_frames
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_seed import set_seed


def object_name_from_scale_key(object_scale_key: str) -> str:
    if "__" in object_scale_key:
        return object_scale_key.split("__", 1)[0]
    return object_scale_key


def _encode_h5_str(value: str):
    return np.bytes_(str(value))


def _scale_from_object_scale_key(object_scale_key: str) -> Optional[float]:
    if "__" not in object_scale_key:
        return None
    suffix = object_scale_key.split("__", 1)[1]
    if not suffix.startswith("scale"):
        return None
    digits = suffix[len("scale") :]
    if not digits.isdigit():
        return None
    return float(int(digits)) / 1000.0


def _write_grasp_npy_from_h5(h5_path: Path, npy_path: Path) -> None:
    payload: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as hf:
        for key in ("qpos_init", "qpos_approach", "qpos_prepared", "qpos_grasp"):
            if key not in hf:
                raise KeyError(f"Missing dataset '{key}' in {h5_path}")
            payload[key] = np.asarray(hf[key][:], dtype=np.float32)
    np.save(npy_path, payload, allow_pickle=True)


def _grasp_outputs_exist(output_dir_abs: str) -> bool:
    out_dir = Path(output_dir_abs)
    return (out_dir / "grasp.h5").exists() and (out_dir / "grasp.npy").exists()


def compose_rot_grasp_to_palm(cfg: Dict) -> np.ndarray:
    base = np.asarray(cfg["transform"]["base_rot_grasp_to_palm"], dtype=float)
    extra = cfg["transform"]["extra_euler"]
    extra_rot = R.from_euler(extra["axis"], float(extra["degrees"]), degrees=True).as_matrix()
    return (base @ extra_rot).T


def sample_frames_from_points(cfg: Dict, pts: np.ndarray, norms: np.ndarray) -> np.ndarray:
    sampling_cfg = cfg["sampling"]
    transforms = sample_grasp_frames(
        pts,
        norms,
        Nd=int(sampling_cfg["Nd"]),
        rot_n=int(sampling_cfg["rot_n"]),
        d_min=float(sampling_cfg["d_min"]),
        d_max=float(sampling_cfg["d_max"]),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=sampling_cfg["max_points"],
    )
    transforms_np = transforms.cpu().numpy()
    del transforms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transforms_np


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
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1).astype(np.float32)

    n = qpos_prepared_sample.shape[0]
    qpos_approach_sample = qpos_prepared_sample.copy()
    qpos_approach_sample[:, 7:] = np.tile(approach_joints, (n, 1))

    shift_local = np.asarray(cfg["hand"]["shift_local"], dtype=float)
    positions = qpos_prepared_sample[:, :3]
    quats_wxyz = qpos_prepared_sample[:, 3:7]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    offset_world = R.from_quat(quats_xyzw).apply(shift_local)

    qpos_init_sample = qpos_prepared_sample.copy()
    qpos_init_sample[:, :3] = positions + offset_world
    qpos_init_sample[:, 7:] = np.tile(approach_joints, (n, 1))

    return qpos_init_sample, qpos_approach_sample, qpos_prepared_sample


def build_valid_backend(
    obj_info: Dict,
    hand_xml_path: str,
    target_body_params: Optional[Dict],
    device: str,
    nworld: int,
    nconmax: int,
    naconmax: int,
    njmax: Optional[int],
    ccd_iterations: int,
    pts_for_sim: np.ndarray,
    norms_for_sim: np.ndarray,
) -> MjWarpHO:
    backend = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=target_body_params,
        object_fixed=False,
        nworld=int(nworld),
        device=str(device),
        nconmax=int(nconmax),
        naconmax=int(naconmax),
        njmax=int(njmax) if njmax is not None else None,
        ccd_iterations=int(ccd_iterations),
    )
    backend._set_obj_pts_norms(pts_for_sim, norms_for_sim)
    return backend


def shrink_tail_pool(
    active_world_ids: np.ndarray,
    mjw_valid_backend: MjWarpHO,
    world_candidate_ids: np.ndarray,
    world_dir_idx: np.ndarray,
    world_chunk_idx: np.ndarray,
    world_initial_pose: np.ndarray,
    world_pos_delta: np.ndarray,
    world_angle_delta: np.ndarray,
    obj_info: Dict,
    hand_xml_path: str,
    target_body_params: Optional[Dict],
    device: str,
    nconmax: int,
    naconmax: int,
    njmax: Optional[int],
    ccd_iterations: int,
    pts_for_sim: np.ndarray,
    norms_for_sim: np.ndarray,
) -> tuple[MjWarpHO, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    new_pool_size = int(active_world_ids.size)
    next_backend = build_valid_backend(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=target_body_params,
        device=device,
        nworld=new_pool_size,
        nconmax=nconmax,
        naconmax=naconmax,
        njmax=njmax,
        ccd_iterations=ccd_iterations,
        pts_for_sim=pts_for_sim,
        norms_for_sim=norms_for_sim,
    )

    next_world_ids = np.arange(new_pool_size, dtype=np.int32)
    next_qpos = mjw_valid_backend.get_qpos_for_worlds(active_world_ids)
    next_qvel = mjw_valid_backend.get_qvel_for_worlds(active_world_ids)
    next_ctrl = mjw_valid_backend.get_ctrl_for_worlds(active_world_ids)
    next_backend.load_state_to_worlds(
        next_world_ids,
        qpos_batch=next_qpos,
        qvel_batch=next_qvel,
        ctrl_batch=next_ctrl,
        do_forward=True,
    )

    return (
        next_backend,
        world_candidate_ids[active_world_ids].copy(),
        world_dir_idx[active_world_ids].copy(),
        world_chunk_idx[active_world_ids].copy(),
        world_initial_pose[active_world_ids].copy(),
        world_pos_delta[active_world_ids].copy(),
        world_angle_delta[active_world_ids].copy(),
        new_pool_size,
    )


def run_sampling(
    cfg: Dict,
    object_scale_key: str,
    object_id: str,
    scale: Optional[float],
    hand_name: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    output_dir_abs: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
    batch_size: int,
    device: str,
    nconmax: int,
    naconmax: int,
    njmax: Optional[int],
    ccd_iterations: int,
) -> str:
    object_name = object_name_from_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    target_body_params = cfg["hand"].get("target_body_params")

    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )

    transforms_np = sample_frames_from_points(cfg, points, normals)
    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    max_candidates = int(qpos_prepared.shape[0])
    if max_candidates <= 0:
        raise RuntimeError(f"No candidate qpos generated for {object_scale_key}.")

    qpos_prepared = qpos_prepared[:max_candidates]
    qpos_approach = qpos_approach[:max_candidates]
    qpos_init = qpos_init[:max_candidates]

    out_dir = Path(output_dir_abs)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "grasp.h5"
    npy_path = out_dir / "grasp.npy"

    d = qpos_prepared.shape[1]
    max_cap = int(cfg["output"]["max_cap"])
    if max_cap <= 0:
        raise ValueError(f"output.max_cap must be positive, got {max_cap}.")
    flush_every = int(cfg.get("output", {}).get("flush_every", 0) or 0)
    contact_min_count = int(cfg["validation"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))
    sim_grasp_cfg.pop("visualize", None)
    extforce_cfg.pop("visualize", None)

    ts_backend = time.perf_counter()
    mjw_grasp = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=target_body_params,
        object_fixed=True,
        nworld=batch_size,
        device=str(device),
        nconmax=int(nconmax),
        naconmax=int(naconmax),
        njmax=int(njmax) if njmax is not None else None,
        ccd_iterations=int(ccd_iterations),
    )
    mjw_valid = build_valid_backend(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=target_body_params,
        device=str(device),
        nworld=batch_size,
        nconmax=int(nconmax),
        naconmax=int(naconmax),
        njmax=njmax,
        ccd_iterations=int(ccd_iterations),
        pts_for_sim=pts_for_sim,
        norms_for_sim=norms_for_sim,
    )
    backend_init_time = time.perf_counter() - ts_backend
    mjw_grasp._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    collision_batches = 0
    sim_grasp_batches = 0
    extforce_batches = 0
    extforce_shrinks = 0
    collision_time = 0.0
    sim_grasp_time = 0.0
    extforce_time = 0.0
    num_no_col = 0
    num_grasp_contact_ok = 0

    collision_mask = np.zeros((max_candidates,), dtype=bool)
    pbar = tqdm(
        total=max_candidates,
        desc=f"collision-{object_scale_key}",
        miniters=1,
        disable=not verbose,
    )
    for start in range(0, max_candidates, batch_size):
        end = min(start + batch_size, max_candidates)
        valid_count = end - start

        ts = time.perf_counter()
        prepared_result = mjw_grasp.check_contact_batch(qpos_prepared[start:end], valid_count=valid_count)
        approach_result = mjw_grasp.check_contact_batch(qpos_approach[start:end], valid_count=valid_count)
        init_result = mjw_grasp.check_contact_batch(qpos_init[start:end], valid_count=valid_count)
        collision_time += time.perf_counter() - ts
        collision_batches += 1

        no_col_mask = ~(prepared_result.has_contact | approach_result.has_contact | init_result.has_contact)
        collision_mask[start:end] = no_col_mask
        num_no_col += int(no_col_mask.sum())
        pbar.update(valid_count)
    pbar.close()

    qpos_init_no_col = qpos_init[collision_mask]
    qpos_approach_no_col = qpos_approach[collision_mask]
    qpos_prepared_no_col = qpos_prepared[collision_mask]
    no_col_count = int(qpos_prepared_no_col.shape[0])

    grasp_qpos_results = []
    grasp_contact_counts = []
    if no_col_count > 0:
        pbar = tqdm(
            total=no_col_count,
            desc=f"simgrasp-{object_scale_key}",
            miniters=1,
            disable=not verbose,
        )
        for start in range(0, no_col_count, batch_size):
            end = min(start + batch_size, no_col_count)
            valid_count = end - start
            ts = time.perf_counter()
            grasp_result = mjw_grasp.sim_grasp_batch(
                qpos_prepared_no_col[start:end],
                valid_count=valid_count,
                **sim_grasp_cfg,
            )
            sim_grasp_time += time.perf_counter() - ts
            sim_grasp_batches += 1
            grasp_qpos_results.append(grasp_result.qpos_grasp[:valid_count].copy())
            grasp_contact_counts.append(grasp_result.ho_contact_counts[:valid_count].copy())
            pbar.update(valid_count)
        pbar.close()

    if grasp_qpos_results:
        qpos_grasp_all = np.concatenate(grasp_qpos_results, axis=0)
        ho_contact_counts_all = np.concatenate(grasp_contact_counts, axis=0)
    else:
        qpos_grasp_all = np.zeros((0, d), dtype=np.float32)
        ho_contact_counts_all = np.zeros((0,), dtype=np.int32)

    contact_ok_mask = ho_contact_counts_all >= contact_min_count
    num_grasp_contact_ok = int(contact_ok_mask.sum())
    qpos_init_contact_ok = qpos_init_no_col[contact_ok_mask]
    qpos_approach_contact_ok = qpos_approach_no_col[contact_ok_mask]
    qpos_prepared_contact_ok = qpos_prepared_no_col[contact_ok_mask]
    qpos_grasp_contact_ok = qpos_grasp_all[contact_ok_mask]
    contact_ok_count = int(qpos_grasp_contact_ok.shape[0])

    num_valid = 0
    flushed_valid = 0
    start_ts = time.time()

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("object_id", data=_encode_h5_str(object_id))
        hf.create_dataset("object_name", data=_encode_h5_str(object_name))
        hf.create_dataset("scale", data=np.float32(scale if scale is not None else np.nan))
        hf.create_dataset("hand_name", data=_encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=_encode_h5_str("wxyz+qpos"))

        ds_init = hf.create_dataset("qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_approach = hf.create_dataset("qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype="f4")

        def write_valid_candidates(candidate_ids: np.ndarray) -> bool:
            nonlocal num_valid, flushed_valid
            if candidate_ids.size == 0:
                return num_valid >= max_cap
            remaining = max_cap - num_valid
            if remaining <= 0:
                return True
            write_ids = np.asarray(candidate_ids[:remaining], dtype=np.int32)
            n_write = int(write_ids.size)
            end = num_valid + n_write
            ds_init[num_valid:end] = qpos_init_contact_ok[write_ids]
            ds_approach[num_valid:end] = qpos_approach_contact_ok[write_ids]
            ds_prepared[num_valid:end] = qpos_prepared_contact_ok[write_ids]
            ds_grasp[num_valid:end] = qpos_grasp_contact_ok[write_ids]
            num_valid = end
            if flush_every > 0 and (num_valid - flushed_valid) >= flush_every:
                hf.flush()
                flushed_valid = num_valid
            return num_valid >= max_cap

        if contact_ok_count > 0:
            side_swing = set(mjw_valid.hand_profile.get("side_swing_indices", [0, 4, 8, 12, 16]))
            grip_delta = float(extforce_cfg["grip_delta"])
            grip_qpos_all = qpos_grasp_contact_ok.copy()
            for idx in range(7, mjw_valid.nq_hand):
                joint_local_idx = idx - 7
                if joint_local_idx not in side_swing:
                    grip_qpos_all[:, idx] += grip_delta
            if mjw_valid.nu > 0:
                grip_ctrl_all = mjw_valid.qpos_to_ctrl_batch(grip_qpos_all)
            else:
                grip_ctrl_all = np.zeros((contact_ok_count, 0), dtype=np.float32)

            dt = float(mjw_valid.mj_model.opt.timestep)
            n_steps = max(1, int(float(extforce_cfg["duration"]) / max(dt, 1e-8)))
            check_step = max(int(extforce_cfg["check_step"]), 1)
            n_chunks = max(1, int(np.ceil(n_steps / check_step)))
            trans_thresh = float(extforce_cfg["trans_thresh"])
            angle_thresh = float(extforce_cfg["angle_thresh"])
            force_mag = float(extforce_cfg["force_mag"])
            force_dirs = np.asarray(
                [
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )

            current_pool_size = int(batch_size)
            world_active = np.zeros((current_pool_size,), dtype=bool)
            world_candidate_ids = np.full((current_pool_size,), -1, dtype=np.int32)
            world_dir_idx = np.zeros((current_pool_size,), dtype=np.int32)
            world_chunk_idx = np.zeros((current_pool_size,), dtype=np.int32)
            world_initial_pose = np.zeros((current_pool_size, 7), dtype=np.float32)
            world_pos_delta = np.zeros((current_pool_size,), dtype=np.float32)
            world_angle_delta = np.zeros((current_pool_size,), dtype=np.float32)
            next_candidate = 0

            def load_candidates_into_worlds(
                world_ids: np.ndarray,
                candidate_ids: np.ndarray,
                reset_metrics: bool,
            ) -> None:
                if world_ids.size == 0:
                    return
                mjw_valid.reset_worlds(world_ids)
                mjw_valid.load_hand_qpos_to_worlds(
                    world_ids,
                    qpos_grasp_contact_ok[candidate_ids],
                    ctrl_batch=grip_ctrl_all[candidate_ids],
                    do_forward=True,
                )
                world_active[world_ids] = True
                world_candidate_ids[world_ids] = candidate_ids
                world_chunk_idx[world_ids] = 0
                world_initial_pose[world_ids] = mjw_valid.get_obj_pose_for_worlds(world_ids)
                if reset_metrics:
                    world_dir_idx[world_ids] = 0
                    world_pos_delta[world_ids] = 0.0
                    world_angle_delta[world_ids] = 0.0

            initial_fill = min(current_pool_size, contact_ok_count)
            initial_world_ids = np.arange(initial_fill, dtype=np.int32)
            initial_candidate_ids = np.arange(initial_fill, dtype=np.int32)
            load_candidates_into_worlds(initial_world_ids, initial_candidate_ids, reset_metrics=True)
            next_candidate = initial_fill

            pbar = tqdm(
                total=contact_ok_count,
                desc=f"extforce-{object_scale_key}",
                miniters=1,
                disable=not verbose,
            )
            while np.any(world_active):
                active_world_ids = np.flatnonzero(world_active).astype(np.int32)
                if (
                    next_candidate >= contact_ok_count
                    and active_world_ids.size > 0
                    and active_world_ids.size <= max(1, current_pool_size // 2)
                ):
                    (
                        mjw_valid,
                        world_candidate_ids,
                        world_dir_idx,
                        world_chunk_idx,
                        world_initial_pose,
                        world_pos_delta,
                        world_angle_delta,
                        current_pool_size,
                    ) = shrink_tail_pool(
                        active_world_ids=active_world_ids,
                        mjw_valid_backend=mjw_valid,
                        world_candidate_ids=world_candidate_ids,
                        world_dir_idx=world_dir_idx,
                        world_chunk_idx=world_chunk_idx,
                        world_initial_pose=world_initial_pose,
                        world_pos_delta=world_pos_delta,
                        world_angle_delta=world_angle_delta,
                        obj_info=obj_info,
                        hand_xml_path=hand_xml_path,
                        target_body_params=target_body_params,
                        device=device,
                        nconmax=nconmax,
                        naconmax=naconmax,
                        njmax=njmax,
                        ccd_iterations=ccd_iterations,
                        pts_for_sim=pts_for_sim,
                        norms_for_sim=norms_for_sim,
                    )
                    extforce_shrinks += 1
                    world_active = np.ones((current_pool_size,), dtype=bool)
                    active_world_ids = np.arange(current_pool_size, dtype=np.int32)

                active_force = np.zeros((active_world_ids.size, 6), dtype=np.float32)
                active_force[:, :3] = force_dirs[world_dir_idx[active_world_ids], :3] * force_mag
                mjw_valid.set_object_force_to_worlds(active_world_ids, active_force)

                ts = time.perf_counter()
                mjw_valid.step_batch(check_step)
                extforce_time += time.perf_counter() - ts
                extforce_batches += 1

                has_contact = mjw_valid.read_contact_mask_for_worlds(active_world_ids)
                current_obj_pose = mjw_valid.get_obj_pose_for_worlds(active_world_ids)
                dir_pos_delta, dir_angle_delta = mjw_valid._pose_delta_batch(
                    world_initial_pose[active_world_ids], current_obj_pose
                )
                world_pos_delta[active_world_ids] = np.maximum(
                    world_pos_delta[active_world_ids], dir_pos_delta
                )
                world_angle_delta[active_world_ids] = np.maximum(
                    world_angle_delta[active_world_ids], dir_angle_delta
                )

                failed_mask = (~has_contact) | (dir_pos_delta >= trans_thresh) | (
                    dir_angle_delta >= angle_thresh
                )
                failed_worlds = active_world_ids[failed_mask]
                alive_worlds = active_world_ids[~failed_mask]

                advance_mask = (world_chunk_idx[alive_worlds] + 1) >= n_chunks
                dir_complete_worlds = alive_worlds[advance_mask]
                still_running_worlds = alive_worlds[~advance_mask]
                world_chunk_idx[still_running_worlds] += 1

                final_valid_worlds = dir_complete_worlds[world_dir_idx[dir_complete_worlds] >= 5]
                next_dir_worlds = dir_complete_worlds[world_dir_idx[dir_complete_worlds] < 5]

                if next_dir_worlds.size > 0:
                    world_dir_idx[next_dir_worlds] += 1
                    next_dir_candidates = world_candidate_ids[next_dir_worlds]
                    load_candidates_into_worlds(
                        next_dir_worlds,
                        next_dir_candidates,
                        reset_metrics=False,
                    )

                reached_cap = False
                if final_valid_worlds.size > 0:
                    final_valid_candidate_ids = world_candidate_ids[final_valid_worlds]
                    reached_cap = write_valid_candidates(final_valid_candidate_ids)

                done_worlds = np.concatenate([failed_worlds, final_valid_worlds], axis=0)
                if done_worlds.size > 0:
                    world_active[done_worlds] = False
                    world_candidate_ids[done_worlds] = -1
                    world_dir_idx[done_worlds] = 0
                    world_chunk_idx[done_worlds] = 0
                    world_initial_pose[done_worlds] = 0.0
                    world_pos_delta[done_worlds] = 0.0
                    world_angle_delta[done_worlds] = 0.0
                    pbar.update(int(done_worlds.size))

                    refill_count = 0
                    if not reached_cap:
                        refill_count = min(int(done_worlds.size), contact_ok_count - next_candidate)
                        if refill_count > 0:
                            refill_worlds = done_worlds[:refill_count]
                            refill_candidate_ids = np.arange(
                                next_candidate,
                                next_candidate + refill_count,
                                dtype=np.int32,
                            )
                            load_candidates_into_worlds(
                                refill_worlds,
                                refill_candidate_ids,
                                reset_metrics=True,
                            )
                            next_candidate += refill_count

                    free_worlds = done_worlds[refill_count:]
                    if free_worlds.size > 0:
                        mjw_valid.reset_worlds(free_worlds)

                if reached_cap:
                    break
            pbar.close()

        ds_init.resize((num_valid, d))
        ds_approach.resize((num_valid, d))
        ds_prepared.resize((num_valid, d))
        ds_grasp.resize((num_valid, d))
        hf.flush()

    duration = time.time() - start_ts
    if verbose:
        print(
            f"[{object_scale_key}] candidates={max_candidates} no_col={num_no_col} "
            f"contact_ok={num_grasp_contact_ok} valid={num_valid} target_valid_cap={max_cap}"
        )
        print(
            f"[{object_scale_key}] batches collision={collision_batches} "
            f"sim_grasp={sim_grasp_batches} extforce_loops={extforce_batches} "
            f"extforce_shrinks={extforce_shrinks}"
        )
        print(
            f"[{object_scale_key}] time backend_init={backend_init_time:.3f}s "
            f"collision={collision_time:.3f}s sim_grasp={sim_grasp_time:.3f}s "
            f"extforce={extforce_time:.3f}s total={duration:.3f}s out={h5_path}"
        )

    _write_grasp_npy_from_h5(h5_path, npy_path)
    if verbose:
        print(f"[{object_scale_key}] converted {h5_path.name} -> {npy_path.name}")
    return str(h5_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Sample grasps for one object-scale entry with MJWarp.")
    p.add_argument("--object-scale-key", type=str, required=True, help="Unique object-scale key.")
    p.add_argument("--coacd-path", type=str, required=True, help="Path to scaled COACD mesh OBJ.")
    p.add_argument("--mjcf-path", type=str, required=True, help="Path to scaled object MJCF.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for grasp artifacts.")
    p.add_argument("--scale", type=float, default=None, help="Object scale metadata for grasp.h5.")
    p.add_argument("--object-id", type=str, default=None, help="Object id metadata for grasp.h5.")
    p.add_argument("--force", action="store_true", help="Re-run even if grasp.h5 and grasp.npy already exist.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--nconmax", type=int, default=32)
    p.add_argument("--naconmax", type=int, default=131072)
    p.add_argument("--njmax", type=int, default=200)
    p.add_argument("--ccd-iterations", type=int, default=200)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")

    if (not args.force) and _grasp_outputs_exist(args.output_dir):
        if verbose:
            print(f"[{args.object_scale_key}] skip existing grasp.h5 and grasp.npy in {args.output_dir}")
        return

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = Path(hand_xml_path).stem
    n_points = int(cfg["sampling"]["n_points"])
    pts, norms = sample_surface_o3d(args.coacd_path, n_points=n_points, method="poisson")
    object_name = object_name_from_scale_key(args.object_scale_key)
    object_id = str(args.object_id) if args.object_id else object_name
    scale = args.scale if args.scale is not None else _scale_from_object_scale_key(args.object_scale_key)

    run_sampling(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        object_id=object_id,
        scale=scale,
        hand_name=hand_name,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        output_dir_abs=args.output_dir,
        points=pts,
        normals=norms,
        verbose=verbose,
        batch_size=int(args.batch_size),
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
        njmax=int(args.njmax) if args.njmax is not None else None,
        ccd_iterations=int(args.ccd_iterations),
    )


if __name__ == "__main__":
    main()
