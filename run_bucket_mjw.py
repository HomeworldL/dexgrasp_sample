#!/usr/bin/env python3
"""Run one MJWarp multi-object bucket and write per-object grasp outputs."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.bucket_writer import ObjectGraspWriter
from src.dataset_objects import DatasetObjects
from src.mjw_bucket_ho import MjWarpBucketHO
from src.sample import downsample_fps
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_generated_dataset_root_cfg,
    data_run_scales_cfg,
    data_use_native_asset_cfg,
    graspdata_tag_cfg,
    hand_anchor_params_cfg,
    hand_profile_cfg,
    hand_root_stabilization_cfg,
    load_run_config,
    objdata_tag_cfg,
    object_profile_cfg,
)
from utils.utils_sample import (
    build_pose_candidates,
    global_pc_exists,
    grasp_fail_outputs_exist,
    grasp_outputs_exist,
    make_qpos_triplets,
    sample_frames_from_points,
    write_global_normals,
    write_global_pc,
)
from utils.utils_seed import set_seed, stable_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a fixed MJWarp bucket for multiple object-scale entries."
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="JSON run config.",
    )
    p.add_argument(
        "--object-scale-keys",
        nargs="+",
        required=True,
        help="Object-scale keys to include in this bucket.",
    )
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--nconmax", type=int, default=32)
    p.add_argument("--naconmax", type=int, default=16384)
    p.add_argument("--njmax", type=int, default=200)
    p.add_argument("--ccd-iterations", type=int, default=200)
    p.add_argument("--slot-spacing", type=float, default=5.0)
    p.add_argument(
        "--max-cap-override",
        type=int,
        default=None,
        help="Debug-only cap override for quick bucket smoke tests.",
    )
    p.add_argument(
        "--n-points-override",
        type=int,
        default=None,
        help="Debug-only point count override for quick bucket smoke tests.",
    )
    return p.parse_args()


def build_dataset(cfg: Dict, config_path: str) -> DatasetObjects:
    return DatasetObjects(
        scales=data_run_scales_cfg(cfg),
        objdata_tag=objdata_tag_cfg(cfg),
        include_native=data_use_native_asset_cfg(cfg),
        graspdata_tag=graspdata_tag_cfg(cfg, config_path),
        generated_dataset_root=data_generated_dataset_root_cfg(cfg),
        pc_subdir=str(cfg["sampling"]["pc_subdir"]),
        verbose=False,
    )


def entry_output_dir(entry: Dict, output_root: Optional[str]) -> str:
    if output_root is None:
        return str(entry["output_dir_abs"])
    return str(
        (
            Path(output_root) / str(entry["object_name"]) / str(entry["scale_tag"])
        ).resolve()
    )


def outputs_complete(output_dir: str, cfg: Dict) -> bool:
    h5_name = str(cfg["data"]["h5_name"])
    npy_name = str(cfg["data"]["npy_name"])
    fail_h5_name = str(cfg["data"]["fail_h5_name"])
    fail_npy_name = str(cfg["data"]["fail_npy_name"])
    render_subdir = str(cfg["sampling"]["pc_subdir"])
    return (
        grasp_outputs_exist(output_dir, h5_name=h5_name, npy_name=npy_name)
        and grasp_fail_outputs_exist(
            output_dir,
            h5_name=fail_h5_name,
            npy_name=fail_npy_name,
        )
        and global_pc_exists(output_dir, render_subdir)
    )


def prepare_object_state(
    cfg: Dict,
    entry: Dict,
    output_dir: str,
    n_points: int,
) -> Dict:
    object_scale_key = str(entry["object_scale_key"])
    del object_scale_key
    points = np.load(str(entry["global_pc_abs"])).astype(np.float32, copy=False)
    normals = np.load(str(entry["global_normals_abs"])).astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"global_pc.npy must have shape (N, 3), got {points.shape}.")
    if normals.shape != points.shape:
        raise ValueError(
            f"global_normals.npy shape {normals.shape} does not match global_pc.npy {points.shape}."
        )
    if int(n_points) > 0 and int(n_points) < points.shape[0]:
        points = points[: int(n_points)]
        normals = normals[: int(n_points)]
    write_global_pc(points, output_dir, str(cfg["sampling"]["pc_subdir"]))
    write_global_normals(normals, output_dir, str(cfg["sampling"]["pc_subdir"]))
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(cfg["sampling"]["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )

    set_seed(
        stable_seed(int(cfg["seed"]), str(entry["object_scale_key"]), "sample_frames")
    )
    transforms_np = sample_frames_from_points(cfg, points, normals)
    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)
    return {
        "entry": entry,
        "output_dir": output_dir,
        "points": points,
        "normals": normals,
        "pts_for_sim": pts_for_sim.astype(np.float32, copy=False),
        "norms_for_sim": norms_for_sim.astype(np.float32, copy=False),
        "qpos_init": qpos_init.astype(np.float32, copy=False),
        "qpos_approach": qpos_approach.astype(np.float32, copy=False),
        "qpos_prepared": qpos_prepared.astype(np.float32, copy=False),
        "cursor": 0,
    }


def take_candidate_batch(
    object_states: List[Dict],
    writers: List[ObjectGraspWriter],
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_ids: List[int] = []
    object_ids: List[int] = []
    progress = True
    while len(candidate_ids) < max_rows and progress:
        progress = False
        for object_id, state in enumerate(object_states):
            if len(candidate_ids) >= max_rows:
                break
            if writers[object_id].done:
                continue
            cursor = int(state["cursor"])
            total = int(state["qpos_prepared"].shape[0])
            if cursor >= total:
                continue
            candidate_ids.append(cursor)
            object_ids.append(object_id)
            state["cursor"] = cursor + 1
            progress = True
    return (
        np.asarray(candidate_ids, dtype=np.int32),
        np.asarray(object_ids, dtype=np.int32),
    )


def build_bucket_backend(
    cfg: Dict,
    object_states: List[Dict],
    hand_xml_path: str,
    object_fixed: bool,
    args: argparse.Namespace,
) -> MjWarpBucketHO:
    object_infos = []
    for state in object_states:
        entry = state["entry"]
        object_infos.append(
            {
                "name": str(entry["object_name"]),
                "xml_abs": str(entry["mjcf_abs"]),
                "scale_tag": str(entry["scale_tag"]),
                "scale": entry.get("scale"),
                "object_scale_key": str(entry["object_scale_key"]),
            }
        )
    backend = MjWarpBucketHO(
        object_infos=object_infos,
        hand_xml_path=hand_xml_path,
        anchor_params=hand_anchor_params_cfg(cfg),
        hand_profile=hand_profile_cfg(cfg),
        object_profile=object_profile_cfg(cfg),
        root_stabilization=hand_root_stabilization_cfg(cfg),
        object_fixed=object_fixed,
        nworld=int(args.batch_size),
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
        njmax=int(args.njmax) if args.njmax is not None else None,
        ccd_iterations=int(args.ccd_iterations),
        slot_spacing=float(args.slot_spacing),
    )
    backend.set_slot_object_points(
        [state["pts_for_sim"] for state in object_states],
        [state["norms_for_sim"] for state in object_states],
    )
    return backend


def run_collision_and_grasp(
    cfg: Dict,
    object_states: List[Dict],
    writers: List[ObjectGraspWriter],
    bucket_fixed: MjWarpBucketHO,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    batch_size = int(args.batch_size)
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    sim_grasp_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    sim_grasp_cfg.pop("record_history", None)

    rows = {
        "object_id": [],
        "qpos_init": [],
        "qpos_approach": [],
        "qpos_prepared": [],
        "qpos_grasp": [],
    }
    no_col_rows = {
        "object_id": [],
        "qpos_init": [],
        "qpos_approach": [],
        "qpos_prepared": [],
    }
    object_count = len(object_states)
    collision_batches = 0
    sim_grasp_batches = 0
    no_col_count = 0
    contact_ok_count = 0
    per_object_no_col = np.zeros((object_count,), dtype=np.int64)
    per_object_contact_ok = np.zeros((object_count,), dtype=np.int64)

    pbar = tqdm(desc="bucket-collision", disable=not args.verbose, unit="cand")
    while not all(writer.done for writer in writers):
        candidate_ids, object_ids = take_candidate_batch(
            object_states, writers, max_rows=batch_size
        )
        if candidate_ids.size == 0:
            break
        valid_count = int(candidate_ids.size)
        world_ids = np.arange(valid_count, dtype=np.int32)
        slot_ids = object_ids.astype(np.int32, copy=False)

        qpos_prepared = np.stack(
            [
                object_states[int(obj_id)]["qpos_prepared"][int(candidate_id)]
                for obj_id, candidate_id in zip(object_ids, candidate_ids)
            ],
            axis=0,
        )
        qpos_approach = np.stack(
            [
                object_states[int(obj_id)]["qpos_approach"][int(candidate_id)]
                for obj_id, candidate_id in zip(object_ids, candidate_ids)
            ],
            axis=0,
        )
        qpos_init = np.stack(
            [
                object_states[int(obj_id)]["qpos_init"][int(candidate_id)]
                for obj_id, candidate_id in zip(object_ids, candidate_ids)
            ],
            axis=0,
        )

        bucket_fixed.reset_worlds(world_ids)
        bucket_fixed.load_hand_qpos_to_worlds(
            world_ids, qpos_prepared, slot_ids, do_forward=True
        )
        prepared_contact = bucket_fixed.read_contact_mask_for_worlds(
            world_ids, slot_ids
        )
        bucket_fixed.load_hand_qpos_to_worlds(
            world_ids, qpos_approach, slot_ids, do_forward=True
        )
        approach_contact = bucket_fixed.read_contact_mask_for_worlds(
            world_ids, slot_ids
        )
        bucket_fixed.load_hand_qpos_to_worlds(
            world_ids, qpos_init, slot_ids, do_forward=True
        )
        init_contact = bucket_fixed.read_contact_mask_for_worlds(world_ids, slot_ids)
        collision_batches += 1

        for row_idx in np.flatnonzero(prepared_contact):
            writers[int(object_ids[row_idx])].add_fail(
                qpos_prepared[int(row_idx)], "prepared_contact"
            )

        no_col_mask = ~(prepared_contact | approach_contact | init_contact)
        no_col_count += int(no_col_mask.sum())
        if np.any(no_col_mask):
            per_object_no_col += np.bincount(
                object_ids[no_col_mask],
                minlength=object_count,
            )
        for row_idx in np.flatnonzero(no_col_mask):
            no_col_rows["object_id"].append(int(object_ids[int(row_idx)]))
            no_col_rows["qpos_init"].append(qpos_init[int(row_idx)].copy())
            no_col_rows["qpos_approach"].append(qpos_approach[int(row_idx)].copy())
            no_col_rows["qpos_prepared"].append(qpos_prepared[int(row_idx)].copy())
        pbar.update(valid_count)
    pbar.close()

    if not no_col_rows["object_id"]:
        out = {}
        for key, value in rows.items():
            dtype = np.int32 if key == "object_id" else np.float32
            out[key] = np.asarray(value, dtype=dtype)
        out["stats"] = np.asarray(
            [collision_batches, sim_grasp_batches, no_col_count, contact_ok_count],
            dtype=np.int64,
        )
        out["per_object_no_col"] = per_object_no_col
        out["per_object_contact_ok"] = per_object_contact_ok
        return out

    no_col_object_ids = np.asarray(no_col_rows["object_id"], dtype=np.int32)
    no_col_init = np.asarray(no_col_rows["qpos_init"], dtype=np.float32)
    no_col_approach = np.asarray(no_col_rows["qpos_approach"], dtype=np.float32)
    no_col_prepared_all = np.asarray(no_col_rows["qpos_prepared"], dtype=np.float32)

    pbar = tqdm(
        total=int(no_col_prepared_all.shape[0]),
        desc="bucket-simgrasp",
        disable=not args.verbose,
        unit="cand",
    )
    for start in range(0, int(no_col_prepared_all.shape[0]), batch_size):
        end = min(start + batch_size, int(no_col_prepared_all.shape[0]))
        valid_count = end - start
        no_col_worlds = np.arange(valid_count, dtype=np.int32)
        no_col_slots = no_col_object_ids[start:end]
        no_col_prepared = no_col_prepared_all[start:end]
        bucket_fixed.reset_worlds(no_col_worlds)
        bucket_fixed.load_hand_qpos_to_worlds(
            no_col_worlds,
            no_col_prepared,
            no_col_slots,
            do_forward=True,
        )
        qpos_grasp, ho_counts = bucket_fixed.sim_grasp_for_worlds(
            no_col_worlds,
            no_col_slots,
            **sim_grasp_cfg,
        )
        sim_grasp_batches += 1

        contact_ok_mask = ho_counts >= contact_min_count
        contact_ok_count += int(contact_ok_mask.sum())
        if np.any(contact_ok_mask):
            per_object_contact_ok += np.bincount(
                no_col_slots[contact_ok_mask],
                minlength=object_count,
            )
        for local_idx in np.flatnonzero(~contact_ok_mask):
            src_idx = start + int(local_idx)
            writers[int(no_col_object_ids[src_idx])].add_fail(
                qpos_grasp[int(local_idx)], "insufficient_contact"
            )
        for local_idx in np.flatnonzero(contact_ok_mask):
            src_idx = start + int(local_idx)
            rows["object_id"].append(int(no_col_object_ids[src_idx]))
            rows["qpos_init"].append(no_col_init[src_idx].copy())
            rows["qpos_approach"].append(no_col_approach[src_idx].copy())
            rows["qpos_prepared"].append(no_col_prepared_all[src_idx].copy())
            rows["qpos_grasp"].append(qpos_grasp[int(local_idx)].copy())
        pbar.update(valid_count)
    pbar.close()

    out = {}
    for key, value in rows.items():
        dtype = np.int32 if key == "object_id" else np.float32
        out[key] = np.asarray(value, dtype=dtype)
    out["stats"] = np.asarray(
        [collision_batches, sim_grasp_batches, no_col_count, contact_ok_count],
        dtype=np.int64,
    )
    out["per_object_no_col"] = per_object_no_col
    out["per_object_contact_ok"] = per_object_contact_ok
    return out


def run_extforce(
    cfg: Dict,
    contact_rows: Dict[str, np.ndarray],
    writers: List[ObjectGraspWriter],
    bucket_free: MjWarpBucketHO,
    args: argparse.Namespace,
) -> Dict[str, int]:
    if contact_rows["object_id"].size == 0:
        return {"loops": 0, "settle_fail": 0, "force_fail": 0, "valid": 0}

    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_cfg.pop("visualize", None)
    qpos_grasp = contact_rows["qpos_grasp"]
    object_ids = contact_rows["object_id"].astype(np.int32, copy=False)
    qpos_squeeze = bucket_free.build_squeeze_qpos_batch(
        qpos_grasp,
        grip_delta=float(extforce_cfg["grip_delta"]),
    )
    qpos_prepared_valid = bucket_free.build_pregrasp_qpos_batch(
        qpos_squeeze,
        contact_rows["qpos_prepared"][:, 7:],
    )
    squeeze_ctrl = (
        bucket_free.qpos_to_ctrl_batch(qpos_squeeze)
        if bucket_free.nu > 0
        else np.zeros((qpos_squeeze.shape[0], 0), dtype=np.float32)
    )

    dt = float(bucket_free.mj_model.opt.timestep)
    n_steps = max(1, int(float(extforce_cfg["duration"]) / max(dt, 1e-8)))
    check_steps = max(int(extforce_cfg["check_steps"]), 1)
    n_chunks = max(1, int(np.ceil(n_steps / check_steps)))
    close_steps = max(int(extforce_cfg.get("close_steps", 100)), 1)
    settle_chunks = max(1, int(np.ceil(close_steps / check_steps)))
    trans_thresh = float(extforce_cfg["trans_thresh"])
    angle_thresh = float(extforce_cfg["angle_thresh"])
    force_mag = float(extforce_cfg["force_mag"])
    external_force_dirs = np.array(
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

    nworld = int(args.batch_size)
    world_active = np.zeros((nworld,), dtype=bool)
    world_candidate = np.full((nworld,), -1, dtype=np.int32)
    world_slot = np.full((nworld,), -1, dtype=np.int32)
    world_dir_idx = np.zeros((nworld,), dtype=np.int32)
    world_chunk_idx = np.zeros((nworld,), dtype=np.int32)
    world_initial_pose = np.zeros((nworld, 7), dtype=np.float32)
    next_candidate = 0
    loops = 0
    settle_fail = 0
    force_fail = 0
    valid_written = 0
    object_count = len(writers)
    per_object_settle_fail = np.zeros((object_count,), dtype=np.int64)
    per_object_force_fail = np.zeros((object_count,), dtype=np.int64)
    per_object_valid = np.zeros((object_count,), dtype=np.int64)

    def next_ids(max_count: int) -> np.ndarray:
        nonlocal next_candidate
        out: List[int] = []
        while next_candidate < object_ids.size and len(out) < max_count:
            cand = int(next_candidate)
            next_candidate += 1
            if writers[int(object_ids[cand])].done:
                continue
            out.append(cand)
        return np.asarray(out, dtype=np.int32)

    def load(
        world_ids: np.ndarray,
        candidate_ids: np.ndarray,
        reset_direction: bool = True,
    ) -> None:
        slots = object_ids[candidate_ids].astype(np.int32, copy=False)
        bucket_free.reset_worlds(world_ids)
        bucket_free.load_hand_qpos_to_worlds(
            world_ids,
            qpos_prepared_valid[candidate_ids],
            slots,
            ctrl_batch=squeeze_ctrl[candidate_ids],
            do_forward=True,
        )
        world_active[world_ids] = True
        world_candidate[world_ids] = candidate_ids
        world_slot[world_ids] = slots
        if reset_direction:
            world_dir_idx[world_ids] = 0
        world_chunk_idx[world_ids] = -settle_chunks
        world_initial_pose[world_ids] = bucket_free.get_obj_pose_for_worlds(
            world_ids,
            slots,
        )

    first_worlds = np.arange(nworld, dtype=np.int32)
    first_candidates = next_ids(nworld)
    if first_candidates.size > 0:
        load(first_worlds[: first_candidates.size], first_candidates)

    pbar = tqdm(desc="bucket-extforce", disable=not args.verbose, unit="loop")
    while np.any(world_active):
        active_worlds = np.flatnonzero(world_active).astype(np.int32)
        slots = world_slot[active_worlds]
        settling_mask = world_chunk_idx[active_worlds] < 0
        force_worlds = active_worlds[~settling_mask]
        failed_for_output = np.zeros((0,), dtype=np.int32)
        if force_worlds.size > 0:
            force_slots = world_slot[force_worlds]
            force_batch = np.zeros((force_worlds.size, 6), dtype=np.float32)
            force_batch[:, :3] = (
                external_force_dirs[world_dir_idx[force_worlds], :3] * force_mag
            )
            bucket_free.set_object_force_to_worlds(
                force_worlds,
                force_slots,
                force_batch,
            )
        bucket_free.step_batch(check_steps)
        loops += 1

        settling_worlds = active_worlds[settling_mask]
        done_worlds = np.zeros((0,), dtype=np.int32)
        if settling_worlds.size > 0:
            world_chunk_idx[settling_worlds] += 1
            close_done = settling_worlds[world_chunk_idx[settling_worlds] >= 0]
            if close_done.size > 0:
                close_slots = world_slot[close_done]
                has_contact = bucket_free.read_contact_mask_for_worlds(
                    close_done,
                    close_slots,
                )
                current_pose = bucket_free.get_obj_pose_for_worlds(
                    close_done,
                    close_slots,
                )
                pos_delta, angle_delta = bucket_free._pose_delta_batch(
                    world_initial_pose[close_done],
                    current_pose,
                )
                failed = (
                    (~has_contact)
                    | (pos_delta >= trans_thresh)
                    | (angle_delta >= angle_thresh)
                )
                failed_worlds = close_done[failed]
                settle_fail += int(failed_worlds.size)
                if failed_worlds.size > 0:
                    failed_candidates = world_candidate[failed_worlds]
                    per_object_settle_fail += np.bincount(
                        object_ids[failed_candidates],
                        minlength=object_count,
                    )
                failed_for_output = np.concatenate(
                    [failed_for_output, failed_worlds], axis=0
                )
                alive_worlds = close_done[~failed]
                if alive_worlds.size > 0:
                    world_initial_pose[alive_worlds] = (
                        bucket_free.get_obj_pose_for_worlds(
                            alive_worlds,
                            world_slot[alive_worlds],
                        )
                    )
                done_worlds = np.concatenate([done_worlds, failed_worlds], axis=0)

        eval_worlds = active_worlds[~settling_mask]
        if eval_worlds.size > 0:
            eval_slots = world_slot[eval_worlds]
            has_contact = bucket_free.read_contact_mask_for_worlds(
                eval_worlds,
                eval_slots,
            )
            current_pose = bucket_free.get_obj_pose_for_worlds(eval_worlds, eval_slots)
            pos_delta, angle_delta = bucket_free._pose_delta_batch(
                world_initial_pose[eval_worlds],
                current_pose,
            )
            failed = (
                (~has_contact)
                | (pos_delta >= trans_thresh)
                | (angle_delta >= angle_thresh)
            )
            failed_worlds = eval_worlds[failed]
            force_fail += int(failed_worlds.size)
            if failed_worlds.size > 0:
                failed_candidates = world_candidate[failed_worlds]
                per_object_force_fail += np.bincount(
                    object_ids[failed_candidates],
                    minlength=object_count,
                )
            failed_for_output = np.concatenate(
                [failed_for_output, failed_worlds], axis=0
            )
            alive_worlds = eval_worlds[~failed]
            complete_dir = alive_worlds[(world_chunk_idx[alive_worlds] + 1) >= n_chunks]
            running = alive_worlds[(world_chunk_idx[alive_worlds] + 1) < n_chunks]
            world_chunk_idx[running] += 1

            final_valid = complete_dir[world_dir_idx[complete_dir] >= 5]
            next_dir = complete_dir[world_dir_idx[complete_dir] < 5]
            if next_dir.size > 0:
                world_dir_idx[next_dir] += 1
                next_dir_candidates = world_candidate[next_dir]
                load(
                    next_dir,
                    next_dir_candidates,
                    reset_direction=False,
                )
            for world_id in final_valid:
                cand = int(world_candidate[int(world_id)])
                obj_id = int(object_ids[cand])
                added = writers[obj_id].add_valid(
                    contact_rows["qpos_init"][cand],
                    contact_rows["qpos_approach"][cand],
                    contact_rows["qpos_prepared"][cand],
                    contact_rows["qpos_grasp"][cand],
                    qpos_squeeze[cand],
                )
                valid_written += int(added)
                per_object_valid[obj_id] += int(added)
            done_worlds = np.concatenate(
                [done_worlds, failed_worlds, final_valid], axis=0
            )

        if done_worlds.size > 0:
            for world_id in failed_for_output:
                cand = int(world_candidate[int(world_id)])
                if cand < 0:
                    continue
                obj_id = int(object_ids[cand])
                if not writers[obj_id].done:
                    writers[obj_id].add_fail(qpos_squeeze[cand], "extforce_failure")

            world_active[done_worlds] = False
            world_candidate[done_worlds] = -1
            world_slot[done_worlds] = -1
            refill = next_ids(done_worlds.size)
            if refill.size > 0:
                load(done_worlds[: refill.size], refill)
            free_worlds = done_worlds[refill.size :]
            if free_worlds.size > 0:
                bucket_free.reset_worlds(free_worlds)

        if all(writer.done for writer in writers):
            break
        pbar.update(1)
    pbar.close()
    return {
        "loops": loops,
        "settle_fail": settle_fail,
        "force_fail": force_fail,
        "valid": valid_written,
        "per_object_settle_fail": per_object_settle_fail.tolist(),
        "per_object_force_fail": per_object_force_fail.tolist(),
        "per_object_valid": per_object_valid.tolist(),
    }


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args.config)
    if args.max_cap_override is not None:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["max_cap"] = int(args.max_cap_override)
    set_seed(int(cfg["seed"]))

    dataset = build_dataset(cfg, args.config)
    object_states = []
    selected_entries = []
    for key in args.object_scale_keys:
        entry = dataset.get_obj_info_by_scale_key(str(key))
        output_dir = entry_output_dir(entry, args.output_root)
        if (not args.force) and outputs_complete(output_dir, cfg):
            print(f"[bucket] skip existing outputs for {key}: {output_dir}")
            continue
        selected_entries.append(entry)
        n_points = (
            int(args.n_points_override)
            if args.n_points_override is not None
            else int(cfg["sampling"]["n_points"])
        )
        object_states.append(
            prepare_object_state(
                cfg=cfg,
                entry=entry,
                output_dir=output_dir,
                n_points=n_points,
            )
        )
    if not object_states:
        print("[bucket] no entries to run.")
        return

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = Path(hand_xml_path).stem
    print(
        f"[bucket] objects={len(object_states)} nworld={args.batch_size} "
        f"max_cap={cfg['data']['max_cap']}"
    )
    ts = time.perf_counter()
    bucket_fixed = build_bucket_backend(
        cfg,
        object_states,
        hand_xml_path,
        object_fixed=True,
        args=args,
    )
    bucket_free = build_bucket_backend(
        cfg,
        object_states,
        hand_xml_path,
        object_fixed=False,
        args=args,
    )
    writers = []
    for state in object_states:
        entry = state["entry"]
        writers.append(
            ObjectGraspWriter(
                output_dir=state["output_dir"],
                object_name=str(entry["object_name"]),
                scale=entry.get("scale"),
                hand_name=hand_name,
                qpos_dim=bucket_fixed.nq_hand,
                max_cap=int(cfg["data"]["max_cap"]),
                fail_keep_ratio=float(cfg["data"]["fail_keep_ratio"]),
                seed=int(cfg["seed"]),
                h5_name=str(cfg["data"]["h5_name"]),
                npy_name=str(cfg["data"]["npy_name"]),
                fail_h5_name=str(cfg["data"]["fail_h5_name"]),
                fail_npy_name=str(cfg["data"]["fail_npy_name"]),
                flush_every=int(cfg["data"].get("flush_every", 0) or 0),
            )
        )

    contact_rows = run_collision_and_grasp(
        cfg,
        object_states,
        writers,
        bucket_fixed,
        args,
    )
    ext_stats = run_extforce(cfg, contact_rows, writers, bucket_free, args)
    for writer in writers:
        writer.close()
        print(
            f"[bucket] {writer.object_name} scale={writer.scale} "
            f"valid={writer.valid_count} fail_collected={len(writer.fail_qpos_rows)} "
            f"out={writer.h5_path}"
        )
    for obj_id, state in enumerate(object_states):
        entry = state["entry"]
        object_name = str(entry["object_name"])
        print(
            f"[bucket] per_object {object_name} "
            f"no_col={int(contact_rows['per_object_no_col'][obj_id])} "
            f"contact_ok={int(contact_rows['per_object_contact_ok'][obj_id])} "
            f"ext_settle_fail={int(ext_stats['per_object_settle_fail'][obj_id])} "
            f"ext_force_fail={int(ext_stats['per_object_force_fail'][obj_id])} "
            f"ext_valid={int(ext_stats['per_object_valid'][obj_id])}"
        )
    collision_batches, sim_grasp_batches, no_col_count, contact_ok_count = contact_rows[
        "stats"
    ].tolist()
    print(
        f"[bucket] stats collision_batches={collision_batches} "
        f"sim_grasp_batches={sim_grasp_batches} no_col={no_col_count} "
        f"contact_ok={contact_ok_count} extforce={ext_stats} "
        f"total={time.perf_counter() - ts:.3f}s"
    )


if __name__ == "__main__":
    main()
