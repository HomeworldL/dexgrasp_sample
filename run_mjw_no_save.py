import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.dataset_objects import DatasetObjects
from src.mjw_ho import MjWarpHO
from src.sample import downsample_fps, sample_grasp_frames
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config
from utils.utils_seed import set_seed


def compose_rot_grasp_to_palm(cfg: Dict) -> np.ndarray:
    transform_cfg = cfg["hand"]["transform"]
    base = np.asarray(transform_cfg["base_rot_grasp_to_palm"], dtype=float)
    extra = transform_cfg["extra_euler"]
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


def make_qpos_triplets(cfg: Dict, pose: np.ndarray):
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


def resolve_object_info(ds: DatasetObjects, obj_id: Optional[int], obj_key: Optional[str]) -> Dict:
    if obj_key:
        return ds.get_obj_info_by_scale_key(obj_key)
    if obj_id is not None:
        return ds.get_obj_info_by_index(int(obj_id))
    raise ValueError("run_mjw_no_save requires either --obj-id or --obj-key.")


def build_valid_backend(
    obj_info: Dict,
    hand_xml_path: str,
    cfg: Dict,
    args: argparse.Namespace,
    pts_for_sim: np.ndarray,
    norms_for_sim: np.ndarray,
    nworld: int,
) -> MjWarpHO:
    backend = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=cfg["hand"].get("target_body_params"),
        object_fixed=False,
        nworld=int(nworld),
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
        njmax=int(args.njmax) if args.njmax is not None else None,
        ccd_iterations=int(args.ccd_iterations),
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
    cfg: Dict,
    args: argparse.Namespace,
    pts_for_sim: np.ndarray,
    norms_for_sim: np.ndarray,
) -> tuple[MjWarpHO, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    new_pool_size = int(active_world_ids.size)
    next_backend = build_valid_backend(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        cfg=cfg,
        args=args,
        pts_for_sim=pts_for_sim,
        norms_for_sim=norms_for_sim,
        nworld=new_pool_size,
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

    next_world_candidate_ids = world_candidate_ids[active_world_ids].copy()
    next_world_dir_idx = world_dir_idx[active_world_ids].copy()
    next_world_chunk_idx = world_chunk_idx[active_world_ids].copy()
    next_world_initial_pose = world_initial_pose[active_world_ids].copy()
    next_world_pos_delta = world_pos_delta[active_world_ids].copy()
    next_world_angle_delta = world_angle_delta[active_world_ids].copy()
    return (
        next_backend,
        next_world_candidate_ids,
        next_world_dir_idx,
        next_world_chunk_idx,
        next_world_initial_pose,
        next_world_pos_delta,
        next_world_angle_delta,
        new_pool_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MJWarp end-to-end grasp validation without saving outputs.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--naconmax", type=int, default=512)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--ccd-iterations", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    ds = DatasetObjects(
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag_from_config(args.config),
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )
    info = resolve_object_info(ds, args.obj_id, args.obj_key)

    pts, norms = ds.sample_surface_o3d(
        info["coacd_abs"],
        n_points=min(int(args.n_points), int(cfg.get("sampling", {}).get("n_points", args.n_points))),
        method="poisson",
    )
    # print(f"Sampled {pts.shape[0]} points and normals from object surface.")
    transforms_np = sample_frames_from_points(cfg, pts, norms)
    # print(f"Sampled {transforms_np.shape[0]} grasp candidate frames from points.")
    # exit(0)  # --- IGNORE ---
    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)
    max_candidates = min(int(args.max_candidates), int(qpos_prepared.shape[0]))
    if max_candidates <= 0:
        raise RuntimeError("No candidate qpos generated for run_mjw_no_save.")
    qpos_prepared = qpos_prepared[:max_candidates]
    qpos_approach = qpos_approach[:max_candidates]
    qpos_init = qpos_init[:max_candidates]

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    obj_info = {"name": info["object_name"], "xml_abs": info["mjcf_abs"], "scale": float(info["scale"])}
    batch_size = int(args.batch_size)
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))
    sim_grasp_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    extforce_cfg.pop("visualize", None)
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    target_valid_cap = int(cfg.get("output", {}).get("max_cap", 100))
    if target_valid_cap <= 0:
        raise ValueError(f"output.max_cap must be positive, got {target_valid_cap}.")

    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        pts,
        norms,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )

    ts_backend = time.perf_counter()
    mjw_grasp = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=cfg["hand"].get("target_body_params"),
        object_fixed=True,
        nworld=batch_size,
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
        njmax=int(args.njmax) if args.njmax is not None else None,
        ccd_iterations=int(args.ccd_iterations),
    )
    mjw_valid = build_valid_backend(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        cfg=cfg,
        args=args,
        pts_for_sim=pts_for_sim,
        norms_for_sim=norms_for_sim,
        nworld=batch_size,
    )
    backend_init_time = time.perf_counter() - ts_backend

    mjw_grasp._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    num_no_col = 0
    num_grasp_contact_ok = 0
    num_valid = 0
    collision_batches = 0
    sim_grasp_batches = 0
    extforce_batches = 0
    extforce_shrinks = 0
    collision_time = 0.0
    sim_grasp_time = 0.0
    extforce_time = 0.0

    collision_mask = np.zeros((max_candidates,), dtype=bool)
    pbar = tqdm(total=max_candidates, desc=f"collision-{info['object_scale_key']}", miniters=1)
    for start in range(0, max_candidates, batch_size):
        end = min(start + batch_size, max_candidates)
        valid_count = end - start
        batch_prepared = qpos_prepared[start:end]
        batch_approach = qpos_approach[start:end]
        batch_init = qpos_init[start:end]

        ts = time.perf_counter()
        prepared_result = mjw_grasp.check_contact_batch(batch_prepared, valid_count=valid_count)
        approach_result = mjw_grasp.check_contact_batch(batch_approach, valid_count=valid_count)
        init_result = mjw_grasp.check_contact_batch(batch_init, valid_count=valid_count)
        collision_time += time.perf_counter() - ts
        collision_batches += 1

        no_col_mask = ~(prepared_result.has_contact | approach_result.has_contact | init_result.has_contact)
        collision_mask[start:end] = no_col_mask
        num_no_col += int(no_col_mask.sum())

        pbar.update(valid_count)
    pbar.close()

    qpos_prepared_no_col = qpos_prepared[collision_mask]
    no_col_count = int(qpos_prepared_no_col.shape[0])
    dropped_after_collision = 0

    grasp_results = []
    grasp_contact_counts = []
    if no_col_count > 0:
        pbar = tqdm(
            total=no_col_count,
            desc=f"simgrasp-{info['object_scale_key']}",
            miniters=1,
        )
        for start in range(0, no_col_count, batch_size):
            end = min(start + batch_size, no_col_count)
            valid_count = end - start
            sim_grasp_batches += 1
            grasp_input = qpos_prepared_no_col[start:end]
            ts = time.perf_counter()
            grasp_result = mjw_grasp.sim_grasp_batch(
                grasp_input,
                valid_count=valid_count,
                **sim_grasp_cfg,
            )
            sim_grasp_time += time.perf_counter() - ts
            grasp_results.append(grasp_result.qpos_grasp[:valid_count].copy())
            grasp_contact_counts.append(grasp_result.ho_contact_counts[:valid_count].copy())
            pbar.update(valid_count)
        pbar.close()

    if grasp_results:
        qpos_grasp_all = np.concatenate(grasp_results, axis=0)
        ho_contact_counts_all = np.concatenate(grasp_contact_counts, axis=0)
    else:
        qpos_grasp_all = np.zeros((0, qpos_prepared.shape[1]), dtype=np.float32)
        ho_contact_counts_all = np.zeros((0,), dtype=np.int32)

    contact_ok_mask = ho_contact_counts_all >= contact_min_count
    num_grasp_contact_ok = int(contact_ok_mask.sum())
    qpos_grasp_contact_ok = qpos_grasp_all[contact_ok_mask]

    contact_ok_count = int(qpos_grasp_contact_ok.shape[0])
    dropped_after_grasp = 0

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
            desc=f"extforce-{info['object_scale_key']}",
            miniters=1,
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
                    cfg=cfg,
                    args=args,
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

            failed_mask = (~has_contact) | (dir_pos_delta >= trans_thresh) | (dir_angle_delta >= angle_thresh)
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

            done_worlds = np.concatenate([failed_worlds, final_valid_worlds], axis=0)
            reached_cap = False
            if final_valid_worlds.size > 0:
                num_valid += int(final_valid_worlds.size)
                reached_cap = num_valid >= target_valid_cap

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
                            next_candidate, next_candidate + refill_count, dtype=np.int32
                        )
                        load_candidates_into_worlds(refill_worlds, refill_candidate_ids, reset_metrics=True)
                        next_candidate += refill_count

                free_worlds = done_worlds[refill_count:]
                if free_worlds.size > 0:
                    mjw_valid.reset_worlds(free_worlds)

            if reached_cap:
                break
        pbar.close()

    total_time = backend_init_time + collision_time + sim_grasp_time + extforce_time

    print(
        f"[run_mjw_no_save] object={info['object_name']} scale={info['scale']:.3f} "
        f"candidates={max_candidates} no_col={num_no_col} "
        f"contact_ok={num_grasp_contact_ok} valid={num_valid} target_valid_cap={target_valid_cap}"
    )
    print(
        f"[run_mjw_no_save] batches collision={collision_batches} "
        f"sim_grasp={sim_grasp_batches} extforce_loops={extforce_batches} "
        f"extforce_shrinks={extforce_shrinks} "
        f"dropped_after_collision={dropped_after_collision} "
        f"dropped_after_grasp={dropped_after_grasp}"
    )
    print(
        f"[run_mjw_no_save] time backend_init={backend_init_time:.3f}s "
        f"collision={collision_time:.3f}s sim_grasp={sim_grasp_time:.3f}s "
        f"extforce={extforce_time:.3f}s total={total_time:.3f}s"
    )


if __name__ == "__main__":
    main()
