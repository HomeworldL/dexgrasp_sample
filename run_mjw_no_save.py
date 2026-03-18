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


def resolve_object_info(ds: DatasetObjects, obj_id: Optional[int], obj_key: Optional[str], cfg: Dict) -> Dict:
    if obj_key:
        return ds.get_obj_info_by_scale_key(obj_key)
    resolved_id = int(obj_id) if obj_id is not None else int(cfg.get("object", {}).get("id", 0))
    return ds.get_obj_info_by_index(resolved_id)


def take_full_batch(queue: list[np.ndarray], batch_size: int) -> np.ndarray:
    batch = np.asarray(queue[:batch_size], dtype=np.float32)
    del queue[:batch_size]
    return batch


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
    info = resolve_object_info(ds, args.obj_id, args.obj_key, cfg)

    pts, norms = ds.sample_surface_o3d(
        info["coacd_abs"],
        n_points=min(int(args.n_points), int(cfg.get("sampling", {}).get("n_points", args.n_points))),
        method="poisson",
    )
    transforms_np = sample_frames_from_points(cfg, pts, norms)
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
    extforce_cfg.pop("visualize", None)
    contact_min_count = int(cfg["validation"]["contact_min_count"])

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
        ccd_iterations=int(args.ccd_iterations),
    )
    mjw_valid = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=cfg["hand"].get("target_body_params"),
        object_fixed=False,
        nworld=batch_size,
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
        ccd_iterations=int(args.ccd_iterations),
    )
    backend_init_time = time.perf_counter() - ts_backend

    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        pts,
        norms,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjw_grasp._set_obj_pts_norms(pts_for_sim, norms_for_sim)
    mjw_valid._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    num_no_col = 0
    num_grasp_contact_ok = 0
    num_valid = 0
    collision_batches = 0
    sim_grasp_batches = 0
    extforce_batches = 0
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
    processed_no_col = (no_col_count // batch_size) * batch_size
    dropped_after_collision = no_col_count - processed_no_col

    grasp_results = []
    grasp_contact_counts = []
    if processed_no_col > 0:
        pbar = tqdm(
            total=processed_no_col,
            desc=f"simgrasp-{info['object_scale_key']}",
            miniters=1,
        )
        for start in range(0, processed_no_col, batch_size):
            end = start + batch_size
            sim_grasp_batches += 1
            grasp_input = qpos_prepared_no_col[start:end]
            ts = time.perf_counter()
            grasp_result = mjw_grasp.sim_grasp_batch(grasp_input, valid_count=batch_size, **sim_grasp_cfg)
            sim_grasp_time += time.perf_counter() - ts
            grasp_results.append(grasp_result.qpos_grasp.copy())
            grasp_contact_counts.append(grasp_result.ho_contact_counts.copy())
            pbar.update(batch_size)
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
    processed_contact_ok = (contact_ok_count // batch_size) * batch_size
    dropped_after_grasp = contact_ok_count - processed_contact_ok

    if processed_contact_ok > 0:
        pbar = tqdm(
            total=processed_contact_ok,
            desc=f"extforce-{info['object_scale_key']}",
            miniters=1,
        )
        for start in range(0, processed_contact_ok, batch_size):
            end = start + batch_size
            extforce_batches += 1
            extforce_input = qpos_grasp_contact_ok[start:end]
            ts = time.perf_counter()
            extforce_result = mjw_valid.sim_under_extforce_batch(
                extforce_input,
                valid_count=batch_size,
                **extforce_cfg,
            )
            extforce_time += time.perf_counter() - ts
            num_valid += int(extforce_result.is_valid.sum())
            if num_valid >= 100:
                break
            pbar.update(batch_size)
        pbar.close()

    total_time = backend_init_time + collision_time + sim_grasp_time + extforce_time

    print(
        f"[run_mjw_no_save] object={info['object_name']} scale={info['scale']:.3f} "
        f"candidates={max_candidates} no_col={num_no_col} "
        f"contact_ok={num_grasp_contact_ok} valid={num_valid}"
    )
    print(
        f"[run_mjw_no_save] batches collision={collision_batches} "
        f"sim_grasp={sim_grasp_batches} extforce={extforce_batches} "
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
