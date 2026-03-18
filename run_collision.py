import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.dataset_objects import DatasetObjects
from src.mjw_ho import MjWarpHO
from src.sample import sample_grasp_frames
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MJWarp batched collision filtering on one object-scale."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--naconmax", type=int, default=512)
    parser.add_argument("-v", "--verbose", action="store_true")
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
    print(
        f"[run_collision] object={info['object_name']} global_id={info['global_id']} "
        f"scale={info['scale']:.3f}"
    )

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
        raise RuntimeError("No candidate qpos generated for collision validation.")

    qpos_prepared = qpos_prepared[:max_candidates]
    qpos_approach = qpos_approach[:max_candidates]
    qpos_init = qpos_init[:max_candidates]

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    obj_info = {"name": info["object_name"], "xml_abs": info["mjcf_abs"], "scale": float(info["scale"])}
    ts_backend = time.perf_counter()
    mjw_ho = MjWarpHO(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        target_body_params=cfg["hand"].get("target_body_params"),
        object_fixed=True,
        nworld=int(args.batch_size),
        device=str(args.device),
        nconmax=int(args.nconmax),
        naconmax=int(args.naconmax),
    )
    backend_init_time = time.perf_counter() - ts_backend
    print(
        f"[run_collision] mjwarp_backend_init={backend_init_time:.3f}s "
        f"(includes model upload and first forward compile when cache is cold)"
    )

    warm_count = min(int(args.batch_size), max_candidates)
    warmup_time = mjw_ho.warmup(qpos_prepared[:warm_count], valid_count=warm_count)
    print(
        f"[run_collision] warmup candidates={warm_count} time={warmup_time:.3f}s "
        f"(steady-state probe after backend init)"
    )

    prepared_hits = 0
    approach_hits = 0
    init_hits = 0
    fully_clear = 0
    steady_stage_time = 0.0

    for start in range(0, max_candidates, int(args.batch_size)):
        end = min(start + int(args.batch_size), max_candidates)
        valid_count = end - start

        batch_prepared = qpos_prepared[start:end]
        batch_approach = qpos_approach[start:end]
        batch_init = qpos_init[start:end]

        ts = time.perf_counter()
        prepared_result = mjw_ho.check_contact_batch(batch_prepared, valid_count=valid_count)
        t_prepared = time.perf_counter() - ts

        ts = time.perf_counter()
        approach_result = mjw_ho.check_contact_batch(batch_approach, valid_count=valid_count)
        t_approach = time.perf_counter() - ts

        ts = time.perf_counter()
        init_result = mjw_ho.check_contact_batch(batch_init, valid_count=valid_count)
        t_init = time.perf_counter() - ts

        prepared_hits += int(prepared_result.has_contact.sum())
        approach_hits += int(approach_result.has_contact.sum())
        init_hits += int(init_result.has_contact.sum())
        stage_clear = ~(prepared_result.has_contact | approach_result.has_contact | init_result.has_contact)
        fully_clear += int(stage_clear.sum())
        steady_stage_time += t_prepared + t_approach + t_init

        print(
            f"[run_collision] batch={start // int(args.batch_size):02d} "
            f"valid={valid_count} "
            f"prepared_hit={int(prepared_result.has_contact.sum())} "
            f"approach_hit={int(approach_result.has_contact.sum())} "
            f"init_hit={int(init_result.has_contact.sum())} "
            f"clear={int(stage_clear.sum())} "
            f"time=({t_prepared:.4f}s,{t_approach:.4f}s,{t_init:.4f}s)"
        )

        if args.verbose:
            print(
                f"[run_collision] prepared_contacts={prepared_result.active_contact_count} "
                f"approach_contacts={approach_result.active_contact_count} "
                f"init_contacts={init_result.active_contact_count}"
            )

    avg_candidate_time_ms = (steady_stage_time / max_candidates) * 1000.0
    print(
        f"[run_collision] summary candidates={max_candidates} "
        f"prepared_hit={prepared_hits} approach_hit={approach_hits} init_hit={init_hits} "
        f"clear={fully_clear} steady_total={steady_stage_time:.4f}s "
        f"avg_per_candidate={avg_candidate_time_ms:.3f}ms"
    )


if __name__ == "__main__":
    main()
