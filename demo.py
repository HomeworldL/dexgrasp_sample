import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.mj_ho import MjHO
from src.sample import downsample_fps, sample_grasp_frames
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_pointcloud import sample_surface_o3d


def set_seed(random_seed: int):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def object_name_from_scale_key(object_scale_key: str) -> str:
    if "__" in object_scale_key:
        return object_scale_key.split("__", 1)[0]
    return object_scale_key


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
) -> str:
    object_name = object_name_from_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    target_body_params = cfg["hand"]["target_body_params"]

    mjho = MjHO(obj_info, hand_xml_path, target_body_params=target_body_params, visualize=True)
    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    ts = time.time()
    transforms_np = sample_frames_from_points(cfg, points, normals)
    if verbose:
        print(f"[{object_scale_key}] frame sampling time: {time.time() - ts:.3f}s, N={len(transforms_np)}")

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    max_cap = int(cfg["output"]["max_cap"])
    contact_min_count = int(cfg["validation"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    sim_grasp_cfg.pop("visualize", None)
    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    ts = time.time()
    valid_qpos_grasp = []

    for i in tqdm(
        range(qpos_prepared.shape[0]),
        desc=f"sampling-{object_scale_key}",
        miniters=50,
        disable=not verbose,
    ):
        if num_valid >= max_cap:
            num_samples = i
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
        qpos_grasp, _ = mjho.sim_grasp(visualize=True, **sim_grasp_cfg)
        ho_contact, _ = mjho.get_contact_info(obj_margin=0.00)

        if len(ho_contact) >= contact_min_count:
            valid_qpos_grasp.append(qpos_grasp.astype(np.float32, copy=False))
            num_valid += 1

    duration = time.time() - ts
    if verbose:
        print(
            f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
            f"valid={num_valid} time={duration:.2f}s"
        )
        print(
            f"[{object_scale_key}] demo_only object_id={object_id} scale={scale} "
            f"hand={hand_name} grasps_in_memory={len(valid_qpos_grasp)} output_dir={Path(output_dir_abs)}"
        )
    return object_scale_key


def main():
    p = argparse.ArgumentParser(description="Sample grasps for one object-scale entry.")
    p.add_argument("--object-scale-key", type=str, required=True, help="Unique object-scale key.")
    p.add_argument("--coacd-path", type=str, required=True, help="Path to scaled COACD mesh OBJ.")
    p.add_argument("--mjcf-path", type=str, required=True, help="Path to scaled object MJCF.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for grasp artifacts.")
    p.add_argument("--scale", type=float, default=None, help="Object scale metadata for grasp.h5.")
    p.add_argument("--object-id", type=str, default=None, help="Object id metadata for grasp.h5.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")

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
    )


if __name__ == "__main__":
    main()
