import argparse
import gc
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.dataset_objects import DatasetObjects, resolve_dataset_root
from src.mj_ho import MjHO
from src.sample import downsample_fps, sample_grasp_frames
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config


def set_seed(random_seed: int):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compose_rot_grasp_to_palm(cfg: Dict) -> np.ndarray:
    base = np.asarray(cfg["transform"]["base_rot_grasp_to_palm"], dtype=float)
    extra = cfg["transform"]["extra_euler"]
    extra_rot = R.from_euler(extra["axis"], float(extra["degrees"]), degrees=True).as_matrix()
    return (base @ extra_rot).T


def resolve_object_name(ds: DatasetObjects, cfg: Dict, obj_id_arg: int | None, obj_arg: str | None) -> str:
    if obj_arg is not None:
        return obj_arg
    if obj_id_arg is not None:
        return ds.id2name[int(obj_id_arg)]
    return ds.id2name[int(cfg.get("object", {}).get("id", 0))]


def sample_frames_from_points(cfg: Dict, pts: np.ndarray, norms: np.ndarray) -> np.ndarray:
    sampling_cfg = cfg["sampling"]
    transforms = sample_grasp_frames(
        pts,
        norms,
        Nd=int(sampling_cfg["Nd"]),
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


def run_scale_sampling(
    cfg: Dict,
    object_name: str,
    scale_key: str,
    scale_asset: Dict,
    hand_xml_path: str,
    points: np.ndarray,
    normals: np.ndarray,
) -> str:
    obj_info = {"name": object_name, "xml_abs": scale_asset["xml_abs"], "scale": 1.0}
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
    print(f"[{scale_key}] frame sampling time: {time.time() - ts:.3f}s, N={len(transforms_np)}")

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    out_dir = Path(scale_asset["xml_abs"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "grasp.h5"

    d = qpos_prepared.shape[1]
    max_cap = int(cfg["output"]["max_cap"])
    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    ts = time.time()

    with h5py.File(h5_path, "w") as hf:
        ds_init = hf.create_dataset("qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_approach = hf.create_dataset("qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype="f4")
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype="f4")

        for i in tqdm(range(qpos_prepared.shape[0]), desc=f"sampling-{scale_key}", miniters=50):
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
            qpos_grasp, _ = mjho.sim_grasp(visualize=False)
            ho_contact, _ = mjho.get_contact_info(obj_margin=0.00)

            if len(ho_contact) >= int(cfg["validation"]["contact_min_count"]):
                is_valid, _, _ = mjho_valid.sim_under_extforce(qpos_grasp.copy(), visualize=False)
                if is_valid:
                    ds_init[num_valid] = qpos_init[i].astype("f4")
                    ds_approach[num_valid] = qpos_approach[i].astype("f4")
                    ds_prepared[num_valid] = qpos_prepared[i].astype("f4")
                    ds_grasp[num_valid] = qpos_grasp.astype("f4")
                    num_valid += 1

            if (i + 1) % 500 == 0:
                hf.flush()
                gc.collect()

        final_size = num_valid
        ds_init.resize((final_size, d))
        ds_approach.resize((final_size, d))
        ds_prepared.resize((final_size, d))
        ds_grasp.resize((final_size, d))
        hf.flush()

    duration = time.time() - ts
    print(f"[{scale_key}] samples={num_samples} no_col={num_no_col} valid={num_valid} time={duration:.2f}s out={h5_path}")
    return str(h5_path)


def main():
    p = argparse.ArgumentParser(description="Sample grasps for one object across fixed scales.")
    p.add_argument("-i", "--obj-id", type=int, default=None, help="Global object id in merged DatasetObjects.")
    p.add_argument("-o", "--obj", type=str, default=None, help="Object name override (optional).")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    config_stem = Path(args.config).stem
    ds = DatasetObjects(
        resolve_dataset_root(cfg["dataset"].get("root")),
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=config_stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        prebuild_scales=True,
        object_mass_kg=float(cfg["dataset"]["object_mass_kg"]),
    )

    obj_name = resolve_object_name(ds, cfg, args.obj_id, args.obj)
    info = ds.get_info(obj_name)
    print(f"Using object id={info['global_id']} name={info['object_name']}")

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    n_points = int(cfg["sampling"]["n_points"])

    for scale_key, scale_asset in sorted(info.get("scale_assets", {}).items()):
        pts, norms = ds.sample_surface_o3d(scale_asset["coacd_abs"], n_points=n_points, method="poisson")
        run_scale_sampling(cfg, obj_name, scale_key, scale_asset, hand_xml_path, pts, norms)


if __name__ == "__main__":
    main()
