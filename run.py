import argparse
import gc
import os
import random
import time
from typing import Dict

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


def main():
    p = argparse.ArgumentParser(description="Sample grasps for a configured object/dataset/hand task.")
    p.add_argument("-i", "--obj-id", type=int, default=None, help="Global object id in merged DatasetObjects.")
    p.add_argument("-o", "--obj", type=str, default=None, help="Object name override (optional).")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    ds_root = resolve_dataset_root(cfg["dataset"].get("root"))
    ds = DatasetObjects(
        ds_root,
        dataset_names=list(cfg["dataset"].get("include", [])),
        shapenet_scale_range=tuple(cfg["dataset"].get("shapenet_scale_range", [0.06, 0.15])),
        shapenet_scale_seed=int(cfg["seed"]),
    )

    if args.obj is not None:
        obj_name = args.obj
    elif args.obj_id is not None:
        obj_name = ds.id2name[int(args.obj_id)]
    else:
        obj_id_default = int(cfg.get("object", {}).get("id", 0))
        obj_name = ds.id2name[obj_id_default]

    obj_info = ds.get_info(obj_name)
    print(
        f"Using object id={obj_info['global_id']} name={obj_name} dataset={obj_info['dataset']} scale={obj_info['scale']:.4f}"
    )

    # Mesh loading is explicit so config can swap mesh_type without touching pipeline logic.
    _ = ds.get_mesh(obj_name, "inertia")
    _ = ds.get_mesh(obj_name, "coacd")
    _ = ds.get_mesh(obj_name, "convex_parts")

    sampling_cfg = cfg["sampling"]
    pcs_sample_poisson, norms_sample_poisson = ds.get_point_cloud(
        obj_name,
        n_points=int(sampling_cfg["n_points"]),
        method="poisson",
    )

    ts = time.time()
    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
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

    print("Time to sample:", time.time() - ts)
    print("Shape of transforms:", transforms_np.shape)

    rot_grasp_to_palm = compose_rot_grasp_to_palm(cfg)
    rotation_matrices = transforms_np[:, :3, :3] @ rot_grasp_to_palm
    positions = transforms_np[:, :3, 3]
    quaternions = R.from_matrix(rotation_matrices).as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)  # xyzw -> wxyz
    pose = np.concatenate([positions, quaternions], axis=1).astype(np.float32)

    xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = os.path.basename(xml_path).split(".")[0]
    print(f"hand_name: {hand_name}")

    target_body_params = cfg["hand"]["target_body_params"]

    mjho = MjHO(obj_info, xml_path, target_body_params=target_body_params)
    pcs_for_sim, norms_for_sim, _ = downsample_fps(
        pcs_sample_poisson,
        norms_sample_poisson,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjho._set_obj_pts_norms(pcs_for_sim, norms_for_sim)

    mjho_valid = MjHO(obj_info, xml_path, target_body_params=target_body_params, object_fixed=False)

    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float32)
    approach_joints = np.asarray(cfg["hand"]["approach_joints"], dtype=np.float32)

    q_expanded = np.tile(prepared_joints, (pose.shape[0], 1)).astype(np.float32)
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1).astype(np.float32)

    N = qpos_prepared_sample.shape[0]
    qpos_approach_sample = qpos_prepared_sample.copy()
    qpos_approach_sample[:, 7:] = np.tile(approach_joints, (N, 1))

    shift_local = np.asarray(cfg["hand"]["shift_local"], dtype=float)
    positions = qpos_prepared_sample[:, :3]
    quats_wxyz = qpos_prepared_sample[:, 3:7]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    offset_world = R.from_quat(quats_xyzw).apply(shift_local)
    qpos_init_sample = qpos_prepared_sample.copy()
    qpos_init_sample[:, :3] = positions + offset_world
    qpos_init_sample[:, 7:] = np.tile(approach_joints, (N, 1))

    print(f"Shape of qpos_prepared: {qpos_prepared_sample.shape}")
    print(f"Shape of qpos_approach: {qpos_approach_sample.shape}")
    print(f"Shape of qpos_init: {qpos_init_sample.shape}")

    out_cfg = cfg["output"]
    save_dir = os.path.join(out_cfg["base_dir"], hand_name, obj_name)
    os.makedirs(save_dir, exist_ok=True)
    h5path = os.path.join(save_dir, out_cfg["h5_name"])
    npypath = os.path.join(save_dir, out_cfg["npy_name"])

    print(f"Saving to {h5path} ...")
    D = qpos_prepared_sample.shape[1]
    MAX_CAP = int(out_cfg["max_cap"])

    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    ts = time.time()

    with h5py.File(h5path, "w") as hf:
        ds_init = hf.create_dataset("qpos_init", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4")
        ds_approach = hf.create_dataset("qpos_approach", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4")
        ds_prepared = hf.create_dataset("qpos_prepared", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4")
        ds_grasp = hf.create_dataset("qpos_grasp", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4")

        for i in tqdm(range(qpos_prepared_sample.shape[0]), desc="sampling", miniters=50):
            if num_valid >= MAX_CAP:
                num_samples = i
                break

            mjho.set_hand_qpos(qpos_prepared_sample[i])
            if mjho.is_contact():
                continue

            mjho.set_hand_qpos(qpos_approach_sample[i])
            if mjho.is_contact():
                continue

            mjho.set_hand_qpos(qpos_init_sample[i])
            if mjho.is_contact():
                continue

            num_no_col += 1

            mjho.set_hand_qpos(qpos_prepared_sample[i])
            qpos_grasp, _ = mjho.sim_grasp(visualize=False)
            ho_contact, _ = mjho.get_contact_info(obj_margin=0.00)

            if len(ho_contact) >= int(cfg["validation"]["contact_min_count"]):
                is_valid, _, _ = mjho_valid.sim_under_extforce(qpos_grasp.copy(), visualize=False)
                if is_valid:
                    ds_init[num_valid] = qpos_init_sample[i].astype("f4")
                    ds_approach[num_valid] = qpos_approach_sample[i].astype("f4")
                    ds_prepared[num_valid] = qpos_prepared_sample[i].astype("f4")
                    ds_grasp[num_valid] = qpos_grasp.astype("f4")
                    num_valid += 1

            if (i + 1) % 500 == 0:
                hf.flush()
                gc.collect()

        final_size = num_valid
        ds_init.resize((final_size, D))
        ds_approach.resize((final_size, D))
        ds_prepared.resize((final_size, D))
        ds_grasp.resize((final_size, D))
        hf.flush()

    duration = time.time() - ts
    print(f"Time to generate grasp: {duration}")
    if num_samples > 0:
        print(f"Avg time per sample: {duration / num_samples}")
    if num_no_col > 0:
        print(f"Avg time per no-collision: {duration / num_no_col}")
    print(f"Num of samples: {num_samples}")
    print(f"Num of no-collision samples: {num_no_col}")
    print(f"Num of valid samples: {num_valid}")
    if num_samples > 0:
        print(f"Rate of no-collision: {num_no_col / num_samples}")
    if num_no_col > 0:
        print(f"Rate of valid: {num_valid / num_no_col}")

    with h5py.File(h5path, "r") as hf:
        grasp_dict = {name: hf[name][()] for name in hf.keys()}

    if len(grasp_dict.get("qpos_grasp", [])):
        print(grasp_dict["qpos_grasp"][0])
    np.save(npypath, grasp_dict)
    print(f"Saved grasp data to {npypath}")


if __name__ == "__main__":
    main()
