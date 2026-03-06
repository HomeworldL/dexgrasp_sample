import os
import argparse
import torch
import math
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
import random
import gc
import h5py
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser
from src.fc_metric import calcu_dfc_metric
from src.sample import *


def set_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sample grasps for a YCB object.")
    p.add_argument(
        "-o",
        "--obj",
        type=str,
        default="006_mustard_bottle",
        help="Object name (e.g. '035_power_drill'). "
        "If omitted, uses the first object in the dataset.",
    )
    args = p.parse_args()
    obj_name = args.obj
    print(f"Using object: {obj_name}")

    ds = DatasetObjects("assets/ycb_datasets")
    obj_info = ds.get_info(obj_name)

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")
    mesh_coacd = ds.get_mesh(obj_name, "coacd")
    convex_pieces = ds.get_mesh(obj_name, "convex_pieces")

    pcs_sample_poisson, norms_sample_poisson = ds.get_point_cloud(
        obj_name, n_points=4096, method="poisson"
    )

    ts = time.time()
    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        Nd=1,
        d_min=0.02,
        d_max=0.06,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=None,
    )
    transforms_np = transforms.cpu().numpy()
    del transforms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Time to sample:", time.time() - ts)
    print("Shape of transforms:", transforms_np.shape)  # (M,4,4)

    # trans to hand coord
    rot_grasp_to_palm = (
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        @ R.from_euler("y", 45, degrees=True).as_matrix()
    )

    T_grasp_to_palm = np.eye(4)
    T_grasp_to_palm[:3, :3] = rot_grasp_to_palm
    T_grasp_to_palm[:3, 3] = np.array([0.03, 0, 0.065])
    transforms_np = transforms_np @ np.linalg.inv(T_grasp_to_palm)

    # N,4,4 -> N, 7 pos quat
    positions = transforms_np[:, :3, 3]
    rot = R.from_matrix(transforms_np[:, :3, :3])
    quaternions = rot.as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)
    pose = np.concatenate(
        [
            positions,
            quaternions,
        ],
        axis=1,
    ).astype(np.float32)

    xml_path = os.path.join(
        os.path.dirname(__file__), "./assets/hands/liberhand2/liberhand2_right.xml"
    )
    hand_name = os.path.basename(xml_path).split(".")[0]
    print(f"hand_name: {hand_name}")

    target_body_params = {}
    for finger in range(1, 6):
        for digit, value in [("3", (0.5, 1.0)), ("4_end", (0.3, 0.2))]:
            key = f"hand_right_f{finger}{digit}"
            target_body_params[key] = value
    mjho = MjHO(obj_info, xml_path, target_body_params=target_body_params)
    pcs_for_sim, norms_for_sim, sel_idx = downsample_fps(
        pcs_sample_poisson, norms_sample_poisson, 1024, seed=0
    )
    mjho._set_obj_pts_norms(pcs_for_sim, norms_for_sim)
    # mjho.open_viewer()

    mjho_valid = MjHO(
        obj_info, xml_path, target_body_params=target_body_params, object_fixed=False
    )
    # mjho_valid.open_viewer()

    rk = RobotKinematics(xml_path)

    # joints
    q = np.array([0, 0.5, 0.4, 0.4] * 4 + [1.5, 0.5, 0.3, 0.3])

    q_expanded = np.tile(q, (pose.shape[0], 1)).astype(np.float32)
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1).astype(np.float32)

    N = qpos_prepared_sample.shape[0]
    approach_joints = np.array([0, 0.0, 0.0, 0.0] * 4 + [1.5, 0.0, 0.0, 0.0])
    # build full qpos_approach array
    qpos_approach_sample = np.tile(qpos_prepared_sample, (1, 1)).copy()
    qpos_approach_sample[:, 7:] = np.tile(approach_joints, (N, 1))

    shift_local = np.array([0.0, 0.0, -0.02], dtype=float)
    positions = qpos_prepared_sample[:, :3]  # (N,3)
    quats_wxyz = qpos_prepared_sample[:, 3:7]  # (N,4) 你的存放是 w,x,y,z
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]  # (N,4)
    rot_batch = R.from_quat(quats_xyzw)  # batch rotation
    offset_world = rot_batch.apply(shift_local)  # (N,3) — rot_i * shift_local
    qpos_init_sample = qpos_prepared_sample.copy()
    qpos_init_sample[:, :3] = positions + offset_world
    qpos_init_sample[:, 7:] = np.tile(approach_joints, (N, 1))

    print(f"Shape of qpos_prepared: {qpos_prepared_sample.shape}")  # (M, 27)
    print(f"Shape of qpos_approach: {qpos_approach_sample.shape}")  # (M, 27)
    print(f"Shape of qpos_init: {qpos_init_sample.shape}")

    save_dir = os.path.join("outputs", hand_name, obj_name)
    os.makedirs(save_dir, exist_ok=True)
    h5path = os.path.join(save_dir, "grasp_data.h5")
    npypath = os.path.join(save_dir, "grasp_data.npy")
    print(f"Saving to {h5path} ...")
    D = qpos_prepared_sample.shape[1]
    MAX_CAP = 2000

    # qpos_init_valid_list = []
    # qpos_approach_valid_list = []
    # qpos_prepared_valid_list = []
    # qpos_grasp_valid_list = []
    # ho_contact_valid_list = []
    # hh_contact_valid_list = []
    # history_valid_list = []
    # grasp_stats_valid_list = []
    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]

    ts = time.time()

    with h5py.File(h5path, "w") as hf:
        ds_init = hf.create_dataset(
            "qpos_init", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4"
        )
        ds_approach = hf.create_dataset(
            "qpos_approach", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4"
        )
        ds_prepared = hf.create_dataset(
            "qpos_prepared", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4"
        )
        ds_grasp = hf.create_dataset(
            "qpos_grasp", shape=(MAX_CAP, D), maxshape=(None, D), dtype="f4"
        )

        for i in tqdm(
            range(qpos_prepared_sample.shape[0]), desc="sampling", miniters=50
        ):
            if num_valid >= MAX_CAP:
                # 提前退出
                num_samples = i
                break

            mjho.set_hand_qpos(qpos_prepared_sample[i])

            is_contact = mjho.is_contact()
            if is_contact:
                continue

            # approach
            mjho.set_hand_qpos(qpos_approach_sample[i])
            is_contact = mjho.is_contact()
            if is_contact:
                continue

            # init
            mjho.set_hand_qpos(qpos_init_sample[i])
            is_contact = mjho.is_contact()
            if is_contact:
                continue

            num_no_col += 1

            # sim_grasp
            mjho.set_hand_qpos(qpos_prepared_sample[i])
            qpos_grasp, _ = mjho.sim_grasp(visualize=False)
            ho_contact, _ = mjho.get_contact_info(obj_margin=0.00)

            if len(ho_contact) >= 4:

                # contact_pos = np.array([h["contact_pos"] for h in ho_contact])
                # contact_norm = np.array([h["contact_normal"] for h in ho_contact])

                # dfc_metric = calcu_dfc_metric(contact_pos, contact_norm)
                # print(f"DFC metric: {dfc_metric}")
                # exit()

                # hand_qpos = mjho.get_hand_qpos()
                # col0 = rk.get_posed_meshes(hand_qpos, kind="collision")

                is_valid, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
                    qpos_grasp.copy(), visualize=False
                )
                if is_valid:

                    ds_init[num_valid] = qpos_init_sample[i].astype("f4")
                    ds_approach[num_valid] = qpos_approach_sample[i].astype("f4")
                    ds_prepared[num_valid] = qpos_prepared_sample[i].astype("f4")
                    ds_grasp[num_valid] = qpos_grasp.astype("f4")
                    num_valid += 1

                    # qpos_init_valid_list.append(qpos_init_sample[i].copy())
                    # qpos_approach_valid_list.append(qpos_approach_sample[i].copy())
                    # qpos_prepared_valid_list.append(qpos_prepared_sample[i].copy())
                    # qpos_grasp_valid_list.append(qpos_grasp.copy())
                    # ho_contact_valid_list.append(ho_contact)
                    # hh_contact_valid_list.append(hh_contact)
                    # history_valid_list.append(history)

                    # grasp_stats_valid_list.append(
                    #     {
                    #         "pos_delta": pos_delta,
                    #         "angle_delta": angle_delta,
                    #     }
                    # )

            # periodic flush and GC
            if (i + 1) % 500 == 0:
                hf.flush()
                gc.collect()

        # ---- truncate to actual size ----
        final_size = num_valid
        ds_init.resize((final_size, D))
        ds_approach.resize((final_size, D))
        ds_prepared.resize((final_size, D))
        ds_grasp.resize((final_size, D))
        hf.flush()

    duration = time.time() - ts
    print(f"Time to generate grasp: {duration}")
    print(f"Avg time per sample: {duration / num_samples}")
    print(f"Avg time per no-collision: {duration / num_no_col}")
    print(f"Num of samples: {num_samples}")
    print(f"Num of no-collision samples: {num_no_col}")
    print(f"Num of valid samples: {num_valid}")
    print(f"Rate of no-collision: {num_no_col / num_samples}")
    print(f"Rate of valid: {num_valid / num_no_col}")

    with h5py.File(h5path, "r") as hf:
        grasp_dict = {name: hf[name][()] for name in hf.keys()}

    print(grasp_dict["qpos_grasp"][0])
    np.save(npypath, grasp_dict)
    print(f"Saved grasp data to {npypath}")
