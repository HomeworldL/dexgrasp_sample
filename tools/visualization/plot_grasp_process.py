import os
import torch
import math
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser
from src.fc_metric import calcu_dfc_metric
from src.sample import *

torch.random.manual_seed(0)

if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")

    obj_name = "035_power_drill"
    obj_info = ds.get_info(obj_name)
    # print(obj_info)

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")

    pcs_sample_poisson, norms_sample_poisson = ds.get_point_cloud(
        obj_name, n_points=512, method="poisson"
    )

    ts = time.time()
    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        Nd=1,
        d_min=0.04,
        d_max=0.041,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=None,
    )
    transforms_np = transforms.cpu().numpy()
    print("Time to sample:", time.time() - ts)
    print("Shape of transforms:", transforms.shape)  # (M,4,4)

    # trans to hand coord
    rot_grasp_to_palm = (
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        @ R.from_euler("y", 45, degrees=True).as_matrix()
    ).T

    # N,4,4 -> N, 7 pos quat
    rotation_matrices = transforms_np[:, :3, :3] @ rot_grasp_to_palm

    positions = transforms_np[:, :3, 3]
    rot = R.from_matrix(rotation_matrices)
    quaternions = rot.as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)
    pose = np.concatenate(
        [
            positions,
            quaternions,
        ],
        axis=1,
    )

    xml_path = os.path.join(
        os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
    )
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
    rk = RobotKinematics(xml_path)

    # joints
    q = np.array([0, 0.5, 0.4, 0.4] * 2 + [0, 0.3, 0.3, 0.3] * 2 + [1.6, 0.0, 0.3, 0.3])

    q_expanded = np.tile(q, (pose.shape[0], 1))
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1)

    N = qpos_prepared_sample.shape[0]
    approach_joints = np.array(
        [0, 0.0, 0.0, 0.0] * 2 + [0, 0.0, 0.0, 0.0] * 2 + [1.6, 0.0, 0.0, 0.0]
    )
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

    # mjho.open_viewer()
    qpos_init_list = []
    qpos_approach_list = []
    qpos_prepared_list = []
    qpos_grasp_list = []
    ho_contact_list = []
    hh_contact_list = []
    history_list = []
    for i in tqdm(range(qpos_prepared_sample.shape[0]), desc="sampling"):
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

        # sim_grasp
        mjho.set_hand_qpos(qpos_prepared_sample[i])
        qpos_grasp, history = mjho.sim_grasp(visualize=False, record_history=True)
        ho_contact, hh_contact = mjho.get_contact_info(obj_margin=0.00)

        if len(ho_contact) >= 3:

            # contact_pos = np.array([h["contact_pos"] for h in ho_contact])
            # contact_norm = np.array([h["contact_normal"] for h in ho_contact])

            # dfc_metric = calcu_dfc_metric(contact_pos, contact_norm)
            # print(f"DFC metric: {dfc_metric}")
            # exit()

            qpos_init_list.append(qpos_init_sample[i].copy())
            qpos_approach_list.append(qpos_approach_sample[i].copy())
            qpos_prepared_list.append(qpos_prepared_sample[i].copy())
            qpos_grasp_list.append(qpos_grasp.copy())
            ho_contact_list.append(ho_contact)
            hh_contact_list.append(hh_contact)
            history_list.append(history)

            # hand_qpos = mjho.get_hand_qpos()
            # col0 = rk.get_posed_meshes(hand_qpos, kind="collision")

        if len(qpos_prepared_list) > 40:
            break

    # validation sim
    qpos_init_valid_list = []
    qpos_approach_valid_list = []
    qpos_prepared_valid_list = []
    qpos_grasp_valid_list = []
    ho_contact_valid_list = []
    hh_contact_valid_list = []
    history_valid_list = []
    grasp_stats_valid_list = []

    mjho_valid = MjHO(
        obj_info, xml_path, target_body_params=target_body_params, object_fixed=False
    )
    # mjho_valid.open_viewer()
    for i in tqdm(range(len(qpos_grasp_list)), desc="validation"):
        is_valid, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
            qpos_grasp_list[i], visualize=False
        )
        if is_valid:
            qpos_init_valid_list.append(qpos_init_list[i])
            qpos_approach_valid_list.append(qpos_approach_list[i])
            qpos_prepared_valid_list.append(qpos_prepared_list[i])
            qpos_grasp_valid_list.append(qpos_grasp_list[i])
            ho_contact_valid_list.append(ho_contact_list[i])
            hh_contact_valid_list.append(hh_contact_list[i])
            history_valid_list.append(history_list[i])

            grasp_stats_valid_list.append(
                {
                    "pos_delta": pos_delta,
                    "angle_delta": angle_delta,
                }
            )

    print(f"Num of valid samples: {len(qpos_grasp_valid_list)}")

    # http://localhost:8086 
    for i in range(len(qpos_grasp_valid_list)):

        hand_qpos = qpos_prepared_valid_list[i]
        mesh_hand = rk.get_posed_meshes(hand_qpos, kind="visual")
        his = history_valid_list[i]
        tip_positions = np.concatenate(his["tip_positions"])

        Ntips = tip_positions.shape[0]
        cmap = cm.get_cmap("jet")
        # 按索引渐变，从 0->1 映射到 colormap，得到 Nx4 RGBA，取前三列 RGB
        colors_tip = cmap(np.linspace(0.0, 1.0, Ntips))[:, :3].astype(np.float32)

        # vis
        meshes_for_vis = {
            "mesh_base": mesh_base,
            "mesh_hand": mesh_hand,
        }

        pointclouds_for_vis = {
            "pcs_sample_poisson": (pcs_sample_poisson, None),
            "tip_positions": (tip_positions, colors_tip),
        }

        visualize_with_viser(meshes_for_vis, pointclouds_for_vis, blocking=False)

    while 1:
        time.sleep(1)
