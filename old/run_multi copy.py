import os
import torch
import math
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser
from src.fc_metric import calcu_dfc_metric
from src.sample import sample_grasp_frames
from concurrent.futures import ProcessPoolExecutor, as_completed
torch.random.manual_seed(0)

DATA_ROOT = "assets/ycb_datasets"
XML_PATH = "assets/hands/liberhand/liberhand_right.xml"
OUTPUT_ROOT = "outputs"
MAX_WORKERS = 8


def prepare_sampling_for_object(obj_name, ds: DatasetObjects):
    """
    在主进程完成 GPU 采样（如果 sample_grasp_frames 使用 GPU），
    并把 qpos_prepared 保存到磁盘，返回保存路径和一些元数据。
    """

    obj_info = ds.get_info(obj_name)

    pcs_sample_poisson, norms_sample_poisson = ds.get_point_cloud(
        obj_name, n_points=1024, method="poisson"
    )

    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        Nd=3,
        d_min=0.015,
        d_max=0.04,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=None,
    )
    transforms_np = transforms.cpu().numpy()

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

    # joints
    q = np.array([0, 0.5, 0.4, 0.4] * 2 + [0, 0.3, 0.3, 0.3] * 2 + [1.6, 0.0, 0.3, 0.3])

    q_expanded = np.tile(q, (pose.shape[0], 1))
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1)

    return {
        "obj_name": obj_name,
        "obj_info": obj_info,  # small metadata ok to keep
        "qpos_prepared": qpos_prepared_sample,
        "pcs": pcs_sample_poisson,
        "norms": norms_sample_poisson,
    }


def process_object_worker(job_info, xml_path=XML_PATH, output_root=OUTPUT_ROOT):
    """
    在子进程中运行：加载 qpos_prepared（磁盘），构造 MjHO，运行 sim_grasp 和验证，
    把结果保存到 outputs/<obj_name>/grasp_data.npy，返回 summary（数量等）。
    """
    xml_path = os.path.join(os.path.dirname(__file__), xml_path)
    obj_name = job_info["obj_name"]
    obj_info = job_info["obj_info"]
    qpos_prepared = job_info["qpos_prepared"]
    pcs = job_info["pcs"]
    norms = job_info["norms"]

    # prepare target_body_params
    target_body_params = {}
    for finger in range(1, 6):
        for digit, value in [("3", (0.5, 1.0)), ("4_end", (0.3, 0.2))]:
            key = f"hand_right_f{finger}{digit}"
            target_body_params[key] = value

    mjho = MjHO(obj_info, xml_path, target_body_params=target_body_params)
    mjho._set_obj_pts_norms(pcs, norms)

    # iterate samples
    shift_local = np.array([0.0, 0.0, -0.02], dtype=float)
    qpos_init_list = []
    qpos_approach_list = []
    qpos_prepared_list = []
    qpos_grasp_list = []
    ho_contact_list = []
    hh_contact_list = []
    history_list = []

    for i in range(qpos_prepared.shape[0]):
        mjho.set_hand_qpos(qpos_prepared[i])
        is_contact = mjho.is_contact()
        if is_contact:
            continue

        # approach
        qpos_approach = qpos_prepared[i].copy()
        qpos_approach[7:] = np.array(
            [0, 0.0, 0.0, 0.0] * 2 + [0, 0.0, 0.0, 0.0] * 2 + [1.6, 0.0, 0.0, 0.0]
        )
        mjho.set_hand_qpos(qpos_approach)
        is_contact = mjho.is_contact()
        if is_contact:
            continue

        # init
        pos = qpos_prepared[i][:3].copy()
        quat_wxyz = qpos_prepared[i][3:7].copy()
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float
        )
        rot = R.from_quat(quat_xyzw)
        offset_world = rot.apply(shift_local)
        qpos_init = qpos_prepared[i].copy()
        qpos_init[:3] = pos + offset_world
        qpos_init[3:7] = quat_wxyz
        qpos_init[7:] = np.array(
            [0, 0.0, 0.0, 0.0] * 2 + [0, 0.0, 0.0, 0.0] * 2 + [1.6, 0.0, 0.0, 0.0]
        )
        mjho.set_hand_qpos(qpos_init)
        is_contact = mjho.is_contact()
        if is_contact:
            continue

        # sim_grasp
        mjho.set_hand_qpos(qpos_prepared[i])
        qpos_grasp, history = mjho.sim_grasp(visualize=False)
        ho_contact, hh_contact = mjho.get_contact_info(obj_margin=0.00)

        if len(ho_contact) >= 3:
            # store
            qpos_init_list.append(qpos_init)
            qpos_approach_list.append(qpos_approach)
            qpos_prepared_list.append(qpos_prepared[i])
            qpos_grasp_list.append(qpos_grasp)
            # ho_contact_list.append(ho_contact)
            # hh_contact_list.append(hh_contact)
            # history_list.append(history)

    # validation (可以在同一进程内执行)
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

    for i in range(len(qpos_grasp_list)):
        is_valid, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
            qpos_grasp_list[i], visualize=False
        )
        if is_valid:
            qpos_init_valid_list.append(qpos_init_list[i])
            qpos_approach_valid_list.append(qpos_approach_list[i])
            qpos_prepared_valid_list.append(qpos_prepared_list[i])
            qpos_grasp_valid_list.append(qpos_grasp_list[i])

    # save outputs
    grasp_data = {
        "qpos_init": qpos_init_valid_list,
        "qpos_approach": qpos_approach_valid_list,
        "qpos_prepared": qpos_prepared_valid_list,
        "qpos_grasp": qpos_grasp_valid_list,
    }
    save_dir = os.path.join(output_root, obj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "grasp_data.npy"), grasp_data, allow_pickle=True)

    print(
        f"Process {obj_name} done. Num of valid samples: {len(qpos_grasp_valid_list)} and Rate of valid: {len(qpos_grasp_valid_list) / len(qpos_grasp_list)}"
    )

    return {
        "obj_name": obj_name,
        "num_valid": len(qpos_grasp_valid_list),
        "num_try": qpos_prepared.shape[0],
        "rate_valid": len(qpos_grasp_valid_list) / len(qpos_grasp_list),
    }


if __name__ == "__main__":
    # list of objects to process
    ds = DatasetObjects(DATA_ROOT)
    obj_list = [ds.id2name[i] for i in range(4)]
    # obj_list = [ds.id2name[i] for i in range(len(ds.id2name))]

    # 1) 主进程先为每个物体做采样并保存（GPU 任务在此进行）
    job_info = []
    for obj in tqdm(obj_list, desc="prepare sampling"):
        job_info.append(prepare_sampling_for_object(obj, ds))

    # 2) 使用 ProcessPoolExecutor 批量并行处理物体（最多 MAX_WORKERS 个进程同时）
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(process_object_worker, job): job["obj_name"] for job in job_info
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="processing objects"
        ):
            try:
                res = fut.result()
                print("Finished", res)
                results.append(res)
            except Exception as e:
                print("Worker failed:", e)
    print("All done. Summary:", results)
