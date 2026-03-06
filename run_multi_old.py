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
from src.sample import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import multiprocessing
from multiprocessing import Manager
import random


def set_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

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
        obj_name, n_points=4096, method="poisson"
    )

    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        Nd=1,
        d_min=0.01,
        d_max=0.09,
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


def process_object_worker(
    job_info, progress_queue=None, xml_path=XML_PATH, output_root=OUTPUT_ROOT
):
    """
    job_info must contain qpos_prepared as numpy array in memory or paths if you use disk-exchange.
    progress_queue: a multiprocessing.Manager().Queue() object for sending progress updates
    """
    xml_path = os.path.join(os.path.dirname(__file__), xml_path)
    obj_name = job_info["obj_name"]
    obj_info = job_info["obj_info"]
    qpos_prepared_sample = job_info["qpos_prepared"]
    pcs = job_info["pcs"]
    norms = job_info["norms"]

    # prepare target_body_params
    target_body_params = {}
    for finger in range(1, 6):
        for digit, value in [("3", (0.5, 1.0)), ("4_end", (0.3, 0.2))]:
            key = f"hand_right_f{finger}{digit}"
            target_body_params[key] = value

    mjho = MjHO(obj_info, xml_path, target_body_params=target_body_params)
    pcs_for_sim, norms_for_sim, sel_idx = downsample_fps(pcs, norms, 1024, seed=0)
    mjho._set_obj_pts_norms(pcs_for_sim, norms_for_sim)

    # iterate samples
    shift_local = np.array([0.0, 0.0, -0.02], dtype=float)
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

    qpos_init_list = []
    qpos_approach_list = []
    qpos_prepared_list = []
    qpos_grasp_list = []
    ho_contact_list = []
    hh_contact_list = []
    history_list = []

    for i in range(qpos_prepared_sample.shape[0]):
        # for i in tqdm(range(qpos_prepared_sample.shape[0]), desc="sampling"):
        # print(f"{obj_name} {i}/{qpos_prepared.shape[0]}")
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
        qpos_grasp, history = mjho.sim_grasp(visualize=False)
        ho_contact, hh_contact = mjho.get_contact_info(obj_margin=0.00)

        if len(ho_contact) >= 3:
            # store
            qpos_init_list.append(qpos_init_sample[i].copy())
            qpos_approach_list.append(qpos_approach_sample[i].copy())
            qpos_prepared_list.append(qpos_prepared_sample[i].copy())
            qpos_grasp_list.append(qpos_grasp.copy())
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

    # 告诉主进程这个物体将要验证多少个样本（用于创建 tqdm 总量）
    num_to_validate = len(qpos_grasp_list)
    if progress_queue is not None:
        progress_queue.put(("set_total", obj_name, num_to_validate))

    # 控制消息频率，避免频繁 put 到 queue（每处理 update_every 个样本发送一次）
    update_every = max(1, num_to_validate // 200)  # 最多 200 条更新，可调
    processed_since_last = 0

    for i in range(num_to_validate):
        is_valid, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
            qpos_grasp_list[i], visualize=False
        )
        if is_valid:
            qpos_init_valid_list.append(qpos_init_list[i])
            qpos_approach_valid_list.append(qpos_approach_list[i])
            qpos_prepared_valid_list.append(qpos_prepared_list[i])
            qpos_grasp_valid_list.append(qpos_grasp_list[i])

        processed_since_last += 1
        # 发送进度（每 update_every 或最后一次）
        if progress_queue is not None and (
            processed_since_last >= update_every or i == num_to_validate - 1
        ):
            # 发送 delta（processed_since_last），然后重置计数
            progress_queue.put(("progress", obj_name, processed_since_last))
            processed_since_last = 0

    # 验证结束，发送 done（确保进度条到满）
    if progress_queue is not None:
        progress_queue.put(("done", obj_name, None))

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
        "num_try": qpos_prepared_sample.shape[0],
        "rate_valid": len(qpos_grasp_valid_list) / len(qpos_grasp_list),
    }


# ---------- 主进程: 创建 manager queue 和 per-object bars ----------
def start_progress_listener(progress_queue):
    """
    监听进程发送的进度消息，并维护动态创建的 tqdm 条。
    支持消息类型：
      ("set_total", obj_name, total)  # worker 在开始验证前发送，告诉要验证多少个样本
      ("progress", obj_name, delta)   # worker 每处理 delta 个验证样本就发送一次
      ("done", obj_name, None)        # 验证完成，标记为完成
      ("__EXIT__", None, None)        # 关闭监听器
    """
    bars = {}
    positions = {}  # obj_name -> position index in terminal
    next_pos = 0
    lock = threading.Lock()

    def listener():
        nonlocal next_pos
        while True:
            msg = progress_queue.get()
            if msg is None:
                break
            typ, obj_name, val = msg
            if typ == "__EXIT__":
                break
            if typ == "set_total":
                total = int(val)
                with lock:
                    if obj_name not in bars:
                        pos = next_pos
                        next_pos += 1
                        positions[obj_name] = pos
                        bars[obj_name] = tqdm(
                            total=total,
                            desc=f"valid-{obj_name}",
                            position=pos,
                            leave=True,
                        )
                    else:
                        # update total if already exists
                        bars[obj_name].total = int(val)
                        bars[obj_name].refresh()
            elif typ == "progress":
                if obj_name in bars:
                    bars[obj_name].update(int(val))
                else:
                    # create bar with unknown total (fallback): use leave=False simple counter
                    with lock:
                        pos = next_pos
                        next_pos += 1
                        positions[obj_name] = pos
                        bars[obj_name] = tqdm(
                            total=0, desc=f"valid-{obj_name}", position=pos, leave=True
                        )
                        bars[obj_name].update(int(val))
            elif typ == "done":
                if obj_name in bars:
                    bars[obj_name].n = bars[obj_name].total
                    bars[obj_name].refresh()

        # close all bars
        for b in bars.values():
            b.close()

    th = threading.Thread(target=listener, daemon=True)
    th.start()
    return th


if __name__ == "__main__":
    # list of objects to process
    ds = DatasetObjects(DATA_ROOT)
    # obj_list = [ds.id2name[i] for i in range(5,12)]
    obj_list = [ds.id2name[i] for i in range(1, len(ds.id2name))]
    # obj_list = [ds.id2name[i] for i in range(20)]

    # 1) 主进程先为每个物体做采样并保存（GPU 任务在此进行）
    job_info = []
    obj_totals = {}  # obj_name -> total_samples
    for obj in tqdm(obj_list, desc="prepare sampling"):
        job_info.append(prepare_sampling_for_object(obj, ds))

    # 2) 使用 ProcessPoolExecutor 批量并行处理物体（最多 MAX_WORKERS 个进程同时）
    manager = Manager()
    progress_queue = manager.Queue()
    listener_thread = start_progress_listener(progress_queue)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(process_object_worker, job, progress_queue): job["obj_name"]
            for job in job_info
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

    # 所有 worker 完成后，告诉 listener 退出
    progress_queue.put(("__EXIT__", None, None))
    listener_thread.join()
    print("All done. Summary:", results)
