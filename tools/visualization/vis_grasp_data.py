import os

from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

ds = DatasetObjects("assets/ycb_datasets")

# obj_name = "035_power_drill"  # 002_master_chef_can 035_power_drill
# obj_name = ds.id2name[0]
obj_name = "006_mustard_bottle"
# obj_name = "002_master_chef_can"
obj_info = ds.get_info(obj_name)

xml_path = os.path.join(
    os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
    # os.path.dirname(__file__), "./assets/hands/liberhand2/liberhand2_right.xml"
)
mjho = MjHO(obj_info, xml_path)
rk = RobotKinematics(xml_path)

grasp_data_path = os.path.join(
    os.path.dirname(__file__), f"./outputs/liberhand_right/{obj_name}/grasp_data.npy"
    # os.path.dirname(__file__), f"./outputs/liberhand2_right/{obj_name}/grasp_data.npy"
)
grasp_data = np.load(grasp_data_path, allow_pickle=True).item()
# print(f"Loaded grasp data: {grasp_data}")

qpos_init_list = grasp_data["qpos_init"]
qpos_approach=grasp_data["qpos_approach"]
qpos_prepared=grasp_data["qpos_prepared"]
qpos_grasp=grasp_data["qpos_grasp"]

#########################################################
# stats matplotlib
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def compute_nearest_metric_and_plot(qpos_init_list, save_fig=None, show=True,
                                    alpha=1.0, beta=1.0, eps=1e-8):
    """
    输入:
      qpos_init_list: list of pose-like arrays, 每项前7维为 [x,y,z, qw,qx,qy,qz] （与原脚本一致）
      alpha, beta: 合并度量的权重（在归一化后）
    输出:
      nearest_idx_pos: 由位置最近邻决定的索引数组 (N,)
      metrics_dict: 字典包含 position, orientation, combined 三类度量的数组与统计信息
    """
    poses = [np.asarray(p[:7], dtype=float) for p in qpos_init_list]
    N = len(poses)
    if N == 0:
        return np.array([], dtype=int), None, None

    # 拆解位置与四元数（保证四元数顺序为 [qw, qx, qy, qz]）
    positions = np.vstack([p[:3] for p in poses])    # (N,3)
    quats = np.vstack([p[3:7] for p in poses])       # (N,4)
    # 归一化四元数以防数值误差
    q_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    q_norms[q_norms < eps] = 1.0
    quats = quats / q_norms

    # KDTree 用于位置最近邻（判断“相邻”只用位置）
    kdt = cKDTree(positions)
    # query k=2 because first neighbor is itself with dist 0
    dists, idxs = kdt.query(positions, k=2)
    # positions nearest neighbor index (excluding self)
    nearest_idx_pos = idxs[:, 1]   # shape (N,)
    nearest_pos_dist = dists[:, 1] # Euclidean distance to nearest by position

    # 计算所有对的姿态误差时可能的代价较大（N^2），但这里只需要每个样本与其位置邻居的度量
    # 因此只计算 (i, j_pos) 对的姿态误差以节省计算
    def quat_dot_abs(q1, q2):
        # q1, q2: (4,) arrays
        s = float(np.dot(q1, q2))
        s = max(min(s, 1.0), -1.0)  # clamp
        return abs(s)

    pos_errors = np.empty(N, dtype=float)
    ori_errors = np.empty(N, dtype=float)
    for i in range(N):
        j = int(nearest_idx_pos[i])
        pos_errors[i] = np.linalg.norm(positions[i] - positions[j])
        s_q = quat_dot_abs(quats[i], quats[j])
        ori_errors[i] = np.sqrt(max(0.0, 1.0 - s_q * s_q))

    # 若需要，也可以计算每个样本与所有其他样本的最小合并度量（可选）
    # 但按你的要求“相邻只用位置判断”，我们以位置邻居为主。

    # 计算尺度（使用非零中位数以防极端0值）
    def nonzero_median(arr):
        nz = arr[arr > 0.0]
        if nz.size == 0:
            return eps
        return float(np.median(nz))

    med_p = nonzero_median(pos_errors)
    med_q = nonzero_median(ori_errors)

    if med_p < eps:
        med_p = max(np.mean(pos_errors[pos_errors > 0]) if np.any(pos_errors > 0) else eps, eps)
    if med_q < eps:
        med_q = max(np.mean(ori_errors[ori_errors > 0]) if np.any(ori_errors > 0) else eps, eps)

    # 归一化到相同数量级
    dp_norm = pos_errors / med_p
    dq_norm = ori_errors / med_q

    # 合并度量（在归一化后，alpha & beta 的作用为调节偏好）
    combined = alpha * dp_norm + beta * dq_norm

    # 统计信息
    stats = {
        "count": int(N),
        "pos_median": float(np.median(pos_errors)),
        "pos_mean": float(np.mean(pos_errors)),
        "ori_median": float(np.median(ori_errors)),
        "ori_mean": float(np.mean(ori_errors)),
        "combined_median": float(np.median(combined)),
        "combined_mean": float(np.mean(combined)),
    }

    # 绘图：位置、姿态、合并度量三条曲线
    plt.rcParams.update({"font.size": 14})
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    x = np.arange(N)

    axs[0].plot(x, pos_errors, marker="o", linestyle="-", linewidth=1)
    axs[0].set_ylabel("position dist (m)", fontsize=14)
    axs[0].set_title("Position distance to nearest-by-position", fontsize=16)
    axs[0].grid(True)

    axs[1].plot(x, ori_errors, marker="o", linestyle="-", linewidth=1)
    axs[1].set_ylabel("orientation dist (sqrt(1 - s_q^2))", fontsize=14)
    axs[1].set_title("Orientation distance to nearest-by-position", fontsize=16)
    axs[1].grid(True)

    axs[2].plot(x, dp_norm, marker="o", linestyle="-", linewidth=1, label="pos_norm")
    axs[2].plot(x, dq_norm, marker="x", linestyle="--", linewidth=1, label="ori_norm")
    axs[2].plot(x, combined, marker="s", linestyle="-.", linewidth=1, label="combined")
    axs[2].set_xlabel("sample index i", fontsize=14)
    axs[2].set_ylabel("normalized metric", fontsize=14)
    axs[2].set_title("Normalized metrics and combined metric", fontsize=16)
    axs[2].legend(fontsize=12)
    axs[2].grid(True)

    # annotate stats on the figure (top-right of first subplot)
    stats_text = (
        f"count={stats['count']}\n"
        f"pos_mean={stats['pos_mean']:.6f}\n"
        f"pos_med={stats['pos_median']:.6f}\n"
        f"ori_mean={stats['ori_mean']:.6f}\n"
        f"ori_med={stats['ori_median']:.6f}\n"
        f"comb_mean={stats['combined_mean']:.6f}\n"
        f"comb_med={stats['combined_median']:.6f}"
    )
    axs[0].text(0.98, 0.95, stats_text, transform=axs[0].transAxes,
                fontsize=10, verticalalignment="top", horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    plt.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 返回位置最近邻索引与三个度量数组与统计
    metrics = {
        "pos_errors": pos_errors,
        "ori_errors": ori_errors,
        "dp_norm": dp_norm,
        "dq_norm": dq_norm,
        "combined": combined,
        "med_p": med_p,
        "med_q": med_q,
    }
    return nearest_idx_pos, metrics, stats

    
idx, vals, stats = compute_nearest_metric_and_plot(qpos_init_list)
print("Summary stats:", stats)


# visualize
mesh_base = ds.get_mesh(obj_name, "inertia")

meshes_for_vis = {
    "mesh_base": mesh_base,
}

frame_origin = np.array(qpos_init_list)[:, :3]

pointclouds_for_vis = {
    "frame_origin": (frame_origin, None),
}
visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
# frame_for_vis = np.array(qpos_init_list)[:, :7]
# visualize_with_viser(meshes_for_vis, None,frame_for_vis)