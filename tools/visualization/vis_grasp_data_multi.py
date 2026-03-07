import os

from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

def select_nearest_by_pos(qpos_list, k=1000):
    """
    qpos_list: list or array of qpos (each qpos has >=3 dims, first 3 are position)
    Return: trimmed lists (as numpy arrays) containing only the k samples whose positions
            are nearest to the origin (by Euclidean distance on first 3 dims).
    If qpos_list is empty, returns empty array.
    """
    if qpos_list is None:
        return np.zeros((0, )), np.array([], dtype=int)
    arr = np.asarray(qpos_list)
    if arr.size == 0:
        return arr, np.array([], dtype=int)
    # ensure shape (N, D)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # extract positions safely
    pos = arr[:, :3]
    # compute squared distances to origin
    d2 = np.sum(pos * pos, axis=1)
    N = arr.shape[0]
    if N <= k:
        indices = np.arange(N)
    else:
        # get indices of smallest k distances
        indices = np.argpartition(d2, k-1)[:k]
        # sort these indices by actual distance for consistent ordering (closest first)
        indices = indices[np.argsort(d2[indices])]
    return arr[indices], indices


ds = DatasetObjects("assets/ycb_datasets")
obj_list = [ds.id2name[i] for i in range(len(ds.id2name))]
# obj_list = ["006_mustard_bottle", "002_master_chef_can", "035_power_drill"]
xml_path = os.path.join(
    os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
)

for i, obj_name in enumerate(obj_list):
    print(obj_name)
    obj_info = ds.get_info(obj_name)
    # print(obj_info)
    mesh_base = ds.get_mesh(obj_name, "inertia")

    grasp_data_path = os.path.join(
        os.path.dirname(__file__), f"./outputs/{obj_name}/grasp_data.npy"
    )
    grasp_data = np.load(grasp_data_path, allow_pickle=True).item()
    # print(f"Loaded grasp data: {grasp_data}")

    qpos_init_list = grasp_data["qpos_init"]
    qpos_approach_list=grasp_data["qpos_approach"]
    qpos_prepared_list=grasp_data["qpos_prepared"]
    qpos_grasp_list=grasp_data["qpos_grasp"]

    N = len(qpos_grasp_list)
    print(f"Num of samples: {N}")

    MAX_KEEP = 2000
    if N > MAX_KEEP:
        # we choose trimming indices based on qpos_init_list if available, else prepared, else grasp
        base_list = qpos_prepared_list
        trimmed_base, keep_idx = select_nearest_by_pos(base_list, k=MAX_KEEP)
        # Apply same indices to all lists (but guard sizes)
        def take_idx(lst, idx):
            arr = np.asarray(lst)
            # if list shorter than original (defensive), clip indices
            idx_clipped = idx[idx < arr.shape[0]]
            return arr[idx_clipped]
        qpos_init_trim = take_idx(qpos_init_list, keep_idx)
        qpos_approach_trim = take_idx(qpos_approach_list, keep_idx)
        qpos_prepared_trim = take_idx(qpos_prepared_list, keep_idx)
        qpos_grasp_trim = take_idx(qpos_grasp_list, keep_idx)

        print(f"Trimmed from {N} to {trimmed_base.shape[0]} samples.")
    else:
        # just convert to arrays
        def to_arr(lst):
            arr = np.asarray(lst)
            return arr
        qpos_init_trim = to_arr(qpos_init_list)
        qpos_approach_trim = to_arr(qpos_approach_list)
        qpos_prepared_trim = to_arr(qpos_prepared_list)
        qpos_grasp_trim = to_arr(qpos_grasp_list)

    # visualize
    mesh_base = ds.get_mesh(obj_name, "inertia")

    meshes_for_vis = {
        "mesh_base": mesh_base,
    }

    frame_origin = qpos_prepared_trim[:, :3]

    pointclouds_for_vis = {
        "frame_origin": (frame_origin, None),
    }

    if i == len(obj_list) - 1:
        blocking = True
    else:
        blocking = False
    visualize_with_viser(meshes_for_vis, pointclouds_for_vis, blocking=blocking)