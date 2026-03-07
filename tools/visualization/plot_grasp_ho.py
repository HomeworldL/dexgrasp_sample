import os
import trimesh
from typing import Dict, Any, Tuple, Union, Sequence
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

ds = DatasetObjects("assets/ycb_datasets")

obj_name = "035_power_drill"
obj_info = ds.get_info(obj_name)

xml_path = os.path.join(
    os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
)
mjho = MjHO(obj_info, xml_path)
rk = RobotKinematics(xml_path)

grasp_data_path = os.path.join(
    os.path.dirname(__file__), f"./outputs/{obj_name}/grasp_data.npy"
)
grasp_data = np.load(grasp_data_path, allow_pickle=True).item()
# print(f"Loaded grasp data: {grasp_data}")

qpos_init_list = grasp_data["qpos_init"]
qpos_approach_list=grasp_data["qpos_approach"]
qpos_prepared_list=grasp_data["qpos_prepared"]
qpos_grasp_list=grasp_data["qpos_grasp"]

vis_id_list = [0, 500, -1]
vis_init_list = []
vis_approach_list = []
vis_prepared_list = []
vis_grasp_list = []
for i in vis_id_list:
    vis_init_list.append(
        rk.get_posed_meshes(qpos_init_list[i], kind="visual")
    )
    vis_approach_list.append(
        rk.get_posed_meshes(qpos_approach_list[i], kind="visual")
    )
    vis_prepared_list.append(
        rk.get_posed_meshes(qpos_prepared_list[i], kind="visual")
    )
    vis_grasp_list.append(
        rk.get_posed_meshes(qpos_grasp_list[i], kind="visual")
    )
    
# visualize
mesh_base = ds.get_mesh(obj_name, "inertia")

meshes_for_vis = {
    "mesh_base": mesh_base,
}
for i in range(len(vis_id_list)):
        # meshes_for_vis[f"vis_init_{i}"] = vis_init_list[i]
        # meshes_for_vis[f"vis_approach_{i}"] = vis_approach_list[i]
        # meshes_for_vis[f"vis_prepared_{i}"] = vis_prepared_list[i]
        meshes_for_vis[f"vis_grasp_{i}"] = vis_grasp_list[i]
     
visualize_with_viser(meshes_for_vis)