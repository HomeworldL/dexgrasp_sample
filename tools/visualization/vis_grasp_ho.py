import os

from typing import Sequence, Optional
import numpy as np
import time
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

ds = DatasetObjects("assets/ycb_datasets")

# obj_name = "035_power_drill"  # 002_master_chef_can 035_power_drill
# obj_name = ds.id2name[0]
# obj_name = "006_mustard_bottle"
obj_name = "002_master_chef_can"
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
qpos_approach = grasp_data["qpos_approach"]
qpos_prepared = grasp_data["qpos_prepared"]
qpos_grasp = grasp_data["qpos_grasp"]


# visualize
tmp_dir = f"./tmp/{obj_name}"
os.makedirs(tmp_dir, exist_ok=True)


def merge_trimesh_list(mesh_list):
    """将多个trimesh对象合并为一个"""
    if not mesh_list:
        return None

    vertices_list = []
    faces_list = []

    for mesh in mesh_list:
        if isinstance(mesh, trimesh.Trimesh):
            current_faces = len(vertices_list)
            vertices_list.extend(mesh.vertices)
            faces_list.extend(mesh.faces + current_faces)

    if vertices_list and faces_list:
        merged_vertices = np.array(vertices_list)
        merged_faces = np.array(faces_list)
        return trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces)
    return None


mesh_base = ds.get_mesh(obj_name, "simplified")

for i in range(len(qpos_grasp)):
    col_mesh_grasp = rk.get_posed_meshes(qpos_grasp[i], kind="collision")

    merge_trimesh_list([mesh_base, col_mesh_grasp]).export(
        os.path.join(tmp_dir, f"grasp_{i}.stl")
    )
