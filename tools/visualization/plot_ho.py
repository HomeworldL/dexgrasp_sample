import os
import numpy as np
import time
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser


if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")
    obj_name = "035_power_drill"
    obj_info = ds.get_info(obj_name)
    # print(obj_info)
    mesh_base = ds.get_mesh(obj_name, "inertia")

    xml_path = os.path.join(
        os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
    )
    mjho = MjHO(obj_info, xml_path)
    mjcf = mjho.export_xml("debug.xml")

    print(mjho.nq)
    print(mjho.nq_hand)

    obj_pose = mjho.get_obj_pose()
    print(obj_pose)
    hand_qpos = mjho.get_hand_qpos()
    print(hand_qpos)

    mjho.set_obj_pose(obj_pose)
    q = np.array(
        [
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.3,
            0.3,
            0.3,
            0,
            0.3,
            0.3,
            0.3,
            1.6,
            0.0,
            0.3,
            0.3,
        ]
    )
    # q = np.zeros((20))
    qpos = np.concatenate((np.array([-0.1, 0, -0.08, 1, 0, 0, 0]), q))
    mjho.set_hand_qpos(qpos)

    hand_qpos = mjho.get_hand_qpos()
    print(hand_qpos)

    rk = RobotKinematics(xml_path)
    hand_qpos = mjho.get_hand_qpos()
    rk.forward_kinematics(hand_qpos)
    vis = rk.get_posed_meshes(hand_qpos, kind="visual")

    meshes_for_vis = {}
    pointclouds_for_vis = {}

    meshes_for_vis["mesh_base"] = mesh_base
    if vis is not None:
        meshes_for_vis["mesh_hand_vis"] = vis

    col = rk.get_posed_meshes(hand_qpos, kind="collision")
    if col is not None:
        meshes_for_vis["mesh_hand_col"] = col

    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
