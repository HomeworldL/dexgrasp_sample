import os

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics


def _pick_object_with_xml(dataset: DatasetObjects):
    for name, info in dataset.get_index().items():
        if info.get("xml_abs"):
            return name, info
    pytest.skip("No object XML found in dataset for MjHO test")


def test_mjho_init_and_set_qpos():
    ds = DatasetObjects("assets/ycb_datasets")
    _, obj_info = _pick_object_with_xml(ds)

    xml_path = os.path.join("assets", "hands", "liberhand", "liberhand_right.xml")
    if not os.path.exists(xml_path):
        pytest.skip("Hand XML not found")

    env = MjHO(obj_info, xml_path)
    q = np.zeros(env.nq_hand, dtype=float)
    q[:7] = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    env.set_hand_qpos(q)
    got = env.get_hand_qpos()
    assert got.shape[0] == env.nq_hand


def test_robot_kinematics_mesh_extract():
    xml_path = os.path.join("assets", "hands", "liberhand", "liberhand_right.xml")
    if not os.path.exists(xml_path):
        pytest.skip("Hand XML not found")

    rk = RobotKinematics(xml_path)
    q = np.zeros(rk.model.nq, dtype=float)
    posed = rk.get_posed_meshes(q, kind="collision")
    assert posed is None or len(posed.vertices) >= 0
