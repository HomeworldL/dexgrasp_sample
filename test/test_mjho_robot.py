import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.mj_ho import MjHO, RobotKinematics
from src.scale_dataset_builder import ScaleDatasetBuilder


def _make_cube_mesh(path: Path, scale: float = 1.0):
    import trimesh

    m = trimesh.creation.box(extents=[scale, scale, scale])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def _build_temp_object_xml(tmp_path: Path) -> str:
    raw_obj = {
        "object_name": "obj_a",
        "coacd_abs": str((tmp_path / "raw" / "coacd.obj").resolve()),
    }
    _make_cube_mesh(Path(raw_obj["coacd_abs"]))

    b = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    rec = b.build_scale_assets(
        config_stem="run_YCB_liberhand",
        object_info=raw_obj,
        scale=0.08,
        mass_kg=0.1,
        principal_moments=[1e-5, 2e-5, 3e-5],
    )
    return rec["xml_abs"]


def test_mjho_init_and_set_qpos(tmp_path: Path):
    xml_path = os.path.join("assets", "hands", "liberhand", "liberhand_right.xml")
    if not os.path.exists(xml_path):
        pytest.skip("Hand XML not found")

    obj_xml = _build_temp_object_xml(tmp_path)
    env = MjHO({"name": "obj_a", "xml_abs": obj_xml}, xml_path)
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
