import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from utils.utils_file import load_config
from utils.utils_template_sampling import (
    TemplateAnchorSet,
    _solve_weighted_rigid_transform_row,
    build_template_anchor_set,
    refine_pose_candidates_with_template_alignment,
)


def test_solve_weighted_rigid_transform_row_recovers_known_transform():
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]],
        dtype=torch.float32,
    )
    rot_np = R.from_euler("z", 30.0, degrees=True).as_matrix().astype(np.float32)
    rot = torch.tensor(rot_np).unsqueeze(0)
    trans = torch.tensor([[0.1, -0.2, 0.05]], dtype=torch.float32)
    target = torch.matmul(source, rot) + trans.unsqueeze(1)
    weights = torch.ones((source.shape[1],), dtype=torch.float32)

    solved_rot, solved_trans = _solve_weighted_rigid_transform_row(source, target, weights)

    assert torch.allclose(solved_rot, rot, atol=1e-5)
    assert torch.allclose(solved_trans, trans, atol=1e-5)


def test_build_template_anchor_set_uses_target_bodies(monkeypatch):
    class _FakeBody:
        def __init__(self, name: str):
            self.name = name

    class _FakeModel:
        nq = 9
        nbody = 3

        @staticmethod
        def body(body_id: int):
            return _FakeBody(["world", "finger_a", "finger_b"][body_id])

    class _FakeRobotKinematics:
        def __init__(self, xml_path: str):
            self.model = _FakeModel()
            self.data = SimpleNamespace(
                xpos=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.2, 0.0],
                    ],
                    dtype=np.float64,
                ),
                xmat=np.asarray(
                    [
                        np.eye(3).reshape(-1),
                        np.eye(3).reshape(-1),
                        R.from_euler("x", 90.0, degrees=True).as_matrix().reshape(-1),
                    ],
                    dtype=np.float64,
                ),
            )

        def forward_kinematics(self, qpos: np.ndarray):
            self.qpos = qpos.copy()

    monkeypatch.setattr("utils.utils_template_sampling.RobotKinematics", _FakeRobotKinematics)

    anchors = build_template_anchor_set(
        hand_xml_path="unused.xml",
        prepared_joints=np.asarray([0.1, 0.2], dtype=np.float64),
        target_body_params={"finger_a": [0.5, 1.0], "finger_b": [0.3, 0.2]},
    )

    assert anchors.positions.shape == (2, 3)
    assert np.allclose(anchors.positions[0], np.asarray([0.1, 0.0, 0.0]))
    assert np.allclose(anchors.normals[0], np.asarray([0.0, 0.0, 1.0]))
    assert np.allclose(anchors.contact_weights, np.asarray([0.5, 0.3]))
    assert np.allclose(anchors.distance_weights, np.asarray([1.0, 0.2]))


def test_refine_pose_candidates_returns_input_when_disabled():
    cfg = {"hand": {"prepared_joints": [0.0] * 20}}
    pose = np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    out = refine_pose_candidates_with_template_alignment(
        cfg=cfg,
        hand_xml_path="unused.xml",
        target_body_params={},
        pose_candidates=pose,
        obj_points=np.zeros((4, 3), dtype=np.float64),
        obj_normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1)),
    )
    assert np.array_equal(out, pose)


def test_refine_pose_candidates_keeps_best_pose_and_roundtrips_world_frame(monkeypatch):
    monkeypatch.setattr(
        "utils.utils_template_sampling.build_template_anchor_set",
        lambda **_: TemplateAnchorSet(
            positions=np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64),
            normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            contact_weights=np.asarray([1.0, 1.0], dtype=np.float64),
            distance_weights=np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )

    cfg = {
        "hand": {"prepared_joints": [0.0] * 20},
        "template_sampling": {
            "enabled": True,
            "keep_topk": 1,
            "batch_size": 8,
            "opt_steps": 1,
            "normal_offset": 0.01,
            "device": "cpu",
        },
    }
    pose_candidates = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    obj_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.4, 0.4, 0.4],
        ],
        dtype=np.float64,
    )
    obj_normals = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    refined = refine_pose_candidates_with_template_alignment(
        cfg=cfg,
        hand_xml_path="unused.xml",
        target_body_params={"finger": [1.0, 1.0]},
        pose_candidates=pose_candidates,
        obj_points=obj_points,
        obj_normals=obj_normals,
    )

    assert refined.shape == (1, 7)
    assert np.allclose(refined[0], pose_candidates[0], atol=1e-5)


def test_load_config_validates_template_sampling_fields(tmp_path: Path):
    hand_xml = tmp_path / "hand.xml"
    hand_xml.write_text("<mujoco/>", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "dataset": {
                    "root": "unused",
                    "include": ["YCB"],
                    "scales": [0.08],
                    "verbose": False,
                },
                "hand": {
                    "xml_path": str(hand_xml),
                    "prepared_joints": [0.0] * 20,
                    "approach_joints": [0.0] * 20,
                    "shift_local": [0.0, 0.0, -0.02],
                    "target_body_params": {"finger": [1.0, 1.0]},
                    "transform": {
                        "base_rot_grasp_to_palm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "extra_euler": {"axis": "x", "degrees": 0.0},
                    },
                },
                "sampling": {
                    "n_points": 16,
                    "downsample_for_sim": 8,
                    "Nd": 1,
                    "rot_n": 1,
                    "d_min": 0.01,
                    "d_max": 0.02,
                },
                "template_sampling": {
                    "enabled": True,
                    "batch_size": 128,
                    "opt_steps": 2,
                    "normal_offset": 0.01,
                },
                "sim_grasp": {"contact_min_count": 4},
                "extforce": {
                    "duration": 0.5,
                    "trans_thresh": 0.05,
                    "angle_thresh": 10.0,
                    "grip_delta": 0.05,
                    "force_mag": 1.0,
                    "check_step": 50,
                },
                "output": {
                    "max_cap": 10,
                    "max_time_sec": 60.0,
                    "h5_name": "grasp.h5",
                    "npy_name": "grasp.npy",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="template_sampling.keep_topk"):
        load_config(str(config_path))
