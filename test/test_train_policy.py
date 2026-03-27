from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from train import sim_policy
from train.train_policy import (
    FlattenedGraspDataset,
    QPOS_DIM,
    TARGET_DIM,
    deterministic_resample_points,
    qpos_from_target,
    target_from_qpos,
)


def _write_dataset_sample(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "datasets" / "graspdata_YCB_liberhand"
    sample_dir = dataset_dir / "obj_a" / "scale080"
    render_dir = sample_dir / "partial_pc_warp"
    render_dir.mkdir(parents=True, exist_ok=True)

    partial_pc_0 = np.arange(30, dtype=np.float32).reshape(10, 3) / 100.0
    partial_pc_1 = np.arange(24, dtype=np.float32).reshape(8, 3) / 80.0
    np.save(render_dir / "partial_pc_00.npy", partial_pc_0.astype(np.float16))
    np.save(render_dir / "partial_pc_01.npy", partial_pc_1.astype(np.float16))

    qpos_rows = np.stack(
        [
            np.concatenate(
                [
                    np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.linspace(-0.2, 0.8, 20, dtype=np.float32),
                ]
            ),
            np.concatenate(
                [
                    np.array([-0.1, 0.05, 0.2, 0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),
                    np.linspace(-0.4, 1.2, 20, dtype=np.float32),
                ]
            ),
        ],
        axis=0,
    ).astype(np.float32)

    with h5py.File(sample_dir / "grasp.h5", "w") as handle:
        handle.create_dataset("qpos_squeeze", data=qpos_rows, dtype="f4")

    manifest = [
        {
            "global_id": 0,
            "object_scale_key": "obj_a__scale080",
            "object_name": "obj_a",
            "output_path": "obj_a/scale080",
            "coacd_path": "obj_a/scale080/coacd.obj",
            "mjcf_path": "obj_a/scale080/object.xml",
            "grasp_h5_path": "obj_a/scale080/grasp.h5",
            "grasp_npy_path": "obj_a/scale080/grasp.npy",
            "partial_pc_path": [
                "obj_a/scale080/partial_pc_warp/partial_pc_00.npy",
                "obj_a/scale080/partial_pc_warp/partial_pc_01.npy",
            ],
            "partial_pc_cam_path": [],
            "cam_ex_path": [],
            "cam_in": "obj_a/scale080/partial_pc_warp/cam_in.npy",
            "scale": 0.08,
        }
    ]
    manifest_path = dataset_dir / "train.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_target_roundtrip_from_qpos() -> None:
    qpos = np.concatenate(
        [
            np.array([0.04, -0.05, 0.06, 0.9659258, 0.0, 0.2588190, 0.0], dtype=np.float32),
            np.linspace(-0.3, 1.1, 20, dtype=np.float32),
        ]
    ).astype(np.float32)

    target = target_from_qpos(qpos)
    restored = qpos_from_target(target)

    assert target.shape == (TARGET_DIM,)
    assert restored.shape == (QPOS_DIM,)
    assert np.allclose(restored[:3], qpos[:3], atol=1e-5)
    assert np.allclose(restored[7:], qpos[7:], atol=1e-5)
    assert np.allclose(np.abs(restored[3:7]), np.abs(qpos[3:7]), atol=1e-5)


def test_deterministic_resample_points_is_stable() -> None:
    points = np.arange(24, dtype=np.float32).reshape(8, 3)
    sampled_a = deterministic_resample_points(points, point_count=16, seed=5)
    sampled_b = deterministic_resample_points(points, point_count=16, seed=5)
    sampled_c = deterministic_resample_points(points, point_count=16, seed=6)

    assert sampled_a.shape == (16, 3)
    assert np.array_equal(sampled_a, sampled_b)
    assert not np.array_equal(sampled_a, sampled_c)


def test_flattened_grasp_dataset_is_deterministic(tmp_path: Path) -> None:
    manifest_path = _write_dataset_sample(tmp_path)
    dataset_a = FlattenedGraspDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )
    dataset_b = FlattenedGraspDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )

    assert len(dataset_a) == 2
    assert len(dataset_b) == 2
    assert np.array_equal(dataset_a.points, dataset_b.points)
    assert np.array_equal(dataset_a.targets, dataset_b.targets)
    assert dataset_a.records[0].grasp_index == dataset_b.records[0].grasp_index
    assert dataset_a.records[1].view_index == 1


def test_shared_object_scale_mode_uses_same_grasp_for_all_views(tmp_path: Path) -> None:
    manifest_path = _write_dataset_sample(tmp_path)
    dataset = FlattenedGraspDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="shared_object_scale",
    )

    assert len(dataset) == 2
    assert dataset.records[0].grasp_index == dataset.records[1].grasp_index


class _FakeModel(torch.nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros((points.shape[0], TARGET_DIM), dtype=points.dtype, device=points.device)


class _FakeMjHO:
    def __init__(
        self,
        obj_info,
        hand_xml_path,
        target_body_params,
        object_fixed=False,
        visualize=False,
    ):
        self.obj_info = obj_info

    def sim_under_extforce(self, qpos_squeeze, visualize=False, **kwargs):
        return bool(qpos_squeeze[0] > 0), 0.01, 1.0


def test_sim_policy_counts_successes(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_dataset_sample(tmp_path)
    dataset = FlattenedGraspDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )

    xml_path = tmp_path / "hand.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")
    config_path = tmp_path / "run_YCB_liberhand.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "dataset": {
                    "include": ["YCB"],
                    "root": "unused",
                    "verbose": False,
                    "scales": [0.08],
                },
                "object": {"id": 0},
                "hand": {
                    "xml_path": str(xml_path),
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
                    "n_points": 1,
                    "downsample_for_sim": 1,
                    "Nd": 1,
                    "rot_n": 1,
                    "d_min": 0.01,
                    "d_max": 0.02,
                },
                "sim_grasp": {"contact_min_count": 4},
                "output": {
                    "max_cap": 10,
                    "max_time_sec": 90.0,
                    "h5_name": "grasp.h5",
                    "npy_name": "grasp.npy",
                    "dataset_root": str(tmp_path / "datasets"),
                },
                "extforce": {
                    "duration": 0.5,
                    "trans_thresh": 0.05,
                    "angle_thresh": 10.0,
                    "grip_delta": 0.05,
                    "force_mag": 1.0,
                    "check_step": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    target_mean = dataset.targets[0].copy()
    checkpoint = {
        "target_mean": target_mean,
        "target_std": np.ones_like(target_mean, dtype=np.float32),
    }

    monkeypatch.setattr(sim_policy, "MjHO", _FakeMjHO)
    monkeypatch.setattr(
        sim_policy,
        "load_model_from_checkpoint",
        lambda checkpoint_path, device: (_FakeModel().to(device), checkpoint),
    )
    monkeypatch.setitem(sim_policy.EXPERIMENT, "manifest_path", str(manifest_path))
    monkeypatch.setitem(sim_policy.EXPERIMENT, "run_config_path", str(config_path))
    monkeypatch.setitem(sim_policy.EXPERIMENT, "point_count", 12)
    monkeypatch.setitem(sim_policy.EXPERIMENT, "seed", 9)
    monkeypatch.setitem(sim_policy.EXPERIMENT, "device", "cpu")

    summary = sim_policy.evaluate_policy_checkpoint(tmp_path / "checkpoint.pt", limit=1)

    assert summary["sample_count"] == 1
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 1.0
