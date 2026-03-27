from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from train.train_policy_pose_only import (
    POSE_TARGET_DIM,
    PoseOnlyDataset,
    pose_target_from_qpos,
    transform_from_pose_target,
)
from train.train_policy import _transform_from_qpos


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


def test_pose_target_roundtrip_from_qpos() -> None:
    qpos = np.concatenate(
        [
            np.array([0.04, -0.05, 0.06, 0.9659258, 0.0, 0.2588190, 0.0], dtype=np.float32),
            np.linspace(-0.3, 1.1, 20, dtype=np.float32),
        ]
    ).astype(np.float32)

    target = pose_target_from_qpos(qpos)
    restored = transform_from_pose_target(target)
    original = _transform_from_qpos(qpos)

    assert target.shape == (POSE_TARGET_DIM,)
    assert restored.shape == (4, 4)
    assert np.allclose(restored[:3, 3], qpos[:3], atol=1e-5)
    assert np.allclose(restored[:3, :3], original[:3, :3], atol=1e-5)


def test_pose_only_dataset_is_deterministic(tmp_path: Path) -> None:
    manifest_path = _write_dataset_sample(tmp_path)
    dataset_a = PoseOnlyDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )
    dataset_b = PoseOnlyDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )

    assert len(dataset_a) == 2
    assert np.array_equal(dataset_a.points, dataset_b.points)
    assert np.array_equal(dataset_a.targets, dataset_b.targets)
    assert dataset_a.records[0].grasp_index == dataset_b.records[0].grasp_index
