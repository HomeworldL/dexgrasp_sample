from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from train.train_policy_retrieval import (
    ObjectScaleBatchSampler,
    RetrievalDataset,
    compute_candidate_logits,
    select_candidate_indices,
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


def test_retrieval_dataset_builds_candidate_pool(tmp_path: Path) -> None:
    manifest_path = _write_dataset_sample(tmp_path)
    dataset = RetrievalDataset(
        manifest_path=manifest_path,
        point_count=12,
        seed=9,
        grasp_selection_mode="per_view_random",
    )

    assert len(dataset) == 2
    assert dataset.object_scale_to_qpos["obj_a__scale080"].shape == (2, 27)
    assert dataset.object_scale_to_targets_norm["obj_a__scale080"].shape == (2, 26)
    assert dataset.records[0].object_scale_key == "obj_a__scale080"


def test_object_scale_batch_sampler_groups_indices() -> None:
    sampler = ObjectScaleBatchSampler(
        groups={"obj_a__scale080": [0, 1, 2, 3], "obj_b__scale080": [4, 5, 6, 7]},
        shuffle=False,
        seed=0,
    )
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_candidate_selection_picks_nearest_embedding() -> None:
    query = torch.tensor([[0.0, 0.0], [2.1, 2.2]], dtype=torch.float32)
    candidates = torch.tensor([[0.1, 0.1], [2.0, 2.0]], dtype=torch.float32)
    logits = compute_candidate_logits(query, candidates, temperature=1.0)
    indices = select_candidate_indices(query, candidates, temperature=1.0)

    assert logits.shape == (2, 2)
    assert indices.tolist() == [0, 1]
