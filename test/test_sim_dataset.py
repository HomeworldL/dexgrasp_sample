import json
from pathlib import Path

import h5py
import numpy as np

import sim_dataset
from utils.utils_sample import write_grasp_npy_from_h5


def _write_grasp_h5(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("qpos_grasp", data=rows, dtype="f4")


class _FakeMjHO:
    def __init__(
        self,
        obj_info,
        hand_xml_path,
        anchor_params,
        hand_profile,
        object_profile,
        object_fixed=False,
        visualize=False,
    ):
        self.obj_info = obj_info
        self.hand_xml_path = hand_xml_path
        self.anchor_params = anchor_params
        self.hand_profile = hand_profile
        self.object_profile = object_profile
        self.object_fixed = object_fixed
        self.visualize = visualize
        self.current_qpos = None

    def reset(self):
        self.current_qpos = None

    def set_hand_qpos(self, hand_qpos):
        self.current_qpos = hand_qpos

    def build_pregrasp_qpos(self, qpos_target, prepared_joints):
        return np.asarray(qpos_target, dtype=float)

    def sim_under_extforce(self, qpos_target, qpos_prepared, visualize=False, **kwargs):
        return bool(qpos_target[0] > 0), 0.01, 1.0

    def is_contact(self):
        if self.current_qpos is None:
            return False
        return bool(self.current_qpos[0] < 0)

    def _render_viewer(self):
        return None


TEST_HAND_PROFILE = {
    "ctrl_qpos_slices": [[7, 8]],
    "friction_coef": [0.3, 0.01],
    "solimp": [0.4, 0.99, 0.0001, 0.5, 2.0],
    "solref": [0.003, 1.0],
    "side_swing_indices": [],
    "thumb_relax_indices": [],
    "thumb_relax_divisor": 1.0,
}
TEST_OBJECT_PROFILE = {
    "friction_coef": [0.3, 0.01],
    "solimp": [0.4, 0.99, 0.0001, 0.5, 2.0],
    "solref": [0.003, 1.0],
}


def test_simulate_dataset_manifest_counts_successes(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "graspdata_YCB_liberhand"
    item_dir = dataset_dir / "obj_a" / "scale080"
    grasp_h5_path = item_dir / "grasp.h5"
    mjcf_path = item_dir / "object.xml"
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    _write_grasp_h5(
        grasp_h5_path,
        rows=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    with h5py.File(grasp_h5_path, "a") as handle:
        handle.create_dataset(
            "qpos_init",
            data=[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype="f4",
        )
        handle.create_dataset(
            "qpos_prepared",
            data=[
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype="f4",
        )
        handle.create_dataset(
            "qpos_squeeze",
            data=[
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype="f4",
        )

    (dataset_dir / "test.json").write_text(
        json.dumps(
            [
                {
                    "global_id": 0,
                    "object_scale_key": "obj_a__scale080",
                    "object_name": "obj_a",
                    "output_path": "obj_a/scale080",
                    "coacd_path": "obj_a/scale080/coacd.obj",
                    "mjcf_path": "obj_a/scale080/object.xml",
                    "grasp_h5_path": "obj_a/scale080/grasp.h5",
                    "grasp_npy_path": "obj_a/scale080/grasp.npy",
                    "partial_pc_path": [],
                    "partial_pc_cam_path": [],
                    "cam_ex_path": [],
                    "cam_in": "obj_a/scale080/pc_warp/cam_in.npy",
                    "scale": 0.08,
                }
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "run_YCB_liberhand.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "data": {
                    "raw_dataset_name": "YCB",
                    "raw_dataset_root": "unused",
                    "generated_dataset_root": str(tmp_path / "datasets"),
                    "objdata_tag": "objdata_YCB",
                    "graspdata_tag": "graspdata_YCB_liberhand",
                    "verbose": False,
                    "scales": [0.08],
                    "max_cap": 10,
                    "max_time_sec": 90.0,
                    "h5_name": "grasp.h5",
                    "npy_name": "grasp.npy",
                    "fail_h5_name": "grasp_fail.h5",
                    "fail_npy_name": "grasp_fail.npy",
                    "flush_every": 1,
                    "fail_keep_ratio": 1.0,
                    "min_valid_count": 1
                },
                "object": {"id": 0},
                "hand": {
                    "xml_path": str(mjcf_path),
                    "prepared_joints": [0.0] * 20,
                    "approach_joints": [0.0] * 20,
                    "shift_local": [0.0, 0.0, -0.02],
                    "anchor_params": {"finger": 1.0},
                    "profile": TEST_HAND_PROFILE,
                    "transform": {
                        "base_rot_grasp_to_palm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "extra_euler": {"axis": "x", "degrees": 0.0},
                    },
                },
                "sampling": {"n_points": 1, "downsample_for_sim": 1, "Nd": 1, "rot_n": 1, "d_min": 0.01, "d_max": 0.02, "pc_subdir": "pc_warp"},
                "profile_object": TEST_OBJECT_PROFILE,
                "sim_grasp": {"contact_min_count": 4, "target_point_method": 2},
                "extforce": {"duration": 0.5, "trans_thresh": 0.05, "angle_thresh": 10.0, "grip_delta": 0.05, "force_mag": 1.0, "check_steps": 50},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sim_dataset, "MjHO", _FakeMjHO)

    summary = sim_dataset.evaluate_dataset_manifest(
        run_config_path=str(config_path),
        split="test",
    )

    assert summary["evaluated_items"] == 1
    assert summary["total_attempts"] == 3
    assert summary["total_success"] == 1
    assert abs(summary["success_rate"] - (1.0 / 3.0)) < 1e-6
    assert summary["items"][0]["success_count"] == 1
    assert summary["items"][0]["grasp_count"] == 3
    assert summary["items"][0]["validated_qpos_key"] == "qpos_squeeze"
    assert summary["items"][0]["attempts"][1]["success"] is False
    assert "failure_stage" not in summary["items"][0]["attempts"][1]
    assert summary["items"][0]["attempts"][2]["failure_stage"] == "qpos_init_contact"
    assert summary["qpos_dtype"] == "float32"


def test_simulate_dataset_manifest_allows_explicit_float64_cast(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "graspdata_YCB_liberhand"
    item_dir = dataset_dir / "obj_b" / "scale080"
    grasp_h5_path = item_dir / "grasp.h5"
    mjcf_path = item_dir / "object.xml"
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    _write_grasp_h5(grasp_h5_path, rows=[[1.0, 0.0, 0.0]])
    with h5py.File(grasp_h5_path, "a") as handle:
        handle.create_dataset("qpos_init", data=[[1.0, 0.0, 0.0]], dtype="f4")
        handle.create_dataset("qpos_prepared", data=[[1.0, 0.0, 0.0]], dtype="f4")
        handle.create_dataset("qpos_squeeze", data=[[1.0, 0.0, 0.0]], dtype="f4")

    (dataset_dir / "test.json").write_text(
        json.dumps(
            [
                {
                    "global_id": 0,
                    "object_scale_key": "obj_b__scale080",
                    "object_name": "obj_b",
                    "output_path": "obj_b/scale080",
                    "coacd_path": "obj_b/scale080/coacd.obj",
                    "mjcf_path": "obj_b/scale080/object.xml",
                    "grasp_h5_path": "obj_b/scale080/grasp.h5",
                    "grasp_npy_path": "obj_b/scale080/grasp.npy",
                    "partial_pc_path": [],
                    "partial_pc_cam_path": [],
                    "cam_ex_path": [],
                    "cam_in": "obj_b/scale080/pc_warp/cam_in.npy",
                    "scale": 0.08,
                }
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "run_YCB_liberhand.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "data": {
                    "raw_dataset_name": "YCB",
                    "raw_dataset_root": "unused",
                    "generated_dataset_root": str(tmp_path / "datasets"),
                    "objdata_tag": "objdata_YCB",
                    "graspdata_tag": "graspdata_YCB_liberhand",
                    "verbose": False,
                    "scales": [0.08],
                    "max_cap": 10,
                    "max_time_sec": 90.0,
                    "h5_name": "grasp.h5",
                    "npy_name": "grasp.npy",
                    "fail_h5_name": "grasp_fail.h5",
                    "fail_npy_name": "grasp_fail.npy",
                    "flush_every": 1,
                    "fail_keep_ratio": 1.0,
                    "min_valid_count": 1
                },
                "object": {"id": 0},
                "hand": {
                    "xml_path": str(mjcf_path),
                    "prepared_joints": [0.0] * 20,
                    "approach_joints": [0.0] * 20,
                    "shift_local": [0.0, 0.0, -0.02],
                    "anchor_params": {"finger": 1.0},
                    "profile": TEST_HAND_PROFILE,
                    "transform": {
                        "base_rot_grasp_to_palm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "extra_euler": {"axis": "x", "degrees": 0.0},
                    },
                },
                "sampling": {"n_points": 1, "downsample_for_sim": 1, "Nd": 1, "rot_n": 1, "d_min": 0.01, "d_max": 0.02, "pc_subdir": "pc_warp"},
                "profile_object": TEST_OBJECT_PROFILE,
                "sim_grasp": {"contact_min_count": 4, "target_point_method": 2},
                "extforce": {"duration": 0.5, "trans_thresh": 0.05, "angle_thresh": 10.0, "grip_delta": 0.05, "force_mag": 1.0, "check_steps": 50},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sim_dataset, "MjHO", _FakeMjHO)

    summary = sim_dataset.evaluate_dataset_manifest(
        run_config_path=str(config_path),
        split="test",
        qpos_dtype_name="float64",
    )

    assert summary["total_success"] == 1
    assert summary["qpos_dtype"] == "float64"


def test_simulate_dataset_manifest_requires_qpos_squeeze(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "graspdata_YCB_liberhand"
    item_dir = dataset_dir / "obj_c" / "scale080"
    grasp_h5_path = item_dir / "grasp.h5"
    mjcf_path = item_dir / "object.xml"
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)
    mjcf_path.write_text("<mujoco/>", encoding="utf-8")
    _write_grasp_h5(grasp_h5_path, rows=[[-1.0, 0.0, 0.0]])
    with h5py.File(grasp_h5_path, "a") as handle:
        handle.create_dataset("qpos_init", data=[[1.0, 0.0, 0.0]], dtype="f4")
        handle.create_dataset("qpos_prepared", data=[[1.0, 0.0, 0.0]], dtype="f4")

    (dataset_dir / "test.json").write_text(
        json.dumps(
            [
                {
                    "global_id": 0,
                    "object_scale_key": "obj_c__scale080",
                    "object_name": "obj_c",
                    "output_path": "obj_c/scale080",
                    "coacd_path": "obj_c/scale080/coacd.obj",
                    "mjcf_path": "obj_c/scale080/object.xml",
                    "grasp_h5_path": "obj_c/scale080/grasp.h5",
                    "grasp_npy_path": "obj_c/scale080/grasp.npy",
                    "partial_pc_path": [],
                    "partial_pc_cam_path": [],
                    "cam_ex_path": [],
                    "cam_in": "obj_c/scale080/pc_warp/cam_in.npy",
                    "scale": 0.08,
                }
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "run_YCB_liberhand.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "data": {
                    "raw_dataset_name": "YCB",
                    "raw_dataset_root": "unused",
                    "generated_dataset_root": str(tmp_path / "datasets"),
                    "objdata_tag": "objdata_YCB",
                    "graspdata_tag": "graspdata_YCB_liberhand",
                    "verbose": False,
                    "scales": [0.08],
                    "max_cap": 10,
                    "max_time_sec": 90.0,
                    "h5_name": "grasp.h5",
                    "npy_name": "grasp.npy",
                    "fail_h5_name": "grasp_fail.h5",
                    "fail_npy_name": "grasp_fail.npy",
                    "flush_every": 1,
                    "fail_keep_ratio": 1.0,
                    "min_valid_count": 1
                },
                "object": {"id": 0},
                "hand": {
                    "xml_path": str(mjcf_path),
                    "prepared_joints": [0.0] * 20,
                    "approach_joints": [0.0] * 20,
                    "shift_local": [0.0, 0.0, -0.02],
                    "anchor_params": {"finger": 1.0},
                    "profile": TEST_HAND_PROFILE,
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
                    "pc_subdir": "pc_warp",
                },
                "profile_object": TEST_OBJECT_PROFILE,
                "sim_grasp": {"contact_min_count": 4, "target_point_method": 2},
                "extforce": {
                    "duration": 0.5,
                    "trans_thresh": 0.05,
                    "angle_thresh": 10.0,
                    "grip_delta": 0.05,
                    "force_mag": 1.0,
                    "check_steps": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sim_dataset, "MjHO", _FakeMjHO)

    summary = sim_dataset.evaluate_dataset_manifest(
        run_config_path=str(config_path),
        split="test",
    )

    assert summary["evaluated_items"] == 0
    assert "qpos_squeeze" in summary["skipped_items"][0]["reason"]


def test_write_grasp_npy_preserves_float32(tmp_path: Path):
    h5_path = tmp_path / "grasp.h5"
    npy_path = tmp_path / "grasp.npy"
    rows = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    squeeze_rows = np.asarray([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    with h5py.File(h5_path, "w") as handle:
        for key in ("qpos_init", "qpos_approach", "qpos_prepared", "qpos_grasp"):
            handle.create_dataset(key, data=rows, dtype="f4")
        handle.create_dataset("qpos_squeeze", data=squeeze_rows, dtype="f4")

    write_grasp_npy_from_h5(h5_path, npy_path)

    payload = np.load(npy_path, allow_pickle=True).item()
    assert payload["qpos_grasp"].dtype == np.float32
    assert np.array_equal(payload["qpos_grasp"], rows)
    assert payload["qpos_squeeze"].dtype == np.float32
    assert np.array_equal(payload["qpos_squeeze"], squeeze_rows)
