import json
from pathlib import Path

import h5py

import eval_dataset


def _write_grasp_h5(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("qpos_grasp", data=rows, dtype="f4")


class _FakeMjHO:
    def __init__(self, obj_info, hand_xml_path, target_body_params, object_fixed=False, visualize=False):
        self.obj_info = obj_info
        self.hand_xml_path = hand_xml_path
        self.target_body_params = target_body_params
        self.object_fixed = object_fixed
        self.visualize = visualize
        self.current_qpos = None

    def reset(self):
        self.current_qpos = None

    def set_hand_qpos(self, hand_qpos):
        self.current_qpos = hand_qpos

    def sim_under_extforce(self, qpos_grasp, visualize=False, **kwargs):
        return bool(qpos_grasp[0] > 0), 0.01, 1.0

    def is_contact(self):
        if self.current_qpos is None:
            return False
        return bool(self.current_qpos[0] < 0)

    def _render_viewer(self):
        return None


def test_evaluate_dataset_manifest_counts_successes(tmp_path: Path, monkeypatch):
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
                    "cam_in": "obj_a/scale080/partial_pc_warp/cam_in.npy",
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
                "dataset": {"include": ["YCB"], "root": "unused", "verbose": False, "scales": [0.08]},
                "object": {"id": 0},
                "sampling": {"n_points": 1, "downsample_for_sim": 1, "Nd": 1, "rot_n": 1, "d_min": 0.01, "d_max": 0.02},
                "transform": {
                    "base_rot_grasp_to_palm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "extra_euler": {"axis": "x", "degrees": 0.0},
                },
                "hand": {
                    "xml_path": str(mjcf_path),
                    "prepared_joints": [0.0] * 20,
                    "approach_joints": [0.0] * 20,
                    "shift_local": [0.0, 0.0, -0.02],
                    "target_body_params": {"finger": [1.0, 1.0]},
                },
                "validation": {"contact_min_count": 4},
                "output": {"base_dir": "outputs", "max_cap": 10, "h5_name": "grasp.h5", "npy_name": "grasp.npy", "dataset_root": str(tmp_path / "datasets")},
                "extforce": {"duration": 0.5, "trans_thresh": 0.05, "angle_thresh": 10.0, "grip_delta": 0.05, "force_mag": 1.0, "check_step": 50},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_dataset, "MjHO", _FakeMjHO)

    summary = eval_dataset.evaluate_dataset_manifest(
        run_config_path=str(config_path),
        split="test",
    )

    assert summary["evaluated_items"] == 1
    assert summary["total_attempts"] == 3
    assert summary["total_success"] == 1
    assert abs(summary["success_rate"] - (1.0 / 3.0)) < 1e-6
    assert summary["items"][0]["success_count"] == 1
    assert summary["items"][0]["grasp_count"] == 3
    assert summary["items"][0]["attempts"][1]["failure_stage"] == "qpos_prepared_contact"
    assert summary["items"][0]["attempts"][2]["failure_stage"] == "qpos_init_contact"
