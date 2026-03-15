import json
from pathlib import Path

from run_multi import build_split_records, split_records_by_object


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0")


def _make_entry(dataset_dir: Path, global_id: int, object_name: str, scale_tag: str, scale: float):
    output_dir = dataset_dir / object_name / scale_tag
    _touch(output_dir / "coacd.obj")
    _touch(output_dir / "object.xml")
    _touch(output_dir / "convex_parts" / "part_000.obj")
    _touch(output_dir / "grasp.h5")
    _touch(output_dir / "grasp.npy")
    render_dir = output_dir / "partial_pc_warp"
    _touch(render_dir / "cam_in.npy")
    _touch(render_dir / "cam_ex_00.npy")
    _touch(render_dir / "partial_pc_00.npy")
    _touch(render_dir / "partial_pc_cam_00.npy")
    return {
        "global_id": global_id,
        "object_scale_key": f"{object_name}__{scale_tag}",
        "object_name": object_name,
        "output_dir_abs": str(output_dir),
        "coacd_abs": str(output_dir / "coacd.obj"),
        "convex_parts_abs": [str(output_dir / "convex_parts" / "part_000.obj")],
        "mjcf_abs": str(output_dir / "object.xml"),
        "scale": scale,
    }


def test_build_split_records_and_split_by_object(tmp_path: Path):
    dataset_dir = tmp_path / "datasets" / "graspdata_YCB_liberhand"
    entries = [
        _make_entry(dataset_dir, 0, "obj_a", "scale080", 0.08),
        _make_entry(dataset_dir, 1, "obj_a", "scale100", 0.10),
        _make_entry(dataset_dir, 2, "obj_b", "scale080", 0.08),
        _make_entry(dataset_dir, 3, "obj_b", "scale100", 0.10),
        _make_entry(dataset_dir, 4, "obj_c", "scale080", 0.08),
        _make_entry(dataset_dir, 5, "obj_c", "scale100", 0.10),
        _make_entry(dataset_dir, 6, "obj_d", "scale080", 0.08),
        _make_entry(dataset_dir, 7, "obj_d", "scale100", 0.10),
        _make_entry(dataset_dir, 8, "obj_e", "scale080", 0.08),
        _make_entry(dataset_dir, 9, "obj_e", "scale100", 0.10),
    ]

    records, skipped = build_split_records(
        entries=entries,
        dataset_dir=dataset_dir,
        render_subdir="partial_pc_warp",
    )

    assert skipped == []
    assert len(records) == len(entries)
    assert records[0]["output_path"] == "obj_a/scale080"
    assert records[0]["grasp_h5_path"] == "obj_a/scale080/grasp.h5"
    assert records[0]["partial_pc_path"] == ["obj_a/scale080/partial_pc_warp/partial_pc_00.npy"]
    assert records[0]["partial_pc_cam_path"] == ["obj_a/scale080/partial_pc_warp/partial_pc_cam_00.npy"]
    assert records[0]["cam_ex_path"] == ["obj_a/scale080/partial_pc_warp/cam_ex_00.npy"]
    assert records[0]["cam_in"] == "obj_a/scale080/partial_pc_warp/cam_in.npy"

    train_records, test_records = split_records_by_object(records)
    train_objects = {record["object_name"] for record in train_records}
    test_objects = {record["object_name"] for record in test_records}

    assert train_objects == {"obj_a", "obj_b", "obj_c", "obj_d"}
    assert test_objects == {"obj_e"}
    assert len(train_records) == 8
    assert len(test_records) == 2

    serialized = json.loads(json.dumps(test_records, ensure_ascii=False))
    assert serialized[0]["object_scale_key"].startswith("obj_e__")
