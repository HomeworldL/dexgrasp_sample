from pathlib import Path
import json

import trimesh

from src.dataset_objects import DatasetObjects


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def _make_valid_object(obj_dir: Path, obj_name: str) -> None:
    _touch(obj_dir / "coacd.obj")
    _touch(obj_dir / "convex_parts" / "part_000.obj")


def _make_mesh_object(obj_dir: Path, obj_name: str) -> None:
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "convex_parts").mkdir(parents=True, exist_ok=True)
    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    m.export(obj_dir / "coacd.obj")
    m.export(obj_dir / "convex_parts" / "part_000.obj")
    m.export(obj_dir / "visual.obj")


def test_nested_layout_scan_and_get_info_by_id(tmp_path: Path):
    obj_name = "MSO_obj_a"
    obj = tmp_path / "MSO" / obj_name
    _make_valid_object(obj, obj_name)

    ds = DatasetObjects(str(tmp_path), dataset_names=["MSO"], prebuild_scales=False)
    assert obj_name in ds.get_index()

    info = ds.get_info(obj_name)
    assert info["object_name"] == obj_name
    assert info["coacd_abs"].endswith("coacd.obj")
    assert len(info["convex_parts_abs"]) == 1
    assert "visual_abs" in info

    info_by_id = ds.get_info(0)
    assert info_by_id["object_name"] == obj_name


def test_manifest_only_success_objects(tmp_path: Path):
    ds_dir = tmp_path / "ShapeNetCore"
    ok_name = "ShapeNetCore_ok_001"
    bad_name = "ShapeNetCore_bad_001"
    _make_valid_object(ds_dir / ok_name, ok_name)
    _make_valid_object(ds_dir / bad_name, bad_name)

    manifest = {
        "dataset": "ShapeNetCore",
        "objects": [
            {"object_id": ok_name, "process_status": "success"},
            {"object_id": bad_name, "process_status": "failed"},
        ],
    }
    (ds_dir / "manifest.process_meshes.json").write_text(json.dumps(manifest), encoding="utf-8")

    ds = DatasetObjects(str(tmp_path), dataset_names=["ShapeNetCore"], prebuild_scales=False)
    keys = set(ds.get_index().keys())
    assert ok_name in keys
    assert bad_name not in keys


def test_dataset_include_filter(tmp_path: Path):
    _make_valid_object(tmp_path / "MSO" / "MSO_obj_a", "MSO_obj_a")
    _make_valid_object(tmp_path / "DDG" / "DDG_obj_b", "DDG_obj_b")

    ds = DatasetObjects(str(tmp_path), dataset_names=["MSO"], prebuild_scales=False)
    names = list(ds.get_index().keys())
    assert "MSO_obj_a" in names
    assert "DDG_obj_b" not in names


def test_missing_required_file_skip(tmp_path: Path):
    obj_name = "MSO_incomplete"
    obj = tmp_path / "MSO" / obj_name
    _touch(obj / "convex_parts" / "part_000.obj")

    try:
        DatasetObjects(str(tmp_path), dataset_names=["MSO"], prebuild_scales=False)
        assert False, "Expected RuntimeError for empty valid index"
    except RuntimeError:
        pass


def test_missing_convex_parts_skip(tmp_path: Path):
    obj_name = "MSO_incomplete2"
    obj = tmp_path / "MSO" / obj_name
    _touch(obj / "coacd.obj")

    try:
        DatasetObjects(str(tmp_path), dataset_names=["MSO"], prebuild_scales=False)
        assert False, "Expected RuntimeError for empty valid index"
    except RuntimeError:
        pass


def test_prebuild_scale_assets_and_scale_keys(tmp_path: Path):
    obj_name = "MSO_mesh_obj"
    obj = tmp_path / "MSO" / obj_name
    _make_mesh_object(obj, obj_name)

    ds = DatasetObjects(
        str(tmp_path),
        dataset_names=["MSO"],
        scales=[0.06, 0.08],
        dataset_tag="run_YCB_liberhand",
        dataset_output_root=str(tmp_path / "datasets"),
        prebuild_scales=True,
    )
    info = ds.get_info(obj_name)
    assert info["visual_abs"] is not None
    assert "scale060" in info["scale_assets"]
    assert "scale080" in info["scale_assets"]
    assert Path(info["scale_assets"]["scale060"]["coacd_abs"]).exists()
    assert Path(info["scale_assets"]["scale060"]["xml_abs"]).exists()
