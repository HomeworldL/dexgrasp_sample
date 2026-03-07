from pathlib import Path
import json

from src.dataset_objects import DatasetObjects


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def _make_valid_object(obj_dir: Path, obj_name: str) -> None:
    _touch(obj_dir / "inertia.obj")
    _touch(obj_dir / "visual.obj")
    _touch(obj_dir / "manifold.obj")
    _touch(obj_dir / "coacd.obj")
    _touch(obj_dir / f"{obj_name}.xml")
    _touch(obj_dir / "convex_parts" / "part_000.obj")


def test_nested_layout_scan_and_get_info_by_id(tmp_path: Path):
    obj_name = "MSO_obj_a"
    obj = tmp_path / "MSO" / obj_name
    _make_valid_object(obj, obj_name)

    ds = DatasetObjects(str(tmp_path), dataset_names=["MSO"])
    assert obj_name in ds.get_index()

    info = ds.get_info(obj_name)
    assert info["dataset"] == "MSO"
    assert info["inertia_abs"].endswith("inertia.obj")
    assert info["visual_abs"].endswith("visual.obj")
    assert len(info["convex_parts_abs"]) == 1

    info_by_id = ds.get_info(0)
    assert info_by_id["name"] == obj_name


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

    ds = DatasetObjects(str(tmp_path), dataset_names=["ShapeNetCore"])
    keys = set(ds.get_index().keys())
    assert ok_name in keys
    assert bad_name not in keys


def test_shapenet_scale_range(tmp_path: Path):
    obj_name = "ShapeNetSem_mug_x1"
    obj = tmp_path / "ShapeNetSem" / obj_name
    _make_valid_object(obj, obj_name)

    ds = DatasetObjects(
        str(tmp_path),
        dataset_names=["ShapeNetSem"],
        shapenet_scale_range=(0.06, 0.15),
        shapenet_scale_seed=7,
    )
    info = ds.get_info(obj_name)
    assert 0.06 <= float(info["scale"]) <= 0.15


def test_dataset_include_filter(tmp_path: Path):
    _make_valid_object(tmp_path / "MSO" / "MSO_obj_a", "MSO_obj_a")
    _make_valid_object(tmp_path / "DDG" / "DDG_obj_b", "DDG_obj_b")

    ds = DatasetObjects(str(tmp_path), dataset_names=["MSO"])
    names = list(ds.get_index().keys())
    assert "MSO_obj_a" in names
    assert "DDG_obj_b" not in names


def test_missing_required_file_skip(tmp_path: Path):
    obj_name = "MSO_incomplete"
    obj = tmp_path / "MSO" / obj_name
    _touch(obj / "inertia.obj")
    _touch(obj / "visual.obj")
    _touch(obj / "manifold.obj")
    _touch(obj / f"{obj_name}.xml")
    _touch(obj / "convex_parts" / "part_000.obj")
    # missing coacd.obj -> should skip

    try:
        DatasetObjects(str(tmp_path), dataset_names=["MSO"])
        assert False, "Expected RuntimeError for empty valid index"
    except RuntimeError:
        pass
