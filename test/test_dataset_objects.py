from pathlib import Path
import json

import pytest
import trimesh

from src.dataset_objects import DatasetObjects
from src.scale_dataset_builder import ScaleDatasetBuilder


def _make_mesh(path: Path):
    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def _make_dataset_with_manifest(
    tmp_path: Path, dataset_name: str, obj_name: str
) -> Path:
    obj_dir = tmp_path / dataset_name / obj_name
    _make_mesh(obj_dir / "coacd.obj")
    _make_mesh(obj_dir / "manifold.obj")
    _make_mesh(obj_dir / "convex_parts" / "part_000.obj")

    manifest = {
        "dataset": dataset_name,
        "summary": {"default_mass_kg": 0.1},
        "objects": [
            {
                "object_id": obj_name,
                "name": obj_name,
                "mesh_path": str((obj_dir / "raw.obj").as_posix()),
                "mass_kg": 0.12,
                "principal_moments": [1e-4, 2e-4, 3e-4],
                "process_status": "success",
            }
        ],
    }
    (tmp_path / dataset_name / "manifest.process_meshes.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return obj_dir


def _build_objdata_assets(
    tmp_path: Path,
    dataset_name: str,
    obj_name: str,
    scales: list[float],
) -> None:
    obj_dir = tmp_path / dataset_name / obj_name
    objdata_tag = f"objdata_{dataset_name}"
    builder = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    out = builder.build_multi_scale_assets(
        config_stem=objdata_tag,
        object_info={
            "object_name": obj_name,
            "coacd_abs": str((obj_dir / "coacd.obj").resolve()),
            "manifold_abs": str((obj_dir / "manifold.obj").resolve()),
        },
        scales=scales,
        mass_kg=0.12,
        principal_moments=[1e-4, 2e-4, 3e-4],
        overwrite=False,
    )
    scales_available = sorted(str(scale_tag) for scale_tag in out.keys())
    manifest_path = tmp_path / "datasets" / objdata_tag / "manifest.process_meshes.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": objdata_tag,
            "summary": {"object_count": 0},
            "objects": [],
        }
    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    objects = [
        obj
        for obj in objects
        if str((obj or {}).get("name") or (obj or {}).get("object_id") or "")
        != obj_name
    ]
    objects.append({"name": obj_name, "scales_available": scales_available})
    manifest["objects"] = sorted(objects, key=lambda item: str(item.get("name", "")))
    manifest["summary"]["object_count"] = len(manifest["objects"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_driven_build_returns_object_scale_items(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _build_objdata_assets(tmp_path, "YCB", "YCB_001_obj", scales=[0.06, 0.08])

    ds = DatasetObjects(
        scales=[0.06, 0.08],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    items = ds.get_entries()
    assert len(items) == 2
    assert items[0]["global_id"] == 0
    assert items[1]["global_id"] == 1
    assert items[0]["object_name"] == "YCB_001_obj"
    assert "objdata_YCB" in items[0]["asset_dir_abs"]
    assert "graspdata_YCB_liberhand" in items[0]["output_dir_abs"]
    assert Path(items[0]["coacd_abs"]).exists()
    assert Path(items[0]["mjcf_abs"]).exists()
    assert Path(items[0]["asset_dir_abs"]).is_dir()
    assert items[0]["object_scale_key"].startswith("YCB_001_obj__scale")


def test_dataset_missing_dir_raises(tmp_path: Path):
    try:
        DatasetObjects(
            scales=[0.06],
            objdata_tag="objdata_YCB",
            graspdata_tag="graspdata_YCB_liberhand",
            generated_dataset_root=str(tmp_path / "datasets"),
            verbose=False,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_sample_surface_for_entry_uses_coacd(tmp_path: Path):
    pytest.importorskip("open3d")
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _build_objdata_assets(tmp_path, "YCB", "YCB_001_obj", scales=[0.06])
    ds = DatasetObjects(
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    info = ds.get_obj_info_by_index(0)
    pts, norms = ds.sample_surface_for_entry(info, n_points=128, method="poisson")
    assert pts.shape[1] == 3
    assert norms.shape[1] == 3


def test_object_name_filter_limits_index(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_002_obj")
    _build_objdata_assets(tmp_path, "YCB", "YCB_001_obj", scales=[0.06])
    _build_objdata_assets(tmp_path, "YCB", "YCB_002_obj", scales=[0.06])

    ds = DatasetObjects(
        scales=[0.06],
        object_names=["YCB_002_obj"],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    items = ds.get_entries()
    assert len(items) == 1
    assert items[0]["object_name"] == "YCB_002_obj"


def test_existing_assets_preserve_scale_dir_contents(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _build_objdata_assets(tmp_path, "YCB", "YCB_001_obj", scales=[0.06])

    ds = DatasetObjects(
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    asset_dir = Path(ds.get_entries()[0]["asset_dir_abs"])
    stale_path = asset_dir / "stale.txt"
    stale_path.write_text("stale", encoding="utf-8")
    assert stale_path.exists()

    indexed = DatasetObjects(
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    indexed_asset_dir = Path(indexed.get_entries()[0]["asset_dir_abs"])
    assert indexed_asset_dir == asset_dir
    assert stale_path.exists()
    assert (indexed_asset_dir / "object.xml").exists()
    assert Path(indexed.get_entries()[0]["coacd_abs"]).exists()


def test_existing_assets_do_not_require_urdf_for_indexing(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _build_objdata_assets(tmp_path, "YCB", "YCB_001_obj", scales=[0.06])

    ds = DatasetObjects(
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )
    asset_dir = Path(ds.get_entries()[0]["asset_dir_abs"])
    urdf_path = asset_dir / "object.urdf"
    urdf_path.unlink()

    indexed = DatasetObjects(
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        verbose=False,
    )

    entry = indexed.get_entries()[0]
    assert entry["object_name"] == "YCB_001_obj"
    assert Path(entry["mjcf_abs"]).exists()


def test_missing_objdata_asset_raises_without_rebuild(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")

    try:
        DatasetObjects(
            scales=[0.06],
            objdata_tag="objdata_YCB",
            graspdata_tag="graspdata_YCB_liberhand",
            generated_dataset_root=str(tmp_path / "datasets"),
            verbose=False,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "Objdata manifest not found" in str(exc)
