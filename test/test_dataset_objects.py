from pathlib import Path
import json

import trimesh

from src.dataset_objects import DatasetObjects


def _make_mesh(path: Path):
    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def _make_dataset_with_manifest(tmp_path: Path, dataset_name: str, obj_name: str) -> Path:
    obj_dir = tmp_path / dataset_name / obj_name
    _make_mesh(obj_dir / "coacd.obj")
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


def test_manifest_driven_build_returns_object_scale_items(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")

    ds = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="YCB",
        scales=[0.06, 0.08],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
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
            raw_dataset_root=str(tmp_path),
            raw_dataset_name="MISSING_DATASET",
            scales=[0.06],
            objdata_tag="objdata_YCB",
            graspdata_tag="graspdata_YCB_liberhand",
            generated_dataset_root=str(tmp_path / "datasets"),
            rebuild_existing_assets=True,
            verbose=False,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_get_point_cloud_uses_coacd(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    ds = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="YCB",
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
        verbose=False,
    )

    pts, norms = ds.get_point_cloud(0, n_points=128, method="poisson")
    assert pts.shape[1] == 3
    assert norms.shape[1] == 3


def test_object_name_filter_limits_index(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_002_obj")

    ds = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="YCB",
        scales=[0.06],
        object_names=["YCB_002_obj"],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
        verbose=False,
    )

    items = ds.get_entries()
    assert len(items) == 1
    assert items[0]["object_name"] == "YCB_002_obj"


def test_rebuild_existing_assets_cleans_scale_dir(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")

    ds = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="YCB",
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
        verbose=False,
    )

    asset_dir = Path(ds.get_entries()[0]["asset_dir_abs"])
    stale_path = asset_dir / "stale.txt"
    stale_path.write_text("stale", encoding="utf-8")
    assert stale_path.exists()

    rebuilt = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="YCB",
        scales=[0.06],
        objdata_tag="objdata_YCB",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
        verbose=False,
    )

    rebuilt_asset_dir = Path(rebuilt.get_entries()[0]["asset_dir_abs"])
    assert rebuilt_asset_dir == asset_dir
    assert not stale_path.exists()
    assert (rebuilt_asset_dir / "object.xml").exists()
    assert (rebuilt_asset_dir / "coacd.obj").exists()


def test_missing_objdata_asset_raises_without_rebuild(tmp_path: Path):
    _make_dataset_with_manifest(tmp_path, "YCB", "YCB_001_obj")

    try:
        DatasetObjects(
            raw_dataset_root=str(tmp_path),
            raw_dataset_name="YCB",
            scales=[0.06],
            objdata_tag="objdata_YCB",
            graspdata_tag="graspdata_YCB_liberhand",
            generated_dataset_root=str(tmp_path / "datasets"),
            verbose=False,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "Missing objdata asset" in str(exc)
