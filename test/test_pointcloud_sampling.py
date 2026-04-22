from pathlib import Path
import json

import numpy as np
import pytest
import trimesh

from src.dataset_objects import DatasetObjects
from utils.utils_seed import set_seed
from utils.utils_pointcloud import sample_surface_o3d


def _make_mesh_object(obj_dir: Path):
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "convex_parts").mkdir(parents=True, exist_ok=True)
    m = trimesh.creation.icosphere(radius=0.5)
    m.export(obj_dir / "coacd.obj")
    m.export(obj_dir / "convex_parts" / "part_000.obj")


def test_sample_surface_mesh_with_coacd_path(tmp_path: Path):
    obj_dir = tmp_path / "MSO" / "obj_a"
    _make_mesh_object(obj_dir)

    manifest = {
        "dataset": "MSO",
        "summary": {"default_mass_kg": 0.1},
        "objects": [
            {
                "object_id": "obj_a",
                "name": "obj_a",
                "mesh_path": str((obj_dir / "raw.obj").as_posix()),
                "mass_kg": 0.1,
                "principal_moments": [1e-4, 2e-4, 3e-4],
                "process_status": "success",
            }
        ],
    }
    (tmp_path / "MSO" / "manifest.process_meshes.json").write_text(json.dumps(manifest), encoding="utf-8")

    ds = DatasetObjects(
        raw_dataset_root=str(tmp_path),
        raw_dataset_name="MSO",
        scales=[0.06],
        objdata_tag="objdata_MSO",
        graspdata_tag="graspdata_YCB_liberhand",
        generated_dataset_root=str(tmp_path / "datasets"),
        rebuild_existing_assets=True,
        verbose=False,
    )
    info = ds.get_obj_info_by_index(0)

    pts, norms = ds.sample_surface_o3d(info["coacd_abs"], n_points=128, method="poisson")
    assert pts.shape == (128, 3)
    assert norms.shape == (128, 3)


def test_sample_surface_o3d_is_reproducible_after_set_seed(tmp_path: Path):
    pytest.importorskip("open3d")

    obj_dir = tmp_path / "MSO" / "obj_b"
    _make_mesh_object(obj_dir)

    set_seed(0)
    pts1, norms1 = sample_surface_o3d(str(obj_dir / "coacd.obj"), n_points=128, method="poisson")
    set_seed(0)
    pts2, norms2 = sample_surface_o3d(str(obj_dir / "coacd.obj"), n_points=128, method="poisson")

    assert np.array_equal(pts1, pts2)
    assert np.array_equal(norms1, norms2)
