from pathlib import Path

import trimesh

from src.dataset_objects import DatasetObjects


def _make_mesh_object(obj_dir: Path):
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "convex_parts").mkdir(parents=True, exist_ok=True)
    m = trimesh.creation.icosphere(radius=0.5)
    m.export(obj_dir / "coacd.obj")
    m.export(obj_dir / "convex_parts" / "part_000.obj")


def test_sample_surface_mesh_with_coacd_path(tmp_path: Path):
    obj_dir = tmp_path / "MSO" / "obj_a"
    _make_mesh_object(obj_dir)

    ds = DatasetObjects(str(tmp_path), dataset_names=["MSO"], prebuild_scales=False)
    info = ds.get_info("obj_a")

    pts, norms = ds.sample_surface_mesh(info["coacd_abs"], n_points=128, method="even")
    assert pts.shape == (128, 3)
    assert norms.shape == (128, 3)
