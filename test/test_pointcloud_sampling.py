from pathlib import Path
import json

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
        dataset_root=str(tmp_path),
        dataset_names=["MSO"],
        scales=[0.06],
        dataset_tag="run_YCB_liberhand",
        dataset_output_root=str(tmp_path / "datasets"),
        verbose=False,
    )
    info = ds.get_info(0)

    pts, norms = ds.sample_surface_o3d(info["coacd_abs"], n_points=128, method="poisson")
    assert pts.shape == (128, 3)
    assert norms.shape == (128, 3)
