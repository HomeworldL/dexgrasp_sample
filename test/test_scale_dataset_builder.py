from pathlib import Path

from src.scale_dataset_builder import ScaleDatasetBuilder


def _make_cube_mesh(path: Path, scale: float = 1.0):
    import trimesh

    m = trimesh.creation.box(extents=[scale, scale, scale])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def test_build_scale_assets(tmp_path: Path):
    raw_obj = {
        "object_name": "obj_a",
        "coacd_abs": str((tmp_path / "raw" / "coacd.obj").resolve()),
    }
    _make_cube_mesh(Path(raw_obj["coacd_abs"]))

    b = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    rec = b.build_scale_assets(
        config_stem="run_YCB_liberhand",
        object_info=raw_obj,
        scale=0.08,
        mass_kg=0.2,
        principal_moments=[1e-4, 2e-4, 3e-4],
    )

    assert rec["scale_tag"] == "scale080"
    assert Path(rec["coacd_abs"]).exists()
    assert Path(rec["xml_abs"]).exists()
    assert len(rec["convex_parts_abs"]) == 1

    txt = Path(rec["xml_abs"]).read_text(encoding="utf-8")
    assert "<freejoint" in txt
    assert "diaginertia" in txt


def test_build_multi_scale_assets(tmp_path: Path):
    raw_obj = {
        "object_name": "obj_b",
        "coacd_abs": str((tmp_path / "raw2" / "coacd.obj").resolve()),
    }
    _make_cube_mesh(Path(raw_obj["coacd_abs"]))

    b = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    out = b.build_multi_scale_assets(
        config_stem="run_YCB_liberhand",
        object_info=raw_obj,
        scales=[0.06, 0.08],
        mass_kg=0.1,
        principal_moments=[1e-5, 2e-5, 3e-5],
    )
    assert set(out.keys()) == {"scale060", "scale080"}
