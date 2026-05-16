from pathlib import Path

from src.scale_dataset_builder import ScaleDatasetBuilder


def _make_cube_mesh(path: Path, scale: float = 1.0):
    import trimesh

    m = trimesh.creation.box(extents=[scale, scale, scale])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def _make_visual_assets(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "visual.obj").write_text(
        "\n".join(
            [
                "mtllib textured_visual.mtl",
                "usemtl material_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "textured_visual.mtl").write_text(
        "newmtl material_0\nmap_Kd textured_visual.png\n", encoding="utf-8"
    )
    (path / "textured_visual.png").write_bytes(b"fake-png")


def test_build_scale_assets(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_obj = {
        "object_name": "obj_a",
        "source_dir_abs": str(raw_dir.resolve()),
    }
    _make_cube_mesh(raw_dir / "coacd.obj")
    _make_cube_mesh(raw_dir / "manifold.obj")

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
    shared_root = Path(rec["xml_abs"]).resolve().parent.parent / "meshes_normalized"
    assert (shared_root / "coacd.obj").exists()
    assert (shared_root / "manifold.obj").exists()
    assert len(list((shared_root / "convex_parts").glob("*.obj"))) == 1

    txt = Path(rec["xml_abs"]).read_text(encoding="utf-8")
    assert "<freejoint" in txt
    assert "diaginertia" in txt
    assert 'model="obj_a_scale080"' in txt
    assert 'body name="obj_a_scale080"' in txt
    assert 'mesh name="obj_a_scale080_convex_0"' in txt
    assert 'scale="0.0800000000 0.0800000000 0.0800000000"' in txt
    assert 'name="obj_a_scale080_joint"' in txt
    assert 'contype="1" conaffinity="1" group="3"' in txt


def test_build_scale_assets_with_visual_assets(tmp_path: Path):
    raw_dir = tmp_path / "raw_visual"
    raw_obj = {
        "object_name": "obj_visual",
        "source_dir_abs": str(raw_dir.resolve()),
    }
    _make_cube_mesh(raw_dir / "coacd.obj")
    _make_cube_mesh(raw_dir / "manifold.obj")
    _make_visual_assets(raw_dir)

    b = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    rec = b.build_scale_assets(
        config_stem="run_YCB_liberhand",
        object_info=raw_obj,
        scale=0.08,
        mass_kg=0.2,
        principal_moments=[1e-4, 2e-4, 3e-4],
    )

    scale_dir = Path(rec["xml_abs"]).resolve().parent
    raw_root = scale_dir.parent / "meshes"
    norm_root = scale_dir.parent / "meshes_normalized"
    assert (raw_root / "visual.obj").exists()
    assert (raw_root / "textured_visual.mtl").exists()
    assert (raw_root / "textured_visual.png").exists()
    assert (norm_root / "visual.obj").exists()
    assert (norm_root / "textured_visual.mtl").exists()
    assert (norm_root / "textured_visual.png").exists()

    txt = Path(rec["xml_abs"]).read_text(encoding="utf-8")
    assert 'mesh name="obj_visual_scale080_visual_mesh"' in txt
    assert 'file="../meshes_normalized/visual.obj"' in txt
    assert 'texture name="obj_visual_scale080_texture"' in txt
    assert 'file="../meshes_normalized/textured_visual.png"' in txt
    assert 'geom name="obj_visual_scale080_visual_geom"' in txt
    assert 'contype="0" conaffinity="0" group="0"' in txt
    assert 'material="obj_visual_scale080_material"' in txt


def test_build_multi_scale_assets(tmp_path: Path):
    raw_dir = tmp_path / "raw2"
    raw_obj = {
        "object_name": "obj_b",
        "source_dir_abs": str(raw_dir.resolve()),
    }
    _make_cube_mesh(raw_dir / "coacd.obj")
    _make_cube_mesh(raw_dir / "manifold.obj")

    b = ScaleDatasetBuilder(str(tmp_path / "datasets"))
    out = b.build_multi_scale_assets(
        config_stem="run_YCB_liberhand",
        object_info=raw_obj,
        scales=[0.06, 0.08],
        mass_kg=0.1,
        principal_moments=[1e-5, 2e-5, 3e-5],
    )
    assert set(out.keys()) == {"scale060", "scale080"}
