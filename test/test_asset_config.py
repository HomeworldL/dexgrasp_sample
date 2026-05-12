import json
from pathlib import Path

from utils.utils_file import load_asset_config, usd_convert_cfg


def test_load_asset_config_without_hand_fields(tmp_path: Path):
    config_path = tmp_path / "assets_demo.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 0,
                "data": {
                    "raw_dataset_name": "DEMO",
                    "raw_dataset_root": "assets/objects/processed",
                    "generated_dataset_root": "datasets",
                    "objdata_tag": "objdata_DEMO",
                    "asset_scales": [0.05, 0.06],
                    "verbose": False,
                },
                "sampling": {
                    "n_points": 128,
                },
                "warp_render": {
                    "gpu_lst": [0],
                    "thread_per_gpu": 1,
                    "output_subdir": "pc_warp",
                    "max_point_num": 128,
                    "save_pc": True,
                    "save_rgb": False,
                    "save_depth": False,
                    "skip_existing": True,
                    "depth_max": 5.0,
                    "tile_width": 64,
                    "tile_height": 64,
                    "n_cols": 1,
                    "n_rows": 1,
                    "z_near": 0.1,
                    "z_far": 10.0,
                    "intrinsics": {
                        "preset": "kinect",
                        "fx": 60.0,
                        "fy": 60.0,
                        "cx": 32.0,
                        "cy": 32.0,
                    },
                    "camera": {
                        "type": "spherical",
                        "radius": 0.6,
                        "pos_noise": 0.0,
                        "lookat": [0.0, 0.0, 0.0],
                        "lookat_noise": 0.0,
                        "up": None,
                        "up_noise": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_asset_config(str(config_path))

    assert cfg["data"]["objdata_tag"] == "objdata_DEMO"
    assert cfg["data"]["asset_scales"] == [0.05, 0.06]


def test_usd_convert_cfg_reads_existing_usd_convert_block():
    cfg = {
        "usd_convert": {
            "backend": "urdf",
            "force": True,
            "fix_base": False,
            "import_sites": False,
            "verify_inertial": True,
            "merge_joints": False,
            "make_instanceable": False,
            "convex_decompose_mesh": True,
        }
    }

    out = usd_convert_cfg(cfg)

    assert out["backend"] == "urdf"
    assert out["force"] is True
