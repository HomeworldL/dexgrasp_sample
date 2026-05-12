"""Inspect one scaled object asset and its convex decomposition.

It displays coacd mesh, merged convex parts, and sampled surface points in
viser, then opens a MuJoCo scene with the same object for sanity checks.
"""

import argparse
import os
import time

import numpy as np
import trimesh

from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_generated_dataset_root_cfg,
    data_run_scales_cfg,
    data_use_native_asset_cfg,
    data_verbose_cfg,
    graspdata_tag_cfg,
    hand_profile_cfg,
    load_run_config,
    object_profile_cfg,
    objdata_tag_cfg,
)
from utils.utils_vis import visualize_with_viser


def main():
    p = argparse.ArgumentParser(
        description="Visualize one object-scale entry by global id or key."
    )
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    p.add_argument("-i", "--obj-id", type=int, default=None)
    p.add_argument(
        "-k",
        "--obj-key",
        type=str,
        default=None,
        help="Object-scale key, e.g. 'YCB_002_master_chef_can__scale080'.",
    )
    args = p.parse_args()

    cfg = load_run_config(args.config)
    ds = DatasetObjects(
        scales=data_run_scales_cfg(cfg),
        objdata_tag=objdata_tag_cfg(cfg, args.config),
        include_native=data_use_native_asset_cfg(cfg),
        graspdata_tag=graspdata_tag_cfg(cfg, args.config),
        generated_dataset_root=data_generated_dataset_root_cfg(cfg),
        verbose=data_verbose_cfg(cfg),
    )

    if args.obj_key:
        info = ds.get_obj_info_by_scale_key(args.obj_key)
    elif args.obj_id is not None:
        info = ds.get_obj_info_by_index(int(args.obj_id))
    else:
        raise ValueError("vis_obj requires either --obj-id or --obj-key.")
    obj_name = info["object_name"]
    print(f"[vis_obj] id={info['global_id']} name={obj_name} scale={info['scale']}")
    print(f"[vis_obj] coacd_abs={info['coacd_abs']}")
    print(f"[vis_obj] mjcf_abs={info['mjcf_abs']}")

    meshes_for_vis = {
        "obj_coacd": ds.load_entry_mesh(info, kind="coacd", apply_scale=True),
        "obj_convex_parts": ds.load_entry_mesh(
            info, kind="convex_parts", apply_scale=True
        ),
    }

    pts, _ = ds.sample_surface_for_entry(
        info,
        n_points=min(2048, int(cfg.get("sampling", {}).get("n_points", 2048))),
        method="poisson",
    )
    cols = np.tile(np.array([[90, 160, 255]], dtype=np.uint8), (pts.shape[0], 1))

    _server = visualize_with_viser(
        meshes=meshes_for_vis, pointclouds={"obj_pointcloud": (pts, cols)}
    )
    print("[vis_obj] Viser started. Press Enter to open MuJoCo paused view...")
    input()

    hand_xml = os.path.abspath(cfg["hand"]["xml_path"])
    env = MjHO(
        {"name": obj_name, "xml_abs": info["mjcf_abs"], "scale": 1.0},
        hand_xml,
        hand_profile=hand_profile_cfg(cfg),
        object_profile=object_profile_cfg(cfg),
        object_fixed=False,
        visualize=True,
    )
    print("[vis_obj] MuJoCo viewer opened. Ctrl+C to quit.")
    try:
        while env._viewer_alive():
            env._render_viewer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
