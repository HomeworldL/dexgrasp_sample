import argparse
import os
import time

import numpy as np
import trimesh

from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    graspdata_tag_from_config,
    load_config,
    objdata_tag_from_config,
    raw_dataset_name_from_config,
    raw_dataset_root_from_config,
    run_scales_from_config,
)
from utils.utils_vis import visualize_with_viser


def main():
    p = argparse.ArgumentParser(description="Visualize one object-scale entry by global id or key.")
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

    cfg = load_config(args.config)
    ds = DatasetObjects(
        raw_dataset_root=raw_dataset_root_from_config(cfg),
        raw_dataset_name=raw_dataset_name_from_config(cfg),
        scales=run_scales_from_config(cfg),
        objdata_tag=objdata_tag_from_config(cfg, args.config),
        graspdata_tag=graspdata_tag_from_config(cfg, args.config),
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        verbose=data_verbose_from_config(cfg),
    )

    if args.obj_key:
        info = ds.get_obj_info_by_scale_key(args.obj_key)
    elif args.obj_id is not None:
        info = ds.get_obj_info_by_index(int(args.obj_id))
    else:
        raise ValueError("vis_obj requires either --obj-id or --obj-key.")
    obj_name = info["object_name"]
    print(f"[vis_obj] id={info['global_id']} name={obj_name} scale={info['scale']}")
    print(f"[vis_obj] convex_parts_abs={info['convex_parts_abs']}")
    print(f"[vis_obj] coacd_abs={info['coacd_abs']}")
    print(f"[vis_obj] mjcf_abs={info['mjcf_abs']}")

    parts = [ds.load_mesh(p) for p in info["convex_parts_abs"]]
    meshes_for_vis = {
        "obj_coacd": ds.load_mesh(info["coacd_abs"]),
        "obj_convex_parts": trimesh.util.concatenate(parts),
    }

    pts, _ = ds.sample_surface_o3d(
        info["coacd_abs"],
        n_points=min(2048, int(cfg.get("sampling", {}).get("n_points", 2048))),
        method="poisson",
    )
    cols = np.tile(np.array([[90, 160, 255]], dtype=np.uint8), (pts.shape[0], 1))

    _server = visualize_with_viser(meshes=meshes_for_vis, pointclouds={"obj_pointcloud": (pts, cols)})
    print("[vis_obj] Viser started. Press Enter to open MuJoCo paused view...")
    input()

    hand_xml = os.path.abspath(cfg["hand"]["xml_path"])
    env = MjHO(
        {"name": obj_name, "xml_abs": info["mjcf_abs"], "scale": 1.0},
        hand_xml,
        friction_coef=cfg["hand"]["friction_coef"],
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
