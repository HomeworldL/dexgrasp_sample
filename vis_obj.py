import argparse
import os
import time
from pathlib import Path

import numpy as np
import trimesh

from src.dataset_objects import DatasetObjects, resolve_dataset_root
from src.mj_ho import MjHO
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_vis import visualize_with_viser


def main():
    p = argparse.ArgumentParser(description="Visualize one object under selected fixed scale.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    p.add_argument("-i", "--obj-id", type=int, default=None)
    p.add_argument("--scale-index", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    config_stem = Path(args.config).stem
    ds = DatasetObjects(
        resolve_dataset_root(cfg["dataset"].get("root")),
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=config_stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        prebuild_scales=True,
        object_mass_kg=float(cfg["dataset"]["object_mass_kg"]),
    )

    obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
    obj_name = ds.id2name[obj_id]
    info = ds.get_info(obj_id)

    scale_keys = sorted(info.get("scale_assets", {}).keys())
    if not scale_keys:
        raise RuntimeError(f"No scale_assets found for object {obj_name}")
    scale_key = scale_keys[int(args.scale_index) % len(scale_keys)]
    asset = info["scale_assets"][scale_key]

    print(f"[vis_obj] id={info['global_id']} name={obj_name} scale_key={scale_key}")

    parts = [ds.load_mesh(p) for p in asset["convex_parts_abs"]]
    meshes_for_vis = {
        "obj_coacd": ds.load_mesh(asset["coacd_abs"]),
        "obj_convex_parts": trimesh.util.concatenate(parts),
    }
    pts, _ = ds.sample_surface_o3d(
        asset["coacd_abs"],
        n_points=min(2048, int(cfg.get("sampling", {}).get("n_points", 2048))),
        method="poisson",
    )
    cols = np.tile(np.array([[90, 160, 255]], dtype=np.uint8), (pts.shape[0], 1))

    _server = visualize_with_viser(
        meshes=meshes_for_vis,
        pointclouds={"obj_pointcloud": (pts, cols)},
    )
    print("[vis_obj] Viser started. Press Enter to open MuJoCo paused view...")
    input()

    hand_xml = os.path.abspath(cfg["hand"]["xml_path"])
    env = MjHO({"name": obj_name, "xml_abs": asset["xml_abs"], "scale": 1.0}, hand_xml, object_fixed=False, visualize=True)
    print("[vis_obj] MuJoCo viewer opened. Ctrl+C to quit.")
    try:
        while env._viewer_alive():
            env._render_viewer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
