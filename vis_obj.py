import argparse
import os
import time

from src.dataset_objects import DatasetObjects, resolve_dataset_root
from src.mj_ho import MjHO
from utils.utils_vis import visualize_with_viser
from utils.utils_file import load_config


def main():
    p = argparse.ArgumentParser(description="Visualize object meshes and scaled MuJoCo object by global id.")
    p.add_argument("-c", "--config", type=str, default="configs/run_default.json")
    p.add_argument("-i", "--obj-id", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = DatasetObjects(
        resolve_dataset_root(cfg["dataset"].get("root")),
        dataset_names=list(cfg["dataset"].get("include", [])),
        shapenet_scale_range=tuple(cfg["dataset"].get("shapenet_scale_range", [0.06, 0.15])),
        shapenet_scale_seed=int(cfg["seed"]),
    )

    obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
    obj_name = ds.id2name[obj_id]
    obj_info = ds.get_info(obj_id)

    print(
        f"[vis_obj] id={obj_info['global_id']} name={obj_name} dataset={obj_info['dataset']} scale={obj_info['scale']:.4f}"
    )
    print(obj_info)

    meshes_for_vis = {}
    for mt in ("inertia", "visual", "manifold", "coacd", "convex_parts"):
        try:
            m = ds.get_mesh(obj_id, mt, alpha=0.65)
            if abs(float(obj_info["scale"]) - 1.0) > 1e-12:
                m = m.copy()
                m.apply_scale(float(obj_info["scale"]))
            meshes_for_vis[f"obj_{mt}"] = m
        except Exception:
            continue

    _server = visualize_with_viser(meshes=meshes_for_vis, pointclouds={})
    print("[vis_obj] Viser started (default non-blocking). Press Enter to open MuJoCo paused view...")
    input()

    xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    if not obj_info.get("xml_abs"):
        print("[vis_obj] Object has no xml_abs; skip MuJoCo stage.")
        return

    env = MjHO(obj_info, xml_path, object_fixed=False, visualize=True)
    print("[vis_obj] MuJoCo viewer opened. Scene is paused (no mj_step). Ctrl+C to quit.")
    try:
        while env._viewer_alive():
            env._render_viewer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
