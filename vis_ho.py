import argparse
import os
import time

import numpy as np

from src.dataset_objects import DatasetObjects, resolve_dataset_root
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config


def main():
    p = argparse.ArgumentParser(description="Visualize hand-object pair by global object id.")
    p.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
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
        f"[vis_ho] id={obj_info['global_id']} name={obj_name} dataset={obj_info['dataset']} scale={obj_info['scale']:.4f}"
    )

    xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    if not obj_info.get("xml_abs"):
        print("[vis_ho] Object has no xml_abs; cannot build MuJoCo hand-object scene.")
        return

    target_body_params = cfg["hand"]["target_body_params"]
    env = MjHO(obj_info, xml_path, target_body_params=target_body_params, object_fixed=False)

    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float32)
    hand_qpos = env.get_hand_qpos().copy()
    if hand_qpos.shape[0] >= 7 + prepared_joints.shape[0]:
        hand_qpos[7 : 7 + prepared_joints.shape[0]] = prepared_joints
        env.set_hand_qpos(hand_qpos)

    rk = RobotKinematics(xml_path)
    hand_qpos = env.get_hand_qpos()
    hand_vis = rk.get_posed_meshes(hand_qpos, kind="visual")
    hand_col = rk.get_posed_meshes(hand_qpos, kind="collision")
    obj_mesh = ds.get_mesh(obj_id, "inertia", alpha=0.6)
    if abs(float(obj_info["scale"]) - 1.0) > 1e-12:
        obj_mesh = obj_mesh.copy()
        obj_mesh.apply_scale(float(obj_info["scale"]))

    meshes_for_vis = {"object": obj_mesh}
    if hand_vis is not None:
        meshes_for_vis["hand_visual"] = hand_vis
    if hand_col is not None:
        meshes_for_vis["hand_collision"] = hand_col

    _server = visualize_with_viser(meshes=meshes_for_vis, pointclouds={})
    print("[vis_ho] Viser started (default non-blocking). Press Enter to open MuJoCo paused view...")
    input()

    env.open_viewer()
    print("[vis_ho] MuJoCo viewer opened. Scene is paused (no mj_step). Ctrl+C to quit.")
    try:
        while env._viewer_alive():
            env._render_viewer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
