import argparse
import os
import time

import numpy as np

from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config
from utils.utils_vis import visualize_with_viser


def main():
    p = argparse.ArgumentParser(description="Visualize hand-object for one object-scale entry by global id or key.")
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
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag_from_config(args.config),
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )

    if args.obj_key:
        info = ds.get_obj_info_by_scale_key(args.obj_key)
    else:
        obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
        info = ds.get_obj_info_by_index(obj_id)
    obj_name = info["object_name"]
    print(f"[vis_ho] id={info['global_id']} name={obj_name} scale={info['scale']}")

    hand_xml = os.path.abspath(cfg["hand"]["xml_path"])
    target_body_params = cfg["hand"]["target_body_params"]
    env = MjHO({"name": obj_name, "xml_abs": info["mjcf_abs"]}, hand_xml, target_body_params=target_body_params, object_fixed=False)

    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float32)
    hand_qpos = env.get_hand_qpos().copy()
    if hand_qpos.shape[0] >= 7 + prepared_joints.shape[0]:
        hand_qpos[7 : 7 + prepared_joints.shape[0]] = prepared_joints
        env.set_hand_qpos(hand_qpos)

    rk = RobotKinematics(hand_xml)
    hand_qpos = env.get_hand_qpos()
    hand_vis = rk.get_posed_meshes(hand_qpos, kind="visual")
    hand_col = rk.get_posed_meshes(hand_qpos, kind="collision")
    obj_mesh = ds.load_mesh(info["coacd_abs"])

    meshes_for_vis = {"object": obj_mesh}
    if hand_vis is not None:
        meshes_for_vis["hand_visual"] = hand_vis
    if hand_col is not None:
        meshes_for_vis["hand_collision"] = hand_col

    _server = visualize_with_viser(meshes=meshes_for_vis, pointclouds={})
    print("[vis_ho] Viser started. Press Enter to open MuJoCo paused view...")
    input()

    env.open_viewer()
    print("[vis_ho] MuJoCo viewer opened. Ctrl+C to quit.")
    try:
        while env._viewer_alive():
            env._render_viewer()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
