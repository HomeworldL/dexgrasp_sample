"""Visualize one hand-object setup for a selected object-scale entry.

The script samples candidate grasp poses from object surface points, picks one
random prepared qpos, previews object and hand meshes in viser, then opens
MuJoCo viewer for interactive checking.
"""

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from src.sample import sample_grasp_frames
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    anchor_params_from_config,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    graspdata_tag_from_config,
    hand_profile_from_config,
    load_config,
    object_profile_from_config,
    objdata_tag_from_config,
    run_scales_from_config,
    use_native_asset_from_config,
)
from utils.utils_seed import set_seed
from utils.utils_sample import build_pose_candidates, make_qpos_triplets
from utils.utils_vis import visualize_with_viser


def _sample_qpos_prepared(
    cfg: Dict, ds: DatasetObjects, object_mesh_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    sampling_cfg = cfg["sampling"]
    points, normals = ds.sample_surface_o3d(
        object_mesh_path,
        n_points=int(sampling_cfg["n_points"]),
        method="poisson",
    )
    transforms = sample_grasp_frames(
        points,
        normals,
        d_min=float(sampling_cfg["d_min"]),
        d_max=float(sampling_cfg["d_max"]),
        Nd=int(sampling_cfg["Nd"]),
        rot_n=int(sampling_cfg["rot_n"]),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=sampling_cfg.get("max_points"),
    )
    transforms_np = transforms.detach().cpu().numpy().astype(np.float32, copy=False)
    del transforms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pose = build_pose_candidates(cfg, transforms_np)
    _, _, qpos_prepared = make_qpos_triplets(cfg, pose)
    if qpos_prepared.shape[0] <= 0:
        raise RuntimeError("No sampled prepared qpos generated.")
    return qpos_prepared, transforms_np


def _transform_to_pose_wxyz(transform: np.ndarray) -> np.ndarray:
    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) transform, got {mat.shape}.")
    pos = mat[:3, 3].astype(np.float32)
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([pos, quat_wxyz], axis=0).astype(np.float32)


def main() -> None:
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
    seed = int(cfg["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ds = DatasetObjects(
        scales=run_scales_from_config(cfg),
        objdata_tag=objdata_tag_from_config(cfg, args.config),
        include_native=use_native_asset_from_config(cfg),
        graspdata_tag=graspdata_tag_from_config(cfg, args.config),
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        verbose=data_verbose_from_config(cfg),
    )

    if args.obj_key:
        info = ds.get_obj_info_by_scale_key(args.obj_key)
    elif args.obj_id is not None:
        info = ds.get_obj_info_by_index(int(args.obj_id))
    else:
        raise ValueError("vis_ho requires either --obj-id or --obj-key.")
    obj_name = info["object_name"]
    print(f"[vis_ho] id={info['global_id']} name={obj_name} scale={info['scale']}")

    hand_xml = os.path.abspath(cfg["hand"]["xml_path"])
    env = MjHO(
        {"name": obj_name, "xml_abs": info["mjcf_abs"]},
        hand_xml,
        anchor_params=anchor_params_from_config(cfg),
        hand_profile=hand_profile_from_config(cfg),
        object_profile=object_profile_from_config(cfg),
        object_fixed=False,
    )
    qpos_prepared, transforms_np = _sample_qpos_prepared(cfg, ds, info["coacd_abs"])
    chosen_idx = int(rng.integers(0, qpos_prepared.shape[0]))
    env.set_hand_qpos(qpos_prepared[chosen_idx])
    print(f"[vis_ho] seed={seed} sampled_prepared={qpos_prepared.shape[0]} selected={chosen_idx}")
    hand_root_pose = qpos_prepared[chosen_idx, :7].astype(np.float32, copy=False)
    grasp_pose = _transform_to_pose_wxyz(transforms_np[chosen_idx])

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
    _server.scene.add_frame(
        "hand_root_frame",
        position=tuple(float(v) for v in hand_root_pose[:3]),
        wxyz=tuple(float(v) for v in hand_root_pose[3:7]),
        axes_length=0.04,
        axes_radius=0.0025,
    )
    _server.scene.add_frame(
        "grasp_frame",
        position=tuple(float(v) for v in grasp_pose[:3]),
        wxyz=tuple(float(v) for v in grasp_pose[3:7]),
        axes_length=0.04,
        axes_radius=0.0025,
    )
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
