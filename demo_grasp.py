import argparse
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np

from src.mj_ho import MjHO
from src.sample import downsample_fps
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    hand_profile_from_config,
    load_config,
    object_profile_from_config,
    anchor_params_from_config,
)
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_sample import (
    as_array,
    build_pose_candidates,
    make_qpos_triplets,
    parse_object_scale_key,
    sample_frames_from_points,
)
from utils.utils_seed import set_seed


def _hold_viewer(mjho: MjHO) -> None:
    while mjho._viewer_alive():
        mjho._render_viewer()
        time.sleep(1.0 / 60.0)


def run_demo_grasp(
    cfg: Dict,
    object_scale_key: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
    total_stage_start: float,
) -> int:
    object_name, _ = parse_object_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    anchor_params = anchor_params_from_config(cfg)
    hand_profile = hand_profile_from_config(cfg)
    object_profile = object_profile_from_config(cfg)

    mjho = MjHO(
        obj_info,
        hand_xml_path,
        anchor_params=anchor_params,
        hand_profile=hand_profile,
        object_profile=object_profile,
        visualize=True,
    )
    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    ts = time.time()
    transforms_np = sample_frames_from_points(cfg, points, normals)
    if verbose:
        print(
            f"[{object_scale_key}] frame sampling time: {time.time() - ts:.3f}s, "
            f"N={len(transforms_np)}"
        )

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    max_cap = int(cfg["data"]["max_cap"])
    max_time_sec = float(cfg["data"]["max_time_sec"])
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    sim_grasp_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)

    num_no_col = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    stop_reason = "depleted"

    for i in range(qpos_prepared.shape[0]):
        if num_valid >= max_cap:
            num_samples = i
            stop_reason = "cap"
            break
        if (time.perf_counter() - total_stage_start) >= max_time_sec:
            num_samples = i
            stop_reason = "timeout"
            break

        mjho.set_hand_qpos(qpos_prepared[i])
        if mjho.is_contact():
            continue
        mjho.set_hand_qpos(qpos_approach[i])
        if mjho.is_contact():
            continue
        mjho.set_hand_qpos(qpos_init[i])
        if mjho.is_contact():
            continue

        num_no_col += 1
        mjho.set_hand_qpos(qpos_prepared[i])
        qpos_grasp, _ = mjho.sim_grasp(visualize=True, **sim_grasp_cfg)
        qpos_grasp = as_array(qpos_grasp)
        ho_contact_num = mjho.get_contact_num(obj_margin=0.00)
        if ho_contact_num < contact_min_count:
            continue

        num_valid += 1
        print(
            f"[{object_scale_key}] valid_grasp={num_valid} candidate={i} "
            f"contact_count={ho_contact_num}"
        )
        mjho.set_hand_qpos(qpos_grasp)
        mjho._render_viewer()

    print(
        f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
        f"valid={num_valid} stop_reason={stop_reason}"
    )
    _hold_viewer(mjho)
    return num_valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual demo of grasp sampling and sim_grasp without extforce."
    )
    parser.add_argument(
        "--object-scale-key",
        type=str,
        required=True,
        help="Unique object-scale key.",
    )
    parser.add_argument(
        "--coacd-path",
        type=str,
        required=True,
        help="Path to scaled COACD mesh OBJ.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=str,
        required=True,
        help="Path to scaled object MJCF.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logs.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="JSON config path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    total_stage_start = time.perf_counter()

    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    n_points = int(cfg["sampling"]["n_points"])
    points, normals = sample_surface_o3d(
        args.coacd_path,
        n_points=n_points,
        method="poisson",
    )
    run_demo_grasp(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        points=points,
        normals=normals,
        verbose=verbose,
        total_stage_start=total_stage_start,
    )


if __name__ == "__main__":
    main()
