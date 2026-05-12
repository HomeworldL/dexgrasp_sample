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
    hand_anchor_params_cfg,
    hand_profile_cfg,
    hand_root_stabilization_cfg,
    load_run_config,
    object_profile_cfg,
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


def run_demo(
    cfg: Dict,
    object_scale_key: str,
    hand_name: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
    total_stage_start: float,
) -> bool:
    object_name, _ = parse_object_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    anchor_params = hand_anchor_params_cfg(cfg)
    hand_profile = hand_profile_cfg(cfg)
    object_profile = object_profile_cfg(cfg)
    root_stabilization = hand_root_stabilization_cfg(cfg)

    mjho = MjHO(
        obj_info,
        hand_xml_path,
        anchor_params=anchor_params,
        hand_profile=hand_profile,
        object_profile=object_profile,
        root_stabilization=root_stabilization,
        visualize=False,
    )
    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=int(cfg["seed"]),
    )
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    mjho_valid = MjHO(
        obj_info,
        hand_xml_path,
        anchor_params=anchor_params,
        hand_profile=hand_profile,
        object_profile=object_profile,
        root_stabilization=root_stabilization,
        object_fixed=False,
        visualize=True,
    )

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
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_sim_cfg = dict(extforce_cfg)
    sim_grasp_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    extforce_sim_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("grip_delta", None)

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
        ho_contact_num = mjho.get_contact_num(obj_margin=0.00)
        if ho_contact_num < contact_min_count:
            continue

        qpos_grasp = as_array(qpos_grasp)
        qpos_squeeze = mjho_valid.build_squeeze_qpos(
            qpos_grasp,
            grip_delta=float(extforce_cfg.get("grip_delta", 0.0)),
        )
        qpos_prepared_valid = mjho_valid.build_pregrasp_qpos(
            qpos_squeeze,
            qpos_prepared[i][7:],
        )
        qpos_squeeze = as_array(qpos_squeeze)
        qpos_prepared_valid = as_array(qpos_prepared_valid)
        success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
            qpos_squeeze.copy(),
            qpos_prepared_valid.copy(),
            visualize=True,
            **extforce_sim_cfg,
        )
        if not success:
            continue

        num_valid += 1
        stop_reason = "first_success"
        print(
            f"[{object_scale_key}] success at candidate={i} "
            f"contact_count={ho_contact_num} pos_delta={pos_delta:.6f} "
            f"angle_delta={angle_delta:.6f}"
        )
        print(
            f"[{object_scale_key}] samples_checked={i + 1} no_col={num_no_col} "
            f"valid={num_valid} stop_reason={stop_reason}"
        )
        _hold_viewer(mjho_valid)
        return True

    print(
        f"[{object_scale_key}] samples={num_samples} no_col={num_no_col} "
        f"valid={num_valid} stop_reason={stop_reason}"
    )
    if mjho_valid._viewer_alive():
        _hold_viewer(mjho_valid)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual demo of the run.py grasp pipeline, stopping at first extforce success."
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

    cfg = load_run_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    total_stage_start = time.perf_counter()

    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = Path(hand_xml_path).stem
    n_points = int(cfg["sampling"]["n_points"])
    _, parsed_scale = parse_object_scale_key(args.object_scale_key)
    mesh_scale = 1.0 if parsed_scale is None else float(parsed_scale)
    pts, norms = sample_surface_o3d(
        args.coacd_path,
        n_points=n_points,
        method="poisson",
        scale=mesh_scale,
    )
    run_demo(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_name=hand_name,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        points=pts,
        normals=norms,
        verbose=verbose,
        total_stage_start=total_stage_start,
    )


if __name__ == "__main__":
    main()
