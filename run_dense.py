import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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
from utils.utils_sample import (
    ARRAY_DTYPE,
    H5_DTYPE,
    as_array,
    build_pose_candidates,
    encode_h5_str,
    grasp_outputs_exist,
    make_qpos_triplets,
    parse_object_scale_key,
    sample_frames_from_points,
    write_fail_npy_from_h5,
    write_grasp_npy_from_h5,
)
from utils.utils_seed import set_seed, stable_seed


def _select_fail_samples(
    fail_qpos_rows: List[np.ndarray],
    fail_stages: List[str],
    valid_count: int,
    min_valid_count: int,
    fail_keep_ratio: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[str]]:
    if valid_count < min_valid_count:
        return [], []
    keep_count = min(
        len(fail_qpos_rows),
        int(np.floor(float(fail_keep_ratio) * float(valid_count))),
    )
    if keep_count <= 0:
        return [], []
    indices = np.random.default_rng(int(seed)).permutation(len(fail_qpos_rows))[
        :keep_count
    ]
    selected_qpos = [fail_qpos_rows[int(idx)] for idx in indices]
    selected_stages = [fail_stages[int(idx)] for idx in indices]
    return selected_qpos, selected_stages


def _write_fail_h5(
    fail_h5_path: Path,
    object_name: str,
    scale: Optional[float],
    hand_name: str,
    qpos_dim: int,
    qpos_fail: List[np.ndarray],
    failure_stages: List[str],
) -> None:
    if qpos_fail:
        qpos_fail_np = np.asarray(qpos_fail, dtype=ARRAY_DTYPE)
        failure_stage_np = np.asarray(failure_stages, dtype=object)
    else:
        qpos_fail_np = np.zeros((0, qpos_dim), dtype=ARRAY_DTYPE)
        failure_stage_np = np.asarray([], dtype=object)
    failure_stage_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(fail_h5_path, "w") as hf:
        hf.create_dataset("object_name", data=encode_h5_str(object_name))
        hf.create_dataset(
            "scale", data=np.float32(scale if scale is not None else np.nan)
        )
        hf.create_dataset("hand_name", data=encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))
        hf.create_dataset("qpos_fail", data=qpos_fail_np, dtype=H5_DTYPE)
        hf.create_dataset(
            "failure_stage",
            data=failure_stage_np,
            dtype=failure_stage_dtype,
        )


def _require_densify_cfg(cfg: Dict) -> Dict:
    dense_cfg = cfg.get("densify")
    if not isinstance(dense_cfg, dict):
        raise KeyError("Missing required config field: densify")

    required = [
        "enable",
        "seed_valid_cap",
        "dense_valid_cap",
        "candidates_per_seed",
        "max_per_seed",
        "translation_sigma",
        "rotation_sigma_deg",
        "dedup_pos_threshold",
        "dedup_angle_threshold_deg",
    ]
    for key in required:
        if key not in dense_cfg:
            raise KeyError(f"Missing required config field: densify.{key}")

    if not isinstance(dense_cfg["enable"], bool):
        raise ValueError("densify.enable must be a boolean.")
    for key in [
        "seed_valid_cap",
        "dense_valid_cap",
        "candidates_per_seed",
        "max_per_seed",
    ]:
        value = int(dense_cfg[key])
        if value <= 0:
            raise ValueError(f"densify.{key} must be > 0.")
    translation_sigma = np.asarray(dense_cfg["translation_sigma"], dtype=np.float64)
    rotation_sigma_deg = np.asarray(dense_cfg["rotation_sigma_deg"], dtype=np.float64)
    if translation_sigma.shape != (3,):
        raise ValueError("densify.translation_sigma must have shape [3].")
    if rotation_sigma_deg.shape != (3,):
        raise ValueError("densify.rotation_sigma_deg must have shape [3].")
    if np.any(translation_sigma < 0.0):
        raise ValueError("densify.translation_sigma values must be >= 0.")
    if np.any(rotation_sigma_deg < 0.0):
        raise ValueError("densify.rotation_sigma_deg values must be >= 0.")
    for key in ["dedup_pos_threshold", "dedup_angle_threshold_deg"]:
        value = float(dense_cfg[key])
        if value < 0.0:
            raise ValueError(f"densify.{key} must be >= 0.")
    return dense_cfg


def _sample_local_pose(
    seed_pose: np.ndarray,
    rng: np.random.Generator,
    translation_sigma: np.ndarray,
    rotation_sigma_deg: np.ndarray,
) -> np.ndarray:
    seed_pose = np.asarray(seed_pose, dtype=np.float64).reshape(7)
    seed_quat_wxyz = seed_pose[3:7]
    quat_norm = np.linalg.norm(seed_quat_wxyz)
    if not np.isfinite(quat_norm) or quat_norm <= 1e-12:
        raise ValueError("Dense seed pose has invalid quaternion.")

    base_rot = R.from_quat((seed_quat_wxyz / quat_norm)[[1, 2, 3, 0]])
    delta_local = rng.normal(loc=0.0, scale=translation_sigma, size=3)
    delta_rot = R.from_euler(
        "xyz",
        rng.normal(loc=0.0, scale=np.deg2rad(rotation_sigma_deg), size=3),
    )

    pos = seed_pose[:3] + base_rot.apply(delta_local)
    quat_xyzw = (base_rot * delta_rot).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    return np.concatenate([pos, quat_wxyz]).astype(ARRAY_DTYPE)


def _quat_angle_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    quat_a = np.asarray(pose_a[3:7], dtype=np.float64)
    quat_b = np.asarray(pose_b[3:7], dtype=np.float64)
    quat_a /= np.linalg.norm(quat_a)
    quat_b /= np.linalg.norm(quat_b)
    rot_a = R.from_quat(quat_a[[1, 2, 3, 0]])
    rot_b = R.from_quat(quat_b[[1, 2, 3, 0]])
    return float((rot_a.inv() * rot_b).magnitude())


def _is_duplicate_pose(
    pose: np.ndarray,
    accepted_poses: List[np.ndarray],
    pos_threshold: float,
    angle_threshold: float,
) -> bool:
    if pos_threshold <= 0.0 and angle_threshold <= 0.0:
        return False
    for accepted_pose in accepted_poses:
        pos_dist = float(np.linalg.norm(pose[:3] - accepted_pose[:3]))
        if pos_dist > pos_threshold:
            continue
        if _quat_angle_distance(pose, accepted_pose) <= angle_threshold:
            return True
    return False


def _evaluate_candidate(
    mjho: MjHO,
    mjho_valid: MjHO,
    qpos_init: np.ndarray,
    qpos_approach: np.ndarray,
    qpos_prepared: np.ndarray,
    sim_grasp_cfg: Dict,
    extforce_cfg: Dict,
    extforce_sim_cfg: Dict,
    contact_min_count: int,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    mjho.set_hand_qpos(qpos_prepared)
    if mjho.is_contact():
        return "prepared_contact", as_array(qpos_prepared), None, None
    mjho.set_hand_qpos(qpos_approach)
    if mjho.is_contact():
        return "approach_contact", None, None, None
    mjho.set_hand_qpos(qpos_init)
    if mjho.is_contact():
        return "init_contact", None, None, None

    mjho.set_hand_qpos(qpos_prepared)
    qpos_grasp, _ = mjho.sim_grasp(visualize=False, **sim_grasp_cfg)
    ho_contact_num = mjho.get_contact_num(obj_margin=0.00)
    qpos_grasp = as_array(qpos_grasp)

    if ho_contact_num < contact_min_count:
        return "insufficient_contact", qpos_grasp, None, None

    qpos_squeeze = mjho_valid.build_squeeze_qpos(
        qpos_grasp,
        grip_delta=float(extforce_cfg.get("grip_delta", 0.0)),
    )
    qpos_prepared_valid = mjho_valid.build_pregrasp_qpos(
        qpos_squeeze,
        qpos_prepared[7:],
    )
    qpos_squeeze = as_array(qpos_squeeze)
    qpos_prepared_valid = as_array(qpos_prepared_valid)
    is_valid, _, _ = mjho_valid.sim_under_extforce(
        qpos_squeeze.copy(),
        qpos_prepared_valid.copy(),
        visualize=False,
        **extforce_sim_cfg,
    )
    if not is_valid:
        return (
            "extforce_failure",
            qpos_squeeze.astype(ARRAY_DTYPE, copy=False),
            None,
            None,
        )
    return "valid", None, qpos_grasp, qpos_squeeze


def run_sampling(
    cfg: Dict,
    object_scale_key: str,
    hand_name: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    output_dir_abs: str,
    points: np.ndarray,
    normals: np.ndarray,
    verbose: bool,
    total_stage_start: float,
) -> str:
    object_name, parsed_scale = parse_object_scale_key(object_scale_key)
    scale = parsed_scale
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
    )
    sampling_cfg = cfg["sampling"]
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(sampling_cfg["downsample_for_sim"]),
        seed=stable_seed(int(cfg["seed"]), object_scale_key, "downsample_for_sim"),
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
    )

    ts = time.time()
    set_seed(stable_seed(int(cfg["seed"]), object_scale_key, "sample_frames"))
    transforms_np = sample_frames_from_points(cfg, points, normals)
    if verbose:
        print(
            f"[{object_scale_key}] frame sampling time: {time.time() - ts:.3f}s, N={len(transforms_np)}"
        )

    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    out_dir = Path(output_dir_abs)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / str(cfg["data"]["h5_name"])
    npy_path = out_dir / str(cfg["data"]["npy_name"])
    fail_h5_path = out_dir / str(cfg["data"]["fail_h5_name"])
    fail_npy_path = out_dir / str(cfg["data"]["fail_npy_name"])

    d = qpos_prepared.shape[1]
    max_cap = int(cfg["data"]["max_cap"])
    max_time_sec = float(cfg["data"]["max_time_sec"])
    min_valid_count = int(cfg["data"]["min_valid_count"])
    flush_every = int(cfg["data"]["flush_every"])
    fail_keep_ratio = float(cfg["data"]["fail_keep_ratio"])
    dense_cfg = _require_densify_cfg(cfg)
    densify_enable = bool(dense_cfg["enable"])
    seed_valid_cap = int(dense_cfg["seed_valid_cap"])
    dense_valid_cap = int(dense_cfg["dense_valid_cap"])
    candidates_per_seed = int(dense_cfg["candidates_per_seed"])
    max_per_seed = int(dense_cfg["max_per_seed"])
    translation_sigma = np.asarray(dense_cfg["translation_sigma"], dtype=np.float64)
    rotation_sigma_deg = np.asarray(dense_cfg["rotation_sigma_deg"], dtype=np.float64)
    dedup_pos_threshold = float(dense_cfg["dedup_pos_threshold"])
    dedup_angle_threshold = np.deg2rad(float(dense_cfg["dedup_angle_threshold_deg"]))
    contact_min_count = int(cfg["sim_grasp"]["contact_min_count"])
    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_sim_cfg = dict(extforce_cfg)
    sim_grasp_cfg.pop("visualize", None)
    extforce_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("grip_delta", None)
    sim_grasp_cfg.pop("contact_min_count", None)
    num_no_col = 0
    num_contact_ok = 0
    num_valid = 0
    num_samples = transforms_np.shape[0]
    dense_candidates = 0
    dense_no_col = 0
    dense_contact_ok = 0
    dense_valid = 0
    ts = time.time()
    stop_reason = "depleted"
    fail_qpos_rows: List[np.ndarray] = []
    fail_stages: List[str] = []
    dense_seed_qpos: List[np.ndarray] = []
    accepted_poses: List[np.ndarray] = []

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("object_name", data=encode_h5_str(object_name))
        hf.create_dataset(
            "scale", data=np.float32(scale if scale is not None else np.nan)
        )
        hf.create_dataset("hand_name", data=encode_h5_str(hand_name))
        hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))

        ds_init = hf.create_dataset(
            "qpos_init", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE
        )
        ds_approach = hf.create_dataset(
            "qpos_approach", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE
        )
        ds_prepared = hf.create_dataset(
            "qpos_prepared", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE
        )
        ds_grasp = hf.create_dataset(
            "qpos_grasp", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE
        )
        ds_squeeze = hf.create_dataset(
            "qpos_squeeze", shape=(max_cap, d), maxshape=(None, d), dtype=H5_DTYPE
        )

        for i in tqdm(
            range(qpos_prepared.shape[0]),
            desc=f"sampling-{object_scale_key}",
            miniters=50,
            disable=not verbose,
        ):
            if num_valid >= max_cap:
                num_samples = i
                stop_reason = "cap"
                break
            if densify_enable and num_valid >= seed_valid_cap:
                num_samples = i
                stop_reason = "seed_cap"
                break
            if (time.perf_counter() - total_stage_start) >= max_time_sec:
                num_samples = i
                stop_reason = "timeout"
                break

            status, fail_qpos, qpos_grasp, qpos_squeeze = _evaluate_candidate(
                mjho=mjho,
                mjho_valid=mjho_valid,
                qpos_init=qpos_init[i],
                qpos_approach=qpos_approach[i],
                qpos_prepared=qpos_prepared[i],
                sim_grasp_cfg=sim_grasp_cfg,
                extforce_cfg=extforce_cfg,
                extforce_sim_cfg=extforce_sim_cfg,
                contact_min_count=contact_min_count,
            )
            if status in {
                "insufficient_contact",
                "extforce_failure",
                "valid",
            }:
                num_no_col += 1
            if status in {"extforce_failure", "valid"}:
                num_contact_ok += 1
            if fail_qpos is not None:
                fail_qpos_rows.append(fail_qpos.astype(ARRAY_DTYPE, copy=False))
                fail_stages.append(status)
            if status == "valid":
                if qpos_grasp is None or qpos_squeeze is None:
                    raise RuntimeError("Valid candidate did not return grasp outputs.")
                ds_init[num_valid] = qpos_init[i].astype(ARRAY_DTYPE, copy=False)
                ds_approach[num_valid] = qpos_approach[i].astype(
                    ARRAY_DTYPE, copy=False
                )
                ds_prepared[num_valid] = qpos_prepared[i].astype(
                    ARRAY_DTYPE, copy=False
                )
                ds_grasp[num_valid] = qpos_grasp.astype(ARRAY_DTYPE, copy=False)
                ds_squeeze[num_valid] = qpos_squeeze.astype(ARRAY_DTYPE, copy=False)
                dense_seed_qpos.append(qpos_squeeze.astype(ARRAY_DTYPE, copy=True))
                accepted_poses.append(qpos_squeeze[:7].astype(ARRAY_DTYPE, copy=True))
                num_valid += 1

            if flush_every > 0 and (i + 1) % flush_every == 0:
                hf.flush()

        if densify_enable and stop_reason != "timeout" and dense_seed_qpos:
            rng = np.random.default_rng(
                stable_seed(int(cfg["seed"]), object_scale_key, "densify")
            )
            dense_seed_count = len(dense_seed_qpos)
            dense_accept_counts = np.zeros((dense_seed_count,), dtype=np.int32)

            for candidate_round in tqdm(
                range(candidates_per_seed),
                desc=f"densify-{object_scale_key}",
                miniters=1,
                disable=not verbose,
            ):
                seed_indices = rng.permutation(dense_seed_count)
                for seed_idx in seed_indices:
                    if num_valid >= max_cap:
                        stop_reason = "cap"
                        break
                    if dense_valid >= dense_valid_cap:
                        stop_reason = "dense_cap"
                        break
                    if (time.perf_counter() - total_stage_start) >= max_time_sec:
                        stop_reason = "timeout"
                        break
                    if dense_accept_counts[int(seed_idx)] >= max_per_seed:
                        continue

                    seed_qpos = dense_seed_qpos[int(seed_idx)]
                    dense_pose = _sample_local_pose(
                        seed_qpos[:7],
                        rng=rng,
                        translation_sigma=translation_sigma,
                        rotation_sigma_deg=rotation_sigma_deg,
                    )
                    if _is_duplicate_pose(
                        dense_pose,
                        accepted_poses=accepted_poses,
                        pos_threshold=dedup_pos_threshold,
                        angle_threshold=dedup_angle_threshold,
                    ):
                        continue

                    dense_qpos_init, dense_qpos_approach, dense_qpos_prepared = (
                        make_qpos_triplets(cfg, dense_pose.reshape(1, 7))
                    )
                    dense_candidates += 1
                    status, fail_qpos, qpos_grasp, qpos_squeeze = _evaluate_candidate(
                        mjho=mjho,
                        mjho_valid=mjho_valid,
                        qpos_init=dense_qpos_init[0],
                        qpos_approach=dense_qpos_approach[0],
                        qpos_prepared=dense_qpos_prepared[0],
                        sim_grasp_cfg=sim_grasp_cfg,
                        extforce_cfg=extforce_cfg,
                        extforce_sim_cfg=extforce_sim_cfg,
                        contact_min_count=contact_min_count,
                    )
                    if status in {
                        "insufficient_contact",
                        "extforce_failure",
                        "valid",
                    }:
                        dense_no_col += 1
                    if status in {"extforce_failure", "valid"}:
                        dense_contact_ok += 1
                    if fail_qpos is not None:
                        fail_qpos_rows.append(fail_qpos.astype(ARRAY_DTYPE, copy=False))
                        fail_stages.append(status)
                    if status != "valid":
                        continue
                    if qpos_grasp is None or qpos_squeeze is None:
                        raise RuntimeError("Valid dense candidate missing outputs.")

                    ds_init[num_valid] = dense_qpos_init[0].astype(
                        ARRAY_DTYPE, copy=False
                    )
                    ds_approach[num_valid] = dense_qpos_approach[0].astype(
                        ARRAY_DTYPE, copy=False
                    )
                    ds_prepared[num_valid] = dense_qpos_prepared[0].astype(
                        ARRAY_DTYPE, copy=False
                    )
                    ds_grasp[num_valid] = qpos_grasp.astype(ARRAY_DTYPE, copy=False)
                    ds_squeeze[num_valid] = qpos_squeeze.astype(ARRAY_DTYPE, copy=False)
                    dense_accept_counts[int(seed_idx)] += 1
                    accepted_poses.append(
                        qpos_squeeze[:7].astype(ARRAY_DTYPE, copy=True)
                    )
                    dense_valid += 1
                    num_valid += 1

                    if flush_every > 0 and num_valid % flush_every == 0:
                        hf.flush()

                if stop_reason in {"cap", "dense_cap", "timeout"}:
                    break
            if stop_reason == "seed_cap":
                stop_reason = "dense_depleted"

        final_size = num_valid if num_valid >= min_valid_count else 0
        ds_init.resize((final_size, d))
        ds_approach.resize((final_size, d))
        ds_prepared.resize((final_size, d))
        ds_grasp.resize((final_size, d))
        ds_squeeze.resize((final_size, d))
        hf.flush()

    num_valid = final_size
    kept_fail_qpos, kept_fail_stages = _select_fail_samples(
        fail_qpos_rows=fail_qpos_rows,
        fail_stages=fail_stages,
        valid_count=num_valid,
        min_valid_count=min_valid_count,
        fail_keep_ratio=fail_keep_ratio,
        seed=int(cfg["seed"]),
    )
    _write_fail_h5(
        fail_h5_path=fail_h5_path,
        object_name=object_name,
        scale=scale,
        hand_name=hand_name,
        qpos_dim=d,
        qpos_fail=kept_fail_qpos,
        failure_stages=kept_fail_stages,
    )
    write_fail_npy_from_h5(fail_h5_path, fail_npy_path)

    duration = time.time() - ts
    total_elapsed = time.perf_counter() - total_stage_start
    explore_valid = len(dense_seed_qpos)
    no_col_rate = num_no_col / max(num_samples, 1)
    contact_ok_rate = num_contact_ok / max(num_no_col, 1)
    explore_valid_rate = explore_valid / max(num_contact_ok, 1)
    dense_no_col_rate = dense_no_col / max(dense_candidates, 1)
    dense_contact_ok_rate = dense_contact_ok / max(dense_no_col, 1)
    dense_valid_rate = dense_valid / max(dense_contact_ok, 1)
    print(
        f"[{object_scale_key}] explore_samples={num_samples} "
        f"no_col={num_no_col} ({no_col_rate:.3f}) "
        f"contact_ok={num_contact_ok} ({contact_ok_rate:.3f}) "
        f"valid={explore_valid} ({explore_valid_rate:.3f}) "
        f"dense_candidates={dense_candidates} "
        f"dense_no_col={dense_no_col} ({dense_no_col_rate:.3f}) "
        f"dense_contact_ok={dense_contact_ok} ({dense_contact_ok_rate:.3f}) "
        f"dense_valid={dense_valid} ({dense_valid_rate:.3f}) "
        f"total_valid={num_valid} fail={len(kept_fail_qpos)} "
        f"time={duration:.2f}s total_elapsed={total_elapsed:.2f}s "
        f"stop_reason={stop_reason} out={h5_path}"
    )
    write_grasp_npy_from_h5(h5_path, npy_path)
    print(f"[{object_scale_key}] converted {h5_path.name} -> {npy_path.name}")
    return str(h5_path)


def main():
    p = argparse.ArgumentParser(description="Sample grasps for one object-scale entry.")
    p.add_argument(
        "--object-scale-key", type=str, required=True, help="Unique object-scale key."
    )
    p.add_argument(
        "--mjcf-path", type=str, required=True, help="Path to scaled object MJCF."
    )
    p.add_argument(
        "--global-pc-path",
        type=str,
        required=True,
        help="Path to global_pc.npy prepared by prepare_object_assets.py.",
    )
    p.add_argument(
        "--global-normals-path",
        type=str,
        required=True,
        help="Path to global_normals.npy prepared by prepare_object_assets.py.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for grasp artifacts.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if configured grasp outputs already exist.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="JSON config path.",
    )
    args = p.parse_args()

    cfg = load_run_config(args.config)
    set_seed(int(cfg["seed"]))
    verbose = bool(args.verbose)
    total_stage_start = time.perf_counter()
    if verbose:
        print(f"Using object-scale key: {args.object_scale_key}")
    h5_name = str(cfg["data"]["h5_name"])
    npy_name = str(cfg["data"]["npy_name"])
    has_grasp_outputs = grasp_outputs_exist(
        args.output_dir, h5_name=h5_name, npy_name=npy_name
    )
    if (not args.force) and has_grasp_outputs:
        if verbose:
            print(
                f"[{args.object_scale_key}] skip existing {h5_name} and {npy_name} "
                f"in {args.output_dir}"
            )
        return

    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    hand_name = Path(hand_xml_path).stem
    pts = np.asarray(np.load(args.global_pc_path, allow_pickle=False), dtype=np.float32)
    norms = np.asarray(
        np.load(args.global_normals_path, allow_pickle=False), dtype=np.float32
    )
    if verbose:
        print(
            f"[{args.object_scale_key}] loaded global_pc/global_normals from {args.global_pc_path}"
        )
    run_sampling(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_name=hand_name,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        output_dir_abs=args.output_dir,
        points=pts,
        normals=norms,
        verbose=verbose,
        total_stage_start=total_stage_start,
    )


if __name__ == "__main__":
    main()
