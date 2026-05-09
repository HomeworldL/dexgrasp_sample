#!/usr/bin/env python3
"""Solimp/Solref ablation for one object-scale entry.

What it does:
- Generates 4 configs under scripts/configs/solimp_solref_cases
- Runs run.py for each case (optional)
- Summarizes runtime/sampling counts
- Computes penetration depth from contact.dist at:
  1) post sim_grasp (qpos_grasp)
  2) post build_squeeze static state (qpos_squeeze)
  3) sim_under_extforce dynamics (settling peak / force peak / total peak)

Outputs:
- tmp/solimp_solref_experiment/summary.json
- tmp/solimp_solref_experiment/summary.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mj_ho import MjHO
from utils.utils_file import load_config

SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+)\s+no_col=(?P<no_col>\d+)\s+valid=(?P<valid>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+time=(?P<sim_time>[0-9.]+)s\s+total_elapsed=(?P<total_elapsed>[0-9.]+)s\s+"
    r"stop_reason=(?P<stop_reason>\w+)"
)

PC_SUBDIR = "pc_warp"


@dataclass(frozen=True)
class ContactDepthStats:
    contact_pair_count: int
    depth_mean_m: Optional[float]
    depth_p95_m: Optional[float]
    depth_max_m: Optional[float]
    per_grasp_max_mean_m: Optional[float]
    per_grasp_max_p95_m: Optional[float]
    per_grasp_max_max_m: Optional[float]

    def to_dict(self, prefix: str) -> Dict[str, Any]:
        d = {
            f"{prefix}_contact_pair_count": self.contact_pair_count,
            f"{prefix}_depth_mean_m": self.depth_mean_m,
            f"{prefix}_depth_p95_m": self.depth_p95_m,
            f"{prefix}_depth_max_m": self.depth_max_m,
            f"{prefix}_per_grasp_max_mean_m": self.per_grasp_max_mean_m,
            f"{prefix}_per_grasp_max_p95_m": self.per_grasp_max_p95_m,
            f"{prefix}_per_grasp_max_max_m": self.per_grasp_max_max_m,
            f"{prefix}_depth_mean_mm": None if self.depth_mean_m is None else self.depth_mean_m * 1000.0,
            f"{prefix}_depth_p95_mm": None if self.depth_p95_m is None else self.depth_p95_m * 1000.0,
            f"{prefix}_depth_max_mm": None if self.depth_max_m is None else self.depth_max_m * 1000.0,
            f"{prefix}_per_grasp_max_mean_mm": None
            if self.per_grasp_max_mean_m is None
            else self.per_grasp_max_mean_m * 1000.0,
            f"{prefix}_per_grasp_max_p95_mm": None
            if self.per_grasp_max_p95_m is None
            else self.per_grasp_max_p95_m * 1000.0,
            f"{prefix}_per_grasp_max_max_mm": None
            if self.per_grasp_max_max_m is None
            else self.per_grasp_max_max_m * 1000.0,
        }
        return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4-case solimp/solref ablation for one object-scale.")
    parser.add_argument("--base-config", type=str, default="configs/run_YCB_liberhand_right.json")
    parser.add_argument("--object-scale-key", type=str, default="YCB_001_chips_can__scale080")
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="datasets/objdata_YCB/YCB_001_chips_can/scale080",
        help="Directory containing coacd.obj / object.xml / pc_warp.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="scripts/configs/solimp_solref_cases",
        help="Where generated case configs are written.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="tmp/solimp_solref_experiment",
        help="Where logs/outputs/summary are written.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--max-cap", type=int, default=30)
    parser.add_argument("--max-time-sec", type=float, default=120.0)
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--downsample-for-sim", type=int, default=512)
    parser.add_argument(
        "--parallel-cases",
        action="store_true",
        default=True,
        help="Run four cases in parallel.",
    )
    parser.add_argument(
        "--no-parallel-cases",
        dest="parallel_cases",
        action="store_false",
        help="Run four cases sequentially.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max workers when --parallel-cases is enabled.",
    )
    parser.add_argument(
        "--write-configs-only",
        action="store_true",
        help="Only write 4 configs and exit without running.",
    )
    return parser.parse_args()


def _soft_contact_profile() -> Dict[str, List[float]]:
    return {
        "solimp": [0.20, 0.95, 0.0010, 0.50, 2.0],
        "solref": [0.020, 1.0],
    }


def _hard_contact_profile() -> Dict[str, List[float]]:
    return {
        "solimp": [0.95, 0.99, 0.0001, 0.50, 2.0],
        "solref": [0.001, 1.0],
    }


def build_case_specs() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    soft = _soft_contact_profile()
    hard = _hard_contact_profile()
    return {
        "hand_soft_obj_hard": {"hand": soft, "object": hard},
        "hand_hard_obj_soft": {"hand": hard, "object": soft},
        "hand_hard_obj_hard": {"hand": hard, "object": hard},
        "hand_soft_obj_soft": {"hand": soft, "object": soft},
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_summary_from_stdout(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        match = SUMMARY_RE.search(line)
        if not match:
            continue
        return {
            "samples": int(match.group("samples")),
            "no_col": int(match.group("no_col")),
            "valid": int(match.group("valid")),
            "fail": int(match.group("fail")),
            "sim_time_sec": float(match.group("sim_time")),
            "total_elapsed_sec": float(match.group("total_elapsed")),
            "stop_reason": match.group("stop_reason"),
        }
    return {
        "samples": None,
        "no_col": None,
        "valid": None,
        "fail": None,
        "sim_time_sec": None,
        "total_elapsed_sec": None,
        "stop_reason": None,
    }


def _extract_ho_penetration_depths(mjho: MjHO) -> np.ndarray:
    ho_contact, _ = mjho.get_contact_info(obj_margin=0.0)
    depths = [max(0.0, -float(c["contact_dist"])) for c in ho_contact]
    depths = [d for d in depths if d > 0.0]
    return np.asarray(depths, dtype=np.float64)


def _aggregate_depth_statistics(depth_lists: Iterable[np.ndarray], per_grasp_max: Iterable[float]) -> ContactDepthStats:
    depth_arrays = [arr for arr in depth_lists if arr.size > 0]
    all_depths = np.concatenate(depth_arrays) if depth_arrays else np.zeros((0,), dtype=np.float64)
    per_grasp = np.asarray(list(per_grasp_max), dtype=np.float64)

    if all_depths.size > 0:
        depth_mean = float(np.mean(all_depths))
        depth_p95 = float(np.percentile(all_depths, 95.0))
        depth_max = float(np.max(all_depths))
    else:
        depth_mean = None
        depth_p95 = None
        depth_max = None

    if per_grasp.size > 0:
        per_mean = float(np.mean(per_grasp))
        per_p95 = float(np.percentile(per_grasp, 95.0))
        per_max = float(np.max(per_grasp))
    else:
        per_mean = None
        per_p95 = None
        per_max = None

    return ContactDepthStats(
        contact_pair_count=int(all_depths.size),
        depth_mean_m=depth_mean,
        depth_p95_m=depth_p95,
        depth_max_m=depth_max,
        per_grasp_max_mean_m=per_mean,
        per_grasp_max_p95_m=per_p95,
        per_grasp_max_max_m=per_max,
    )


def _simulate_extforce_with_depth(
    mjho_valid: MjHO,
    qpos_target: np.ndarray,
    qpos_prepared: np.ndarray,
    *,
    duration: float,
    trans_thresh: float,
    angle_thresh: float,
    force_mag: float,
    check_steps: int,
    close_steps: int,
) -> Dict[str, Any]:
    qpos_target = np.asarray(qpos_target, dtype=np.float64).reshape(-1)
    qpos_prepared = np.asarray(qpos_prepared, dtype=np.float64).reshape(-1)
    hand_ctrl = mjho_valid.qpos2ctrl(qpos_target)
    obj_body_id = int(mjho_valid.model.nbody - 1)

    external_force_dirs = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    dt = float(mjho_valid.model.opt.timestep)
    n_steps = max(int(duration / dt), 1)
    chunk_steps = max(int(check_steps), 1)
    n_chunks = max(n_steps // chunk_steps, 1)
    settle_steps = max(int(close_steps), 1)

    settle_peak = 0.0
    force_peak = 0.0
    total_peak = 0.0

    for dir_vec in external_force_dirs:
        mjho_valid.reset()
        mjho_valid.set_hand_qpos(qpos_prepared)
        initial_obj_pose = mjho_valid.get_obj_pose().copy()

        for _ in range(settle_steps):
            mjho_valid.step(1, ctrl=hand_ctrl)
            d = _extract_ho_penetration_depths(mjho_valid)
            local_peak = float(np.max(d)) if d.size > 0 else 0.0
            settle_peak = max(settle_peak, local_peak)
            total_peak = max(total_peak, local_peak)

        if not mjho_valid.is_contact():
            mjho_valid.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)
            return {
                "success": False,
                "pos_delta": float("inf"),
                "angle_delta": float("inf"),
                "settle_peak_m": settle_peak,
                "force_peak_m": force_peak,
                "total_peak_m": total_peak,
            }

        settle_pos_delta, settle_angle_delta = mjho_valid.get_pose_delta(initial_obj_pose, mjho_valid.get_obj_pose())
        if (settle_pos_delta >= trans_thresh) or (settle_angle_delta >= angle_thresh):
            mjho_valid.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)
            return {
                "success": False,
                "pos_delta": float(settle_pos_delta),
                "angle_delta": float(settle_angle_delta),
                "settle_peak_m": settle_peak,
                "force_peak_m": force_peak,
                "total_peak_m": total_peak,
            }

        settled_obj_pose = mjho_valid.get_obj_pose().copy()
        applied = np.zeros(6, dtype=float)
        applied[:3] = dir_vec[:3] * float(force_mag)
        mjho_valid.data.xfrc_applied[obj_body_id] = applied

        for _ in range(n_chunks):
            for _ in range(chunk_steps):
                mjho_valid.step(1, ctrl=hand_ctrl)
                d = _extract_ho_penetration_depths(mjho_valid)
                local_peak = float(np.max(d)) if d.size > 0 else 0.0
                force_peak = max(force_peak, local_peak)
                total_peak = max(total_peak, local_peak)

            if not mjho_valid.is_contact():
                mjho_valid.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)
                return {
                    "success": False,
                    "pos_delta": float("inf"),
                    "angle_delta": float("inf"),
                    "settle_peak_m": settle_peak,
                    "force_peak_m": force_peak,
                    "total_peak_m": total_peak,
                }

            pos_delta, angle_delta = mjho_valid.get_pose_delta(settled_obj_pose, mjho_valid.get_obj_pose())
            if (pos_delta >= trans_thresh) or (angle_delta >= angle_thresh):
                mjho_valid.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)
                return {
                    "success": False,
                    "pos_delta": float(pos_delta),
                    "angle_delta": float(angle_delta),
                    "settle_peak_m": settle_peak,
                    "force_peak_m": force_peak,
                    "total_peak_m": total_peak,
                }

        mjho_valid.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)

    final_pos_delta, final_angle_delta = mjho_valid.get_pose_delta(settled_obj_pose, mjho_valid.get_obj_pose())
    return {
        "success": bool((final_pos_delta < trans_thresh) and (final_angle_delta < angle_thresh)),
        "pos_delta": float(final_pos_delta),
        "angle_delta": float(final_angle_delta),
        "settle_peak_m": settle_peak,
        "force_peak_m": force_peak,
        "total_peak_m": total_peak,
    }


def _run_one_case(
    *,
    python_exe: str,
    config_path: Path,
    object_scale_key: str,
    asset_dir: Path,
    pc_subdir: str,
    output_dir: Path,
    log_path: Path,
) -> Tuple[int, str, float]:
    cmd = [
        python_exe,
        "run.py",
        "-c",
        str(config_path),
        "--object-scale-key",
        object_scale_key,
        "--mjcf-path",
        str(asset_dir / "object.xml"),
        "--global-pc-path",
        str(asset_dir / pc_subdir / "global_pc.npy"),
        "--global-normals-path",
        str(asset_dir / pc_subdir / "global_normals.npy"),
        "--output-dir",
        str(output_dir),
        "--force",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout, wall


def _run_cases_parallel(
    *,
    python_exe: str,
    config_paths: Dict[str, Path],
    object_scale_key: str,
    asset_dir: Path,
    pc_subdir: str,
    out_dir: Path,
    logs_dir: Path,
    max_workers: int,
) -> Dict[str, Tuple[int, str, float]]:
    results: Dict[str, Tuple[int, str, float]] = {}
    worker_count = max(1, min(int(max_workers), len(config_paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_case = {}
        for case_name, cfg_path in config_paths.items():
            case_log = logs_dir / f"{case_name}.log"
            case_output_dir = out_dir / case_name
            fut = executor.submit(
                _run_one_case,
                python_exe=python_exe,
                config_path=cfg_path,
                object_scale_key=object_scale_key,
                asset_dir=asset_dir,
                pc_subdir=pc_subdir,
                output_dir=case_output_dir,
                log_path=case_log,
            )
            future_to_case[fut] = case_name

        for fut in as_completed(future_to_case):
            case_name = future_to_case[fut]
            results[case_name] = fut.result()
    return results


def _build_case_config(
    base_cfg: Dict[str, Any],
    case_spec: Dict[str, Dict[str, List[float]]],
    *,
    max_cap: int,
    max_time_sec: float,
    n_points: int,
    downsample_for_sim: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["max_cap"] = int(max_cap)
    cfg["data"]["max_time_sec"] = float(max_time_sec)
    cfg["sampling"]["n_points"] = int(n_points)
    cfg["sampling"]["downsample_for_sim"] = int(downsample_for_sim)

    cfg["hand"]["profile"]["solimp"] = list(case_spec["hand"]["solimp"])
    cfg["hand"]["profile"]["solref"] = list(case_spec["hand"]["solref"])
    cfg["profile_object"]["solimp"] = list(case_spec["object"]["solimp"])
    cfg["profile_object"]["solref"] = list(case_spec["object"]["solref"])
    return cfg


def _compute_depth_metrics(
    cfg: Dict[str, Any],
    object_name: str,
    object_xml: Path,
    grasp_h5_path: Path,
) -> Dict[str, Any]:
    if not grasp_h5_path.exists():
        return {
            "grasp_count": 0,
            "extforce_replay_success_count": 0,
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("grasp"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("squeeze"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_settle_peak"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_force_peak"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_total_peak"),
        }

    with h5py.File(grasp_h5_path, "r") as hf:
        qpos_grasp = np.asarray(hf["qpos_grasp"][:], dtype=np.float64)
        qpos_squeeze = np.asarray(hf["qpos_squeeze"][:], dtype=np.float64)
        qpos_prepared = np.asarray(hf["qpos_prepared"][:], dtype=np.float64)

    if qpos_squeeze.shape[0] == 0:
        return {
            "grasp_count": 0,
            "extforce_replay_success_count": 0,
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("grasp"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("squeeze"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_settle_peak"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_force_peak"),
            **ContactDepthStats(0, None, None, None, None, None, None).to_dict("extforce_total_peak"),
        }

    obj_info = {"name": object_name, "xml_abs": str(object_xml.resolve())}
    common_kwargs = {
        "obj_info": obj_info,
        "hand_xml_path": str(Path(cfg["hand"]["xml_path"]).resolve()),
        "anchor_params": dict(cfg["hand"]["anchor_params"]),
        "hand_profile": dict(cfg["hand"]["profile"]),
        "object_profile": dict(cfg["profile_object"]),
    }

    mjho_fixed = MjHO(object_fixed=True, **common_kwargs)
    mjho_valid = MjHO(object_fixed=False, **common_kwargs)

    grasp_depth_lists: List[np.ndarray] = []
    squeeze_depth_lists: List[np.ndarray] = []
    grasp_per_max: List[float] = []
    squeeze_per_max: List[float] = []

    ext_settle_peaks: List[float] = []
    ext_force_peaks: List[float] = []
    ext_total_peaks: List[float] = []
    ext_replay_success_count = 0

    ext_cfg = dict(cfg.get("extforce", {}))

    for i in range(qpos_squeeze.shape[0]):
        q_grasp = qpos_grasp[i]
        q_sq = qpos_squeeze[i]
        q_prepared_saved = qpos_prepared[i]

        mjho_fixed.set_hand_qpos(q_grasp)
        depth_grasp = _extract_ho_penetration_depths(mjho_fixed)
        grasp_depth_lists.append(depth_grasp)
        grasp_per_max.append(float(np.max(depth_grasp)) if depth_grasp.size > 0 else 0.0)

        mjho_fixed.set_hand_qpos(q_sq)
        depth_sq = _extract_ho_penetration_depths(mjho_fixed)
        squeeze_depth_lists.append(depth_sq)
        squeeze_per_max.append(float(np.max(depth_sq)) if depth_sq.size > 0 else 0.0)

        q_prepared_valid = mjho_valid.build_pregrasp_qpos(q_sq, q_prepared_saved[7:])
        ext_stats = _simulate_extforce_with_depth(
            mjho_valid,
            qpos_target=q_sq,
            qpos_prepared=q_prepared_valid,
            duration=float(ext_cfg.get("duration", 1.0)),
            trans_thresh=float(ext_cfg.get("trans_thresh", 0.05)),
            angle_thresh=float(ext_cfg.get("angle_thresh", 10.0)),
            force_mag=float(ext_cfg.get("force_mag", 1.0)),
            check_steps=int(ext_cfg.get("check_steps", 50)),
            close_steps=int(ext_cfg.get("close_steps", 100)),
        )

        if bool(ext_stats["success"]):
            ext_replay_success_count += 1

        ext_settle_peaks.append(float(ext_stats["settle_peak_m"]))
        ext_force_peaks.append(float(ext_stats["force_peak_m"]))
        ext_total_peaks.append(float(ext_stats["total_peak_m"]))

    grasp_stats = _aggregate_depth_statistics(grasp_depth_lists, grasp_per_max)
    squeeze_stats = _aggregate_depth_statistics(squeeze_depth_lists, squeeze_per_max)
    ext_settle_stats = _aggregate_depth_statistics(
        depth_lists=[np.asarray([x], dtype=np.float64) for x in ext_settle_peaks if x > 0.0],
        per_grasp_max=ext_settle_peaks,
    )
    ext_force_stats = _aggregate_depth_statistics(
        depth_lists=[np.asarray([x], dtype=np.float64) for x in ext_force_peaks if x > 0.0],
        per_grasp_max=ext_force_peaks,
    )
    ext_total_stats = _aggregate_depth_statistics(
        depth_lists=[np.asarray([x], dtype=np.float64) for x in ext_total_peaks if x > 0.0],
        per_grasp_max=ext_total_peaks,
    )

    return {
        "grasp_count": int(qpos_squeeze.shape[0]),
        "extforce_replay_success_count": int(ext_replay_success_count),
        **grasp_stats.to_dict("grasp"),
        **squeeze_stats.to_dict("squeeze"),
        **ext_settle_stats.to_dict("extforce_settle_peak"),
        **ext_force_stats.to_dict("extforce_force_peak"),
        **ext_total_stats.to_dict("extforce_total_peak"),
    }


def main() -> None:
    args = parse_args()

    repo_root = Path.cwd()
    base_cfg = load_config(args.base_config)
    object_scale_key = str(args.object_scale_key)
    object_name = object_scale_key.split("__", 1)[0]

    asset_dir = Path(args.asset_dir).resolve()
    if not asset_dir.exists():
        raise FileNotFoundError(f"asset-dir not found: {asset_dir}")
    if not (asset_dir / "coacd.obj").exists():
        raise FileNotFoundError(f"Missing coacd.obj in {asset_dir}")
    if not (asset_dir / "object.xml").exists():
        raise FileNotFoundError(f"Missing object.xml in {asset_dir}")

    config_dir = Path(args.config_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    logs_dir = work_dir / "logs"
    out_dir = work_dir / "outputs"

    case_specs = build_case_specs()
    config_paths: Dict[str, Path] = {}

    for case_name, case_spec in case_specs.items():
        case_cfg = _build_case_config(
            base_cfg,
            case_spec,
            max_cap=args.max_cap,
            max_time_sec=args.max_time_sec,
            n_points=args.n_points,
            downsample_for_sim=args.downsample_for_sim,
        )
        cfg_path = config_dir / f"{case_name}.json"
        _write_json(cfg_path, case_cfg)
        config_paths[case_name] = cfg_path

    print("Wrote case configs:")
    for case_name, cfg_path in config_paths.items():
        print(f"- {case_name}: {cfg_path}")

    if args.write_configs_only:
        print("write-configs-only enabled; skip execution.")
        return

    records: List[Dict[str, Any]] = []

    run_results: Dict[str, Tuple[int, str, float]] = {}
    if args.parallel_cases:
        print(f"\nRunning 4 cases in parallel (max_workers={max(1, int(args.max_workers))})...")
        run_results = _run_cases_parallel(
            python_exe=args.python,
            config_paths=config_paths,
            object_scale_key=object_scale_key,
            asset_dir=asset_dir,
            pc_subdir=PC_SUBDIR,
            out_dir=out_dir,
            logs_dir=logs_dir,
            max_workers=int(args.max_workers),
        )
    else:
        for case_name, cfg_path in config_paths.items():
            print(f"\nRunning case: {case_name}")
            case_log = logs_dir / f"{case_name}.log"
            case_output_dir = out_dir / case_name
            run_results[case_name] = _run_one_case(
                python_exe=args.python,
                config_path=cfg_path,
                object_scale_key=object_scale_key,
                asset_dir=asset_dir,
                pc_subdir=PC_SUBDIR,
                output_dir=case_output_dir,
                log_path=case_log,
            )

    for case_name, cfg_path in config_paths.items():
        case_log = logs_dir / f"{case_name}.log"
        case_output_dir = out_dir / case_name
        code, stdout, wall_time = run_results[case_name]

        summary = _parse_summary_from_stdout(stdout)

        depth_metrics = _compute_depth_metrics(
            cfg=load_config(str(cfg_path)),
            object_name=object_name,
            object_xml=asset_dir / "object.xml",
            grasp_h5_path=case_output_dir / "grasp.h5",
        )

        rec: Dict[str, Any] = {
            "case": case_name,
            "return_code": int(code),
            "wall_time_sec": float(wall_time),
            "config_path": str(cfg_path),
            "log_path": str(case_log),
            "output_dir": str(case_output_dir),
            "repo_root": str(repo_root),
        }

        rec.update(summary)
        rec.update(depth_metrics)

        samples = rec.get("samples")
        no_col = rec.get("no_col")
        valid = rec.get("valid")
        rec["no_col_rate"] = None if not samples else float(no_col) / float(samples)
        rec["valid_rate"] = None if not samples else float(valid) / float(samples)
        rec["post_collision_success_rate"] = (
            None if not no_col else float(valid) / float(no_col)
        )

        records.append(rec)

        print(
            f"case={case_name} rc={code} wall={wall_time:.2f}s "
            f"samples={samples} valid={valid} "
            f"squeeze_p95(mm)={rec.get('squeeze_depth_p95_mm')} "
            f"extforce_peak_p95(mm)={rec.get('extforce_total_peak_depth_p95_mm')}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    summary_json = work_dir / "summary.json"
    summary_csv = work_dir / "summary.csv"

    summary_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "case",
        "return_code",
        "wall_time_sec",
        "samples",
        "no_col",
        "valid",
        "fail",
        "sim_time_sec",
        "total_elapsed_sec",
        "stop_reason",
        "no_col_rate",
        "valid_rate",
        "post_collision_success_rate",
        "grasp_count",
        "extforce_replay_success_count",
        "grasp_depth_p95_mm",
        "squeeze_depth_p95_mm",
        "extforce_settle_peak_depth_p95_mm",
        "extforce_force_peak_depth_p95_mm",
        "extforce_total_peak_depth_p95_mm",
        "config_path",
        "log_path",
        "output_dir",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec.get(field) for field in csv_fields})

    print("\nDone.")
    print(f"summary json: {summary_json}")
    print(f"summary csv : {summary_csv}")


if __name__ == "__main__":
    main()
