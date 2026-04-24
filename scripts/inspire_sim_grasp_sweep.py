#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mj_ho import MjHO
from src.sample import downsample_fps
from utils.utils_file import (
    anchor_params_from_config,
    hand_profile_from_config,
    hand_root_stabilization_from_config,
    load_config,
    object_profile_from_config,
)
from utils.utils_sample import (
    build_pose_candidates,
    load_global_pc_and_normals,
    make_qpos_triplets,
    sample_frames_from_points,
)
from utils.utils_seed import set_seed, stable_seed


DEFAULT_CONFIG = "configs/run_YCB_inspire_right.json"
DEFAULT_OBJECT_SCALE_KEY = "YCB_001_chips_can__scale080"
DEFAULT_ASSET_DIR = "datasets/objdata_YCB/YCB_001_chips_can/scale080"
DEFAULT_WORK_DIR = "tmp/inspire_sim_grasp_sweep_YCB_001_chips_can_scale080"
DEFAULT_CPU_SET = "0-15"

PARAM_SWEEPS: dict[str, list[Any]] = {
    "Mp": [10, 15, 20, 25, 30],
    "steps": [30, 40, 50, 70, 90],
    "speed_gain": [0.9, 1.2, 1.5, 1.8, 2.1],
    "max_tip_speed": [0.02, 0.03, 0.04, 0.05, 0.06],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep inspire sim_grasp parameters with stage-level diagnostics."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base config JSON.")
    parser.add_argument(
        "--object-scale-key",
        default=DEFAULT_OBJECT_SCALE_KEY,
        help="Object-scale key, for example YCB_001_chips_can__scale080.",
    )
    parser.add_argument(
        "--asset-dir",
        default=DEFAULT_ASSET_DIR,
        help="Prepared object-scale asset directory.",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Directory for generated configs, logs, outputs, and summary.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=2000,
        help="Maximum candidate grasps to inspect per case.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers per parameter group.",
    )
    parser.add_argument(
        "--cpu-set",
        default=DEFAULT_CPU_SET,
        help="CPU affinity passed to taskset -c.",
    )
    parser.add_argument(
        "--run-groups",
        nargs="*",
        choices=list(PARAM_SWEEPS.keys()),
        help="Run only selected parameter groups. Default: all groups.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip worker execution and recompute summary from existing logs.",
    )
    parser.add_argument(
        "--run-case-config",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-json",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def format_value_for_case_name(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_case_specs(selected_groups: list[str] | None) -> list[dict[str, Any]]:
    groups = selected_groups or list(PARAM_SWEEPS.keys())
    specs: list[dict[str, Any]] = []
    for param_name in groups:
        for param_value in PARAM_SWEEPS[param_name]:
            specs.append(
                {
                    "group": param_name,
                    "param_name": param_name,
                    "param_value": param_value,
                    "case": f"{param_name}_{format_value_for_case_name(param_value)}",
                }
            )
    return specs


def build_case_cfg(base_cfg: dict[str, Any], case_spec: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["sim_grasp"][str(case_spec["param_name"])] = case_spec["param_value"]
    cfg["sim_grasp"]["target_point_method"] = 2
    return cfg


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_object_scale_key(object_scale_key: str) -> tuple[str, float | None]:
    if "__" not in object_scale_key:
        return object_scale_key, None
    object_name, suffix = object_scale_key.split("__", 1)
    if not suffix.startswith("scale"):
        return object_name, None
    digits = suffix[len("scale") :]
    if not digits.isdigit():
        return object_name, None
    return object_name, float(int(digits)) / 1000.0


def param_name_from_case_name(case_name: str) -> str:
    for param_name in sorted(PARAM_SWEEPS.keys(), key=len, reverse=True):
        if case_name == param_name or case_name.startswith(f"{param_name}_"):
            return param_name
    raise ValueError(f"Unable to infer param name from case '{case_name}'.")


def run_case_diagnostics(
    cfg_path: Path,
    asset_dir: Path,
    object_scale_key: str,
    candidate_limit: int,
) -> dict[str, Any]:
    cfg = load_config(str(cfg_path))
    object_name, scale = parse_object_scale_key(object_scale_key)
    hand_xml_path = cfg["hand"]["xml_path"]
    if not os.path.isabs(hand_xml_path):
        hand_xml_path = str((REPO_ROOT / hand_xml_path).resolve())

    points, normals = load_global_pc_and_normals(str(asset_dir), str(cfg["sampling"]["pc_subdir"]))
    set_seed(int(cfg["seed"]))
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(cfg["sampling"]["downsample_for_sim"]),
        seed=stable_seed(int(cfg["seed"]), object_scale_key, "downsample_for_sim"),
    )

    obj_info = {
        "name": object_name,
        "xml_abs": str((asset_dir / "object.xml").resolve()),
    }
    common = dict(
        obj_info=obj_info,
        hand_xml_path=hand_xml_path,
        anchor_params=anchor_params_from_config(cfg),
        hand_profile=hand_profile_from_config(cfg),
        object_profile=object_profile_from_config(cfg),
        root_stabilization=hand_root_stabilization_from_config(cfg),
    )
    mjho = MjHO(**common)
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)
    mjho_valid = MjHO(**common, object_fixed=False)

    set_seed(stable_seed(int(cfg["seed"]), object_scale_key, "sample_frames"))
    transforms = sample_frames_from_points(cfg, points, normals)
    pose = build_pose_candidates(cfg, transforms)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    sim_grasp_cfg = dict(cfg["sim_grasp"])
    extforce_cfg = dict(cfg["extforce"])
    contact_min_count = int(sim_grasp_cfg.pop("contact_min_count"))
    sim_grasp_cfg.pop("visualize", None)
    extforce_cfg.pop("visualize", None)
    grip_delta = float(extforce_cfg.pop("grip_delta"))

    counts: Counter[str] = Counter()
    contact_hist: Counter[int] = Counter()
    checked = min(int(candidate_limit), int(qpos_prepared.shape[0]))
    t0 = time.perf_counter()
    for i in range(checked):
        mjho.set_hand_qpos(qpos_prepared[i])
        if mjho.is_contact():
            counts["prepared_contact"] += 1
            continue
        mjho.set_hand_qpos(qpos_approach[i])
        if mjho.is_contact():
            counts["approach_contact"] += 1
            continue
        mjho.set_hand_qpos(qpos_init[i])
        if mjho.is_contact():
            counts["init_contact"] += 1
            continue

        counts["no_collision"] += 1
        mjho.set_hand_qpos(qpos_prepared[i])
        qpos_grasp, _ = mjho.sim_grasp(visualize=False, **sim_grasp_cfg)
        ho_contact_num = int(mjho.get_contact_num(obj_margin=0.0))
        contact_hist[ho_contact_num] += 1
        if ho_contact_num < contact_min_count:
            counts["insufficient_contact"] += 1
            continue

        qpos_squeeze = mjho_valid.build_squeeze_qpos(qpos_grasp, grip_delta=grip_delta)
        qpos_prepared_valid = mjho_valid.build_pregrasp_qpos(qpos_squeeze, qpos_prepared[i][7:])
        is_valid, _, _ = mjho_valid.sim_under_extforce(
            np.asarray(qpos_squeeze, dtype=np.float32).copy(),
            np.asarray(qpos_prepared_valid, dtype=np.float32).copy(),
            visualize=False,
            **extforce_cfg,
        )
        if is_valid:
            counts["valid"] += 1
        else:
            counts["extforce_failure"] += 1

    contact_values: list[int] = []
    for value, freq in sorted(contact_hist.items()):
        contact_values.extend([int(value)] * int(freq))
    elapsed = time.perf_counter() - t0
    result = {
        "samples": checked,
        "scale": scale,
        "sim_time_sec": elapsed,
        "counts": dict(counts),
        "prepared_contact": int(counts.get("prepared_contact", 0)),
        "approach_contact": int(counts.get("approach_contact", 0)),
        "init_contact": int(counts.get("init_contact", 0)),
        "no_collision": int(counts.get("no_collision", 0)),
        "insufficient_contact": int(counts.get("insufficient_contact", 0)),
        "extforce_failure": int(counts.get("extforce_failure", 0)),
        "valid": int(counts.get("valid", 0)),
        "valid_rate": float(counts.get("valid", 0)) / float(checked) if checked else None,
        "no_col_rate": float(counts.get("no_collision", 0)) / float(checked) if checked else None,
        "contact_mean": float(np.mean(contact_values)) if contact_values else None,
        "contact_median": float(np.median(contact_values)) if contact_values else None,
        "contact_max": int(max(contact_hist)) if contact_hist else None,
        "contact_histogram": {str(k): int(v) for k, v in sorted(contact_hist.items())},
        "contact_min_count": contact_min_count,
        "stop_reason": "candidate_limit",
    }
    return result


def run_worker(args: argparse.Namespace, case_spec: dict[str, Any], cfg_path: Path, output_json: Path) -> dict[str, Any]:
    log_dir = Path(args.work_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_spec['case']}.log"
    cmd = [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "taskset",
        "-c",
        str(args.cpu_set),
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-case-config",
        str(cfg_path),
        "--asset-dir",
        str(Path(args.asset_dir).resolve()),
        "--object-scale-key",
        str(args.object_scale_key),
        "--candidate-limit",
        str(int(args.candidate_limit)),
        "--output-json",
        str(output_json),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall_time_sec = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {case_spec['case']}:\n{proc.stdout}")
    record = json.loads(output_json.read_text(encoding="utf-8"))
    record["return_code"] = int(proc.returncode)
    record["wall_time_sec"] = float(wall_time_sec)
    record["log_path"] = str(log_path)
    record["config_path"] = str(cfg_path)
    return record


def load_existing_record(case_spec: dict[str, Any], output_json: Path, cfg_path: Path, work_dir: Path) -> dict[str, Any]:
    if not output_json.exists():
        raise FileNotFoundError(f"Missing output json for --skip-run: {output_json}")
    record = json.loads(output_json.read_text(encoding="utf-8"))
    record["return_code"] = 0
    record["wall_time_sec"] = None
    record["log_path"] = str(work_dir / "logs" / f"{case_spec['case']}.log")
    record["config_path"] = str(cfg_path)
    return record


def write_summary(work_dir: Path, records: list[dict[str, Any]]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_json = work_dir / "summary.json"
    summary_csv = work_dir / "summary.csv"
    save_json(summary_json, records)

    fields = [
        "case",
        "group",
        "param_name",
        "param_value",
        "return_code",
        "samples",
        "prepared_contact",
        "approach_contact",
        "init_contact",
        "no_collision",
        "insufficient_contact",
        "extforce_failure",
        "valid",
        "valid_rate",
        "no_col_rate",
        "contact_mean",
        "contact_median",
        "contact_max",
        "sim_time_sec",
        "wall_time_sec",
        "stop_reason",
        "config_path",
        "log_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})
    print(f"summary: {summary_csv}")


def run_orchestrator(args: argparse.Namespace) -> None:
    base_cfg = load_config(str(args.config))
    work_dir = Path(args.work_dir).resolve()
    config_dir = work_dir / "configs"
    output_dir = work_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = build_case_specs(args.run_groups)
    case_cfg_paths: dict[str, Path] = {}
    case_output_paths: dict[str, Path] = {}
    for case_spec in case_specs:
        cfg = build_case_cfg(base_cfg, case_spec)
        cfg_path = config_dir / f"{case_spec['case']}.json"
        save_json(cfg_path, cfg)
        case_cfg_paths[str(case_spec["case"])] = cfg_path
        case_output_paths[str(case_spec["case"])] = output_dir / f"{case_spec['case']}.json"

    records: list[dict[str, Any]] = []
    if args.skip_run:
        for case_spec in case_specs:
            records.append(
                load_existing_record(
                    case_spec,
                    case_output_paths[str(case_spec["case"])],
                    case_cfg_paths[str(case_spec["case"])],
                    work_dir,
                )
            )
    else:
        selected_groups = args.run_groups or list(PARAM_SWEEPS.keys())
        max_workers = max(1, int(args.max_workers))
        for group_name in selected_groups:
            group_specs = [spec for spec in case_specs if spec["group"] == group_name]
            worker_count = min(max_workers, len(group_specs))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        run_worker,
                        args,
                        case_spec,
                        case_cfg_paths[str(case_spec["case"])],
                        case_output_paths[str(case_spec["case"])],
                    ): case_spec
                    for case_spec in group_specs
                }
                for future in as_completed(futures):
                    record = future.result()
                    print(
                        f"{record['case']}: valid={record['valid']} "
                        f"no_col={record['no_collision']} "
                        f"prepared_contact={record['prepared_contact']} "
                        f"insufficient_contact={record['insufficient_contact']} "
                        f"contact_max={record['contact_max']}"
                    )
                    records.append(record)

    case_order = {str(spec["case"]): idx for idx, spec in enumerate(case_specs)}
    records.sort(key=lambda record: case_order[record["case"]])
    write_summary(work_dir, records)


def run_case_entry(args: argparse.Namespace) -> None:
    if not args.run_case_config or not args.output_json:
        raise ValueError("--run-case-config and --output-json are required in worker mode.")
    cfg_path = Path(args.run_case_config).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    result = run_case_diagnostics(
        cfg_path=cfg_path,
        asset_dir=asset_dir,
        object_scale_key=str(args.object_scale_key),
        candidate_limit=int(args.candidate_limit),
    )
    cfg = load_config(str(cfg_path))
    case_name = cfg_path.stem
    param_name = param_name_from_case_name(case_name)
    result.update(
        {
            "case": case_name,
            "group": param_name,
            "param_name": param_name,
            "param_value": cfg["sim_grasp"][param_name],
            "target_point_method": cfg["sim_grasp"]["target_point_method"],
            "asset_dir": str(asset_dir),
            "object_scale_key": str(args.object_scale_key),
        }
    )
    output_json = Path(args.output_json).resolve()
    save_json(output_json, result)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.run_case_config:
        run_case_entry(args)
        return
    run_orchestrator(args)


if __name__ == "__main__":
    main()
