#!/usr/bin/env python3
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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_CFG = "configs/run_YCB_liberhand_right.json"
OBJECT_SCALE_KEY = "YCB_001_chips_can__scale100"
ASSET_DIR = Path("datasets/objdata_YCB/YCB_001_chips_can/scale100").resolve()
CONFIG_DIR = Path("scripts/configs/sim_grasp_param_cases").resolve()
WORK_DIR = Path("tmp/sim_grasp_param_sweep_YCB_001_chips_can_scale100").resolve()
OUT_DIR = WORK_DIR / "outputs"
LOG_DIR = WORK_DIR / "logs"

PARAM_SWEEPS: dict[str, list[Any]] = {
    "Mp": [10, 15, 20, 25, 30],
    "steps": [30, 40, 50, 70, 90],
    "speed_gain": [0.9, 1.2, 1.5, 1.8, 2.1],
    "max_tip_speed": [0.03, 0.04, 0.05, 0.06, 0.07],
}

SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+)\s+no_col=(?P<no_col>\d+)\s+valid=(?P<valid>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+time=(?P<sim_time>[0-9.]+)s\s+total_elapsed=(?P<total_elapsed>[0-9.]+)s\s+"
    r"stop_reason=(?P<stop_reason>\w+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep sim_grasp parameters for one fixed object-scale with batched parallel execution."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip run.py and recompute summary from existing logs/configs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Parallel workers per parameter group.",
    )
    parser.add_argument(
        "--run-groups",
        nargs="*",
        choices=list(PARAM_SWEEPS.keys()),
        help="Run only selected parameter groups. Default: all groups.",
    )
    return parser.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_cfg(base_cfg: dict[str, Any], param_name: str, param_value: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["sim_grasp"][param_name] = param_value
    return cfg


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
            value_tag = format_value_for_case_name(param_value)
            case_name = f"{param_name}_{value_tag}"
            specs.append(
                {
                    "case": case_name,
                    "param_name": param_name,
                    "param_value": param_value,
                    "group": param_name,
                }
            )
    return specs


def parse_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        match = SUMMARY_RE.search(line)
        if match:
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


def run_case(case_spec: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    case_name = str(case_spec["case"])
    out_dir = OUT_DIR / case_name
    log_path = LOG_DIR / f"{case_name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "taskset",
        "-c",
        "0-15",
        sys.executable,
        "run.py",
        "-c",
        str(cfg_path),
        "--object-scale-key",
        OBJECT_SCALE_KEY,
        "--coacd-path",
        str(ASSET_DIR / "coacd.obj"),
        "--mjcf-path",
        str(ASSET_DIR / "object.xml"),
        "--asset-dir",
        str(ASSET_DIR),
        "--output-dir",
        str(out_dir),
        "--force",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")

    summary = parse_summary(proc.stdout)
    record = {
        "case": case_name,
        "group": case_spec["group"],
        "param_name": case_spec["param_name"],
        "param_value": case_spec["param_value"],
        "return_code": int(proc.returncode),
        "wall_time_sec": float(wall),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "output_dir": str(out_dir),
    }
    record.update(summary)
    if summary["samples"]:
        record["valid_rate"] = float(summary["valid"]) / float(summary["samples"])
        record["no_col_rate"] = float(summary["no_col"]) / float(summary["samples"])
    else:
        record["valid_rate"] = None
        record["no_col_rate"] = None
    return record


def load_record_from_log(case_spec: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    case_name = str(case_spec["case"])
    log_path = LOG_DIR / f"{case_name}.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log for --skip-run: {log_path}")
    stdout = log_path.read_text(encoding="utf-8")
    summary = parse_summary(stdout)
    record = {
        "case": case_name,
        "group": case_spec["group"],
        "param_name": case_spec["param_name"],
        "param_value": case_spec["param_value"],
        "return_code": 0,
        "wall_time_sec": None,
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "output_dir": str(OUT_DIR / case_name),
    }
    record.update(summary)
    if summary["samples"]:
        record["valid_rate"] = float(summary["valid"]) / float(summary["samples"])
        record["no_col_rate"] = float(summary["no_col"]) / float(summary["samples"])
    else:
        record["valid_rate"] = None
        record["no_col_rate"] = None
    return record


def write_summary(records: list[dict[str, Any]]) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = WORK_DIR / "summary.json"
    summary_csv = WORK_DIR / "summary.csv"
    summary_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "case",
        "group",
        "param_name",
        "param_value",
        "return_code",
        "samples",
        "no_col",
        "valid",
        "fail",
        "valid_rate",
        "no_col_rate",
        "sim_time_sec",
        "total_elapsed_sec",
        "wall_time_sec",
        "stop_reason",
        "config_path",
        "log_path",
        "output_dir",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})

    print(f"summary: {summary_csv}")


def main() -> None:
    args = parse_args()
    base_cfg = load_cfg(BASE_CFG)
    case_specs = build_case_specs(args.run_groups)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    case_cfg_paths: dict[str, Path] = {}
    for case_spec in case_specs:
        cfg = build_cfg(base_cfg, str(case_spec["param_name"]), case_spec["param_value"])
        cfg_path = CONFIG_DIR / f"{case_spec['case']}.json"
        save_cfg(cfg_path, cfg)
        case_cfg_paths[str(case_spec["case"])] = cfg_path

    records: list[dict[str, Any]] = []
    if args.skip_run:
        for case_spec in case_specs:
            records.append(
                load_record_from_log(
                    case_spec=case_spec,
                    cfg_path=case_cfg_paths[str(case_spec["case"])],
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
                        run_case,
                        case_spec,
                        case_cfg_paths[str(case_spec["case"])],
                    ): case_spec
                    for case_spec in group_specs
                }
                for future in as_completed(futures):
                    record = future.result()
                    print(
                        f"{record['case']}: rc={record['return_code']} samples={record['samples']} "
                        f"valid={record['valid']} sim={record['sim_time_sec']}s stop={record['stop_reason']}"
                    )
                    records.append(record)

    case_order = {str(spec["case"]): idx for idx, spec in enumerate(case_specs)}
    records.sort(key=lambda record: case_order[record["case"]])
    write_summary(records)


if __name__ == "__main__":
    main()
