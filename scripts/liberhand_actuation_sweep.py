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
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_CFG = "configs/run_YCB_liberhand_right.json"
OBJECT_SCALE_KEY = "YCB_001_chips_can__scale080"
ASSET_DIR = Path("datasets/objdata_YCB/YCB_001_chips_can/scale080").resolve()
WORK_DIR = Path("tmp/liberhand_actuation_sweep_YCB_001_chips_can_scale080").resolve()
CONFIG_DIR = Path("scripts/configs/liberhand_actuation_cases").resolve()
HAND_XML_TEMPLATE = Path("assets/hands/liberhand/liberhand_right.xml").resolve()
HAND_XML_CASE_DIR = Path("assets/hands/liberhand").resolve()

DEFAULT_KP_VALUES = [1, 30]
DEFAULT_FORCE_VALUES = [1, 30]

SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+)\s+no_col=(?P<no_col>\d+)\s+valid=(?P<valid>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+time=(?P<sim_time>[0-9.]+)s\s+total_elapsed=(?P<total_elapsed>[0-9.]+)s\s+"
    r"stop_reason=(?P<stop_reason>\w+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2x2x2 sweep for liberhand kp/forcerange/actuatorfrcrange.")
    parser.add_argument("--base-config", type=str, default=BASE_CFG, help=f"Base JSON config. Default: {BASE_CFG}")
    parser.add_argument(
        "--object-scale-key",
        type=str,
        default=OBJECT_SCALE_KEY,
        help=f"Object-scale key. Default: {OBJECT_SCALE_KEY}",
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=str(ASSET_DIR),
        help=f"Asset directory with coacd.obj/object.xml. Default: {ASSET_DIR}",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(WORK_DIR),
        help=f"Work directory for outputs/logs/summary. Default: {WORK_DIR}",
    )
    parser.add_argument(
        "--hand-xml-template",
        type=str,
        default=str(HAND_XML_TEMPLATE),
        help=f"Template hand XML path. Default: {HAND_XML_TEMPLATE}",
    )
    parser.add_argument("--cpu-set", type=str, default="0-15", help="CPU affinity for taskset -c.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument(
        "--kp-values",
        nargs="+",
        type=int,
        default=DEFAULT_KP_VALUES,
        help=f"KP sweep values. Default: {DEFAULT_KP_VALUES}",
    )
    parser.add_argument(
        "--force-values",
        nargs="+",
        type=int,
        default=DEFAULT_FORCE_VALUES,
        help=f"Force-range sweep values for both forcerange/actuatorfrcrange. Default: {DEFAULT_FORCE_VALUES}",
    )
    parser.add_argument("--skip-run", action="store_true", help="Skip run.py execution and summarize from logs only.")
    return parser.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


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


def build_case_specs(kp_values: list[int], force_values: list[int]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for kp in kp_values:
        for forcerange in force_values:
            for actuatorfrcrange in force_values:
                case_name = f"kp{kp}_f{forcerange}_af{actuatorfrcrange}"
                specs.append(
                    {
                        "case": case_name,
                        "kp": int(kp),
                        "forcerange": int(forcerange),
                        "actuatorfrcrange": int(actuatorfrcrange),
                    }
                )
    return specs


def _range_text(force_abs: int) -> str:
    return f"-{int(force_abs)} {int(force_abs)}"


def generate_case_xml(template_xml: Path, case_xml: Path, kp: int, forcerange: int, actuatorfrcrange: int) -> None:
    tree = ET.parse(template_xml)
    root = tree.getroot()

    for joint in root.findall(".//joint"):
        if "actuatorfrcrange" in joint.attrib:
            joint.set("actuatorfrcrange", _range_text(actuatorfrcrange))

    for position in root.findall(".//actuator/position"):
        position.set("kp", str(int(kp)))
        if "forcerange" in position.attrib:
            position.set("forcerange", _range_text(forcerange))

    case_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(case_xml, encoding="utf-8", xml_declaration=True)


def run_case(
    case_spec: dict[str, Any],
    cfg_path: Path,
    object_scale_key: str,
    asset_dir: Path,
    out_dir_root: Path,
    log_dir: Path,
    cpu_set: str,
) -> dict[str, Any]:
    case_name = str(case_spec["case"])
    out_dir = out_dir_root / case_name
    log_path = log_dir / f"{case_name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "taskset",
        "-c",
        cpu_set,
        sys.executable,
        "run.py",
        "-c",
        str(cfg_path),
        "--object-scale-key",
        object_scale_key,
        "--coacd-path",
        str(asset_dir / "coacd.obj"),
        "--mjcf-path",
        str(asset_dir / "object.xml"),
        "--asset-dir",
        str(asset_dir),
        "--output-dir",
        str(out_dir),
        "--force",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")

    summary = parse_summary(proc.stdout)
    rec = {
        "case": case_name,
        "kp": case_spec["kp"],
        "forcerange": case_spec["forcerange"],
        "actuatorfrcrange": case_spec["actuatorfrcrange"],
        "return_code": int(proc.returncode),
        "wall_time_sec": float(wall),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "output_dir": str(out_dir),
    }
    rec.update(summary)
    if summary["samples"]:
        rec["valid_rate"] = float(summary["valid"]) / float(summary["samples"])
        rec["no_col_rate"] = float(summary["no_col"]) / float(summary["samples"])
    else:
        rec["valid_rate"] = None
        rec["no_col_rate"] = None
    return rec


def load_record_from_log(case_spec: dict[str, Any], cfg_path: Path, out_dir_root: Path, log_dir: Path) -> dict[str, Any]:
    case_name = str(case_spec["case"])
    log_path = log_dir / f"{case_name}.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log for --skip-run: {log_path}")
    stdout = log_path.read_text(encoding="utf-8")
    summary = parse_summary(stdout)
    rec = {
        "case": case_name,
        "kp": case_spec["kp"],
        "forcerange": case_spec["forcerange"],
        "actuatorfrcrange": case_spec["actuatorfrcrange"],
        "return_code": 0,
        "wall_time_sec": None,
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "output_dir": str(out_dir_root / case_name),
    }
    rec.update(summary)
    if summary["samples"]:
        rec["valid_rate"] = float(summary["valid"]) / float(summary["samples"])
        rec["no_col_rate"] = float(summary["no_col"]) / float(summary["samples"])
    else:
        rec["valid_rate"] = None
        rec["no_col_rate"] = None
    return rec


def write_summary(work_dir: Path, records: list[dict[str, Any]]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_json = work_dir / "summary.json"
    summary_csv = work_dir / "summary.csv"
    summary_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "case",
        "kp",
        "forcerange",
        "actuatorfrcrange",
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
        for rec in records:
            writer.writerow({field: rec.get(field) for field in fields})

    print(f"summary: {summary_csv}")


def main() -> None:
    args = parse_args()
    base_cfg = load_cfg(args.base_config)
    object_scale_key = str(args.object_scale_key)
    asset_dir = Path(args.asset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    template_xml = Path(args.hand_xml_template).resolve()
    out_dir_root = work_dir / "outputs"
    log_dir = work_dir / "logs"

    case_specs = build_case_specs(
        kp_values=[int(v) for v in args.kp_values],
        force_values=[int(v) for v in args.force_values],
    )
    case_cfg_paths: dict[str, Path] = {}
    for case in case_specs:
        case_name = str(case["case"])
        case_xml = HAND_XML_CASE_DIR / f"liberhand_right_{case_name}.xml"
        generate_case_xml(
            template_xml=template_xml,
            case_xml=case_xml,
            kp=int(case["kp"]),
            forcerange=int(case["forcerange"]),
            actuatorfrcrange=int(case["actuatorfrcrange"]),
        )
        cfg = copy.deepcopy(base_cfg)
        cfg["hand"]["xml_path"] = str(case_xml)
        cfg_path = CONFIG_DIR / f"{case_name}.json"
        save_cfg(cfg_path, cfg)
        case_cfg_paths[case_name] = cfg_path

    records: list[dict[str, Any]] = []
    if args.skip_run:
        for case in case_specs:
            records.append(
                load_record_from_log(
                    case_spec=case,
                    cfg_path=case_cfg_paths[str(case["case"])],
                    out_dir_root=out_dir_root,
                    log_dir=log_dir,
                )
            )
    else:
        workers = max(1, min(int(args.max_workers), len(case_specs)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_case,
                    case,
                    case_cfg_paths[str(case["case"])],
                    object_scale_key,
                    asset_dir,
                    out_dir_root,
                    log_dir,
                    str(args.cpu_set),
                ): case
                for case in case_specs
            }
            for future in as_completed(futures):
                rec = future.result()
                print(
                    f"{rec['case']}: rc={rec['return_code']} samples={rec['samples']} "
                    f"valid={rec['valid']} sim={rec['sim_time_sec']}s stop={rec['stop_reason']}"
                )
                records.append(rec)

    order = {str(case["case"]): idx for idx, case in enumerate(case_specs)}
    records.sort(key=lambda rec: order[rec["case"]])
    write_summary(work_dir, records)


if __name__ == "__main__":
    main()
