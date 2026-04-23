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

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mj_ho import MjHO
from utils.utils_file import hand_root_stabilization_from_config

BASE_CFG = "configs/run_YCB_liberhand_right.json"
OBJECT_SCALE_KEY = "YCB_013_apple__scale080"
ASSET_DIR = Path("datasets/objdata_YCB/YCB_013_apple/scale080").resolve()
CONFIG_DIR = Path("scripts/configs/friction_only_cases").resolve()
WORK_DIR = Path("tmp/friction_only_sweep_YCB_013_apple_scale080").resolve()
OUT_DIR = WORK_DIR / "outputs"
LOG_DIR = WORK_DIR / "logs"

# keep current baseline contact solver settings fixed
FIXED_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]
FIXED_SOLREF = [0.003, 1.0]

# sweep friction_coef[0], keep friction_coef[1] fixed at 0.01
FRICTION_CASES = {
    "fric_0.3": [0.3, 0.01],
    "fric_0.4": [0.4, 0.01],
    "fric_0.5": [0.5, 0.01],
    "fric_0.6": [0.6, 0.01],
    "fric_0.7": [0.7, 0.01],
}

SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+)\s+no_col=(?P<no_col>\d+)\s+valid=(?P<valid>\d+)\s+"
    r"fail=(?P<fail>\d+)\s+time=(?P<sim_time>[0-9.]+)s\s+total_elapsed=(?P<total_elapsed>[0-9.]+)s\s+"
    r"stop_reason=(?P<stop_reason>\w+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep friction_coef with fixed solimp/solref.")
    p.add_argument("--skip-run", action="store_true", help="Skip run.py and only recompute summary from existing logs/outputs.")
    p.add_argument("--no-depth", action="store_true", help="Disable penetration depth stats.")
    p.add_argument("--max-workers", type=int, default=5, help="Parallel case workers.")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_cfg(base_cfg: dict, friction: list[float]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["max_cap"] = 30
    cfg["data"]["max_time_sec"] = 120.0
    cfg["sampling"]["n_points"] = 2048
    cfg["sampling"]["downsample_for_sim"] = 512

    # controlled comparison: same parameters for hand/object
    cfg["hand"]["profile"]["friction_coef"] = [float(v) for v in friction]
    cfg["profile_object"]["friction_coef"] = [float(v) for v in friction]

    cfg["hand"]["profile"]["solimp"] = [float(v) for v in FIXED_SOLIMP]
    cfg["profile_object"]["solimp"] = [float(v) for v in FIXED_SOLIMP]
    cfg["hand"]["profile"]["solref"] = [float(v) for v in FIXED_SOLREF]
    cfg["profile_object"]["solref"] = [float(v) for v in FIXED_SOLREF]
    return cfg


def parse_summary(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        m = SUMMARY_RE.search(line)
        if m:
            return {
                "samples": int(m.group("samples")),
                "no_col": int(m.group("no_col")),
                "valid": int(m.group("valid")),
                "fail": int(m.group("fail")),
                "sim_time_sec": float(m.group("sim_time")),
                "total_elapsed_sec": float(m.group("total_elapsed")),
                "stop_reason": m.group("stop_reason"),
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


def _extract_ho_depths(mjho: MjHO) -> np.ndarray:
    ho_contact, _ = mjho.get_contact_info(obj_margin=0.0)
    vals = [max(0.0, -float(c["contact_dist"])) for c in ho_contact]
    vals = [v for v in vals if v > 0.0]
    return np.asarray(vals, dtype=np.float64)


def _summarize_depths(all_depths: list[np.ndarray], per_grasp_max: list[float]) -> dict:
    chunks = [x for x in all_depths if x.size > 0]
    cat = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float64)
    per = np.asarray(per_grasp_max, dtype=np.float64)
    out = {
        "depth_p95_mm": None,
        "depth_max_mm": None,
        "per_grasp_max_p95_mm": None,
        "per_grasp_max_max_mm": None,
    }
    if cat.size > 0:
        out["depth_p95_mm"] = float(np.percentile(cat, 95.0) * 1000.0)
        out["depth_max_mm"] = float(np.max(cat) * 1000.0)
    if per.size > 0:
        out["per_grasp_max_p95_mm"] = float(np.percentile(per, 95.0) * 1000.0)
        out["per_grasp_max_max_mm"] = float(np.max(per) * 1000.0)
    return out


def compute_depth_metrics(cfg_path: Path, output_dir: Path, object_name: str, object_xml: Path) -> dict:
    grasp_h5 = output_dir / "grasp.h5"
    if not grasp_h5.exists():
        return {
            "grasp_count": 0,
            "grasp_depth_p95_mm": None,
            "grasp_depth_max_mm": None,
            "squeeze_depth_p95_mm": None,
            "squeeze_depth_max_mm": None,
        }

    with h5py.File(grasp_h5, "r") as hf:
        qpos_grasp = np.asarray(hf["qpos_grasp"][:], dtype=np.float64)
        qpos_squeeze = np.asarray(hf["qpos_squeeze"][:], dtype=np.float64)

    if qpos_squeeze.shape[0] == 0:
        return {
            "grasp_count": 0,
            "grasp_depth_p95_mm": None,
            "grasp_depth_max_mm": None,
            "squeeze_depth_p95_mm": None,
            "squeeze_depth_max_mm": None,
        }

    cfg = load_cfg(str(cfg_path))
    obj_info = {"name": object_name, "xml_abs": str(object_xml.resolve())}
    mjho = MjHO(
        obj_info=obj_info,
        hand_xml_path=str(Path(cfg["hand"]["xml_path"]).resolve()),
        anchor_params=dict(cfg["hand"]["anchor_params"]),
        hand_profile=dict(cfg["hand"]["profile"]),
        object_profile=dict(cfg["profile_object"]),
        root_stabilization=hand_root_stabilization_from_config(cfg),
        object_fixed=True,
    )

    grasp_depths: list[np.ndarray] = []
    squeeze_depths: list[np.ndarray] = []
    grasp_per: list[float] = []
    squeeze_per: list[float] = []

    for i in range(qpos_squeeze.shape[0]):
        mjho.set_hand_qpos(qpos_grasp[i])
        d_g = _extract_ho_depths(mjho)
        grasp_depths.append(d_g)
        grasp_per.append(float(np.max(d_g)) if d_g.size > 0 else 0.0)

        mjho.set_hand_qpos(qpos_squeeze[i])
        d_s = _extract_ho_depths(mjho)
        squeeze_depths.append(d_s)
        squeeze_per.append(float(np.max(d_s)) if d_s.size > 0 else 0.0)

    g = _summarize_depths(grasp_depths, grasp_per)
    s = _summarize_depths(squeeze_depths, squeeze_per)
    return {
        "grasp_count": int(qpos_squeeze.shape[0]),
        "grasp_depth_p95_mm": g["depth_p95_mm"],
        "grasp_depth_max_mm": g["depth_max_mm"],
        "squeeze_depth_p95_mm": s["depth_p95_mm"],
        "squeeze_depth_max_mm": s["depth_max_mm"],
    }


def run_case(case_name: str, cfg_path: Path) -> dict:
    out_dir = OUT_DIR / case_name
    log_path = LOG_DIR / f"{case_name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
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

    s = parse_summary(proc.stdout)
    rec = {
        "case": case_name,
        "friction_coef": FRICTION_CASES[case_name],
        "solimp": FIXED_SOLIMP,
        "solref": FIXED_SOLREF,
        "return_code": int(proc.returncode),
        "wall_time_sec": float(wall),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "output_dir": str(out_dir),
    }
    rec.update(s)
    if s["samples"]:
        rec["valid_rate"] = float(s["valid"]) / float(s["samples"])
        rec["no_col_rate"] = float(s["no_col"]) / float(s["samples"])
    else:
        rec["valid_rate"] = None
        rec["no_col_rate"] = None
    return rec


def main() -> None:
    args = parse_args()
    base_cfg = load_cfg(BASE_CFG)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    cfg_paths: dict[str, Path] = {}
    for name, fric in FRICTION_CASES.items():
        cfg = build_cfg(base_cfg, fric)
        cfg_path = CONFIG_DIR / f"{name}.json"
        save_cfg(cfg_path, cfg)
        cfg_paths[name] = cfg_path

    records: list[dict] = []
    if args.skip_run:
        for name, path in cfg_paths.items():
            case_log = LOG_DIR / f"{name}.log"
            if not case_log.exists():
                raise FileNotFoundError(f"Missing log for --skip-run: {case_log}")
            stdout = case_log.read_text(encoding="utf-8")
            s = parse_summary(stdout)
            rec = {
                "case": name,
                "friction_coef": FRICTION_CASES[name],
                "solimp": FIXED_SOLIMP,
                "solref": FIXED_SOLREF,
                "return_code": 0,
                "wall_time_sec": None,
                "config_path": str(path),
                "log_path": str(case_log),
                "output_dir": str(OUT_DIR / name),
            }
            rec.update(s)
            if s["samples"]:
                rec["valid_rate"] = float(s["valid"]) / float(s["samples"])
                rec["no_col_rate"] = float(s["no_col"]) / float(s["samples"])
            else:
                rec["valid_rate"] = None
                rec["no_col_rate"] = None
            records.append(rec)
    else:
        workers = max(1, min(int(args.max_workers), len(cfg_paths)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_case, name, path): name for name, path in cfg_paths.items()}
            for fut in as_completed(futs):
                rec = fut.result()
                print(
                    f"{rec['case']}: rc={rec['return_code']} samples={rec['samples']} "
                    f"valid={rec['valid']} sim={rec['sim_time_sec']}s stop={rec['stop_reason']}"
                )
                records.append(rec)

    order = list(FRICTION_CASES.keys())
    records.sort(key=lambda x: order.index(x["case"]))

    if not args.no_depth:
        object_name = OBJECT_SCALE_KEY.split("__", 1)[0]
        object_xml = ASSET_DIR / "object.xml"
        for rec in records:
            depth = compute_depth_metrics(
                cfg_path=Path(rec["config_path"]),
                output_dir=Path(rec["output_dir"]),
                object_name=object_name,
                object_xml=object_xml,
            )
            rec.update(depth)

    summary_json = WORK_DIR / "summary.json"
    summary_csv = WORK_DIR / "summary.csv"
    summary_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "case",
        "friction_coef",
        "solimp",
        "solref",
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
        "grasp_count",
        "grasp_depth_p95_mm",
        "grasp_depth_max_mm",
        "squeeze_depth_p95_mm",
        "squeeze_depth_max_mm",
        "config_path",
        "log_path",
        "output_dir",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})

    print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
