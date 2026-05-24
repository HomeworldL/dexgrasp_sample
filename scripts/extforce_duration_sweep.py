#!/usr/bin/env python3
"""Run and summarize extforce duration sweeps across object-scale assets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

GRASP_KEYS = (
    "qpos_init",
    "qpos_approach",
    "qpos_prepared",
    "qpos_grasp",
    "qpos_squeeze",
)
SUMMARY_RE = re.compile(
    r"samples=(?P<samples>\d+).*?"
    r"no_col=(?P<no_col>\d+).*?"
    r"contact_ok=(?P<contact_ok>\d+).*?"
    r"valid=(?P<valid>\d+).*?"
    r"fail=(?P<fail>\d+).*?"
    r"time=(?P<time>[0-9.]+)s.*?"
    r"stop_reason=(?P<stop_reason>\w+)"
)
TIMING_RE = re.compile(
    r"timing .*?"
    r"contact=(?P<contact>[0-9.]+)s/\d+ "
    r"sim_grasp=(?P<sim_grasp>[0-9.]+)s/\d+ "
    r"extforce=(?P<extforce>[0-9.]+)s/\d+ "
    r"extforce_avg=(?P<extforce_avg>[0-9.]+)s "
    r"extforce_settle=(?P<extforce_settle>[0-9.]+)s "
    r"extforce_restore=(?P<extforce_restore>[0-9.]+)s "
    r"extforce_force=(?P<extforce_force>[0-9.]+)s"
)


def _scale_tag(scale: float) -> str:
    return f"scale{int(round(float(scale) * 1000)):03d}"


def _duration_tag(duration: float) -> str:
    text = f"{duration:.3f}".rstrip("0").rstrip(".")
    return "d" + text.replace(".", "p")


def _object_key(object_name: str, scale_tag: str) -> str:
    return f"{object_name}__{scale_tag}"


def _row_key(row: np.ndarray, decimals: int) -> Tuple[float, ...]:
    rounded = np.round(np.asarray(row, dtype=np.float64), decimals=decimals)
    return tuple(float(value) for value in rounded.tolist())


def _build_index(rows: np.ndarray, decimals: int) -> Dict[Tuple[float, ...], List[int]]:
    index: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        index[_row_key(row, decimals)].append(row_idx)
    return index


def _match_rows(
    base_rows: np.ndarray,
    candidate_rows: np.ndarray,
    decimals: int,
) -> Tuple[int, int, int]:
    candidate_index = _build_index(candidate_rows, decimals)
    matched = 0
    missing = 0

    for base_row in base_rows:
        key = _row_key(base_row, decimals)
        candidate_matches = candidate_index.get(key)
        if candidate_matches:
            candidate_matches.pop(0)
            matched += 1
        else:
            missing += 1

    extra = sum(
        len(candidate_matches) for candidate_matches in candidate_index.values()
    )
    return matched, missing, extra


def _load_npy_payload(path: Path, name: str) -> Dict[str, np.ndarray]:
    payload_path = path / name
    if not payload_path.exists():
        return {}
    payload = np.load(payload_path, allow_pickle=True).item()
    return {key: np.asarray(value) for key, value in payload.items()}


def _fail_counter(fail_payload: Dict[str, np.ndarray], decimals: int) -> Counter:
    qpos_fail = fail_payload.get("qpos_fail")
    failure_stage = fail_payload.get("failure_stage")
    if qpos_fail is None or failure_stage is None:
        return Counter()
    counter: Counter = Counter()
    for qpos, stage in zip(qpos_fail, failure_stage):
        stage_key = stage.decode("utf-8") if isinstance(stage, bytes) else str(stage)
        counter[(stage_key, _row_key(qpos, decimals))] += 1
    return counter


def _discover_objects(objdata_root: Path, scale_tag: str) -> List[str]:
    objects = []
    for obj_dir in sorted(objdata_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        scale_dir = obj_dir / scale_tag
        required = [
            scale_dir / "object.xml",
            scale_dir / "pc_warp" / "global_pc.npy",
            scale_dir / "pc_warp" / "global_normals.npy",
        ]
        if all(path.exists() for path in required):
            objects.append(obj_dir.name)
    return objects


def _load_object_list(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _select_objects(args: argparse.Namespace, cfg: Dict) -> List[str]:
    scale_tag = _scale_tag(args.scale)
    objdata_root = Path(cfg["data"]["generated_dataset_root"]) / str(
        cfg["data"]["objdata_tag"]
    )
    available = _discover_objects(objdata_root, scale_tag)
    if args.objects_file is not None:
        requested = _load_object_list(args.objects_file)
    elif args.objects:
        requested = list(args.objects)
    else:
        rng = np.random.default_rng(int(args.seed))
        indices = rng.permutation(len(available))[: int(args.limit)]
        requested = [available[int(idx)] for idx in indices]

    missing = [name for name in requested if name not in set(available)]
    if missing:
        raise ValueError(f"Objects missing required {scale_tag} assets: {missing}")
    return requested[: int(args.limit)] if args.limit > 0 else requested


def _write_config(
    base_cfg: Dict,
    config_dir: Path,
    duration: float,
    max_cap: Optional[int],
    max_time_sec: Optional[float],
) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data"]["profile_timing"] = True
    cfg["data"][
        "graspdata_tag"
    ] = f"{cfg['data']['graspdata_tag']}_duration_{_duration_tag(duration)}"
    if max_cap is not None:
        cfg["data"]["max_cap"] = int(max_cap)
    if max_time_sec is not None:
        cfg["data"]["max_time_sec"] = float(max_time_sec)
    extforce = cfg.setdefault("extforce", {})
    extforce["duration"] = float(duration)
    extforce.setdefault("close_steps", 100)
    extforce.setdefault("check_steps", 50)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"duration_{_duration_tag(duration)}.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return config_path


def _parse_log(log_path: Path) -> Dict[str, float | int | str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary_matches = list(SUMMARY_RE.finditer(text))
    timing_matches = list(TIMING_RE.finditer(text))
    result: Dict[str, float | int | str] = {}
    if summary_matches:
        match = summary_matches[-1]
        for key, value in match.groupdict().items():
            if key == "stop_reason":
                result[key] = value
            elif key == "time":
                result[key] = float(value)
            else:
                result[key] = int(value)
    if timing_matches:
        match = timing_matches[-1]
        for key, value in match.groupdict().items():
            result[key] = float(value)
    return result


def _run_one(
    python_exe: str,
    run_script: Path,
    config_path: Path,
    object_name: str,
    scale_tag: str,
    objdata_root: Path,
    output_root: Path,
    duration: float,
    force: bool,
) -> Tuple[Path, Dict[str, float | int | str]]:
    duration_tag = _duration_tag(duration)
    scale_dir = objdata_root / object_name / scale_tag
    output_dir = output_root / object_name / duration_tag
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and (output_dir / "grasp.npy").exists()
        and (output_dir / "grasp_fail.npy").exists()
        and log_path.exists()
    ):
        return output_dir, _parse_log(log_path)

    cmd = [
        python_exe,
        str(run_script),
        "-c",
        str(config_path),
        "--object-scale-key",
        _object_key(object_name, scale_tag),
        "--mjcf-path",
        str(scale_dir / "object.xml"),
        "--global-pc-path",
        str(scale_dir / "pc_warp" / "global_pc.npy"),
        "--global-normals-path",
        str(scale_dir / "pc_warp" / "global_normals.npy"),
        "--output-dir",
        str(output_dir),
        "--force",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Run failed for {object_name} {duration_tag}; see {log_path}"
        )
    return output_dir, _parse_log(log_path)


def _compare_outputs(
    base_dir: Path,
    candidate_dir: Path,
    decimals: int,
) -> Dict[str, float | int]:
    base = _load_npy_payload(base_dir, "grasp.npy")
    candidate = _load_npy_payload(candidate_dir, "grasp.npy")
    if not base or not candidate:
        return {
            "matched_rows": 0,
            "missing_rows": 0,
            "extra_rows": 0,
            "positive_match_rate": 0.0,
            "fail_missing": 0,
            "fail_extra": 0,
        }

    matched, missing, extra = _match_rows(
        base["qpos_prepared"],
        candidate["qpos_prepared"],
        decimals=decimals,
    )
    base_count = int(base["qpos_prepared"].shape[0])

    base_fail = _fail_counter(_load_npy_payload(base_dir, "grasp_fail.npy"), decimals)
    candidate_fail = _fail_counter(
        _load_npy_payload(candidate_dir, "grasp_fail.npy"), decimals
    )
    fail_missing = base_fail - candidate_fail
    fail_extra = candidate_fail - base_fail

    return {
        "matched_rows": matched,
        "missing_rows": missing,
        "extra_rows": extra,
        "positive_match_rate": matched / max(base_count, 1),
        "fail_missing": sum(fail_missing.values()),
        "fail_extra": sum(fail_extra.values()),
    }


def _write_reports(
    rows: List[Dict[str, float | int | str]],
    output_root: Path,
    base_duration: float,
) -> None:
    csv_path = output_root / "summary.csv"
    md_path = output_root / "summary.md"
    fieldnames = [
        "object_name",
        "duration",
        "samples",
        "contact_ok",
        "valid",
        "time",
        "extforce",
        "extforce_avg",
        "stop_reason",
        "matched_rows",
        "missing_rows",
        "extra_rows",
        "positive_match_rate",
        "fail_missing",
        "fail_extra",
        "time_ratio_vs_base",
        "recommendation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# Extforce Duration Sweep",
        "",
        f"Base duration: `{base_duration}`",
        "",
        "| object | duration | valid | time | extforce | match | fail_missing/fail_extra | stop | recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {object_name} | {duration} | {valid} | {time:.2f} | {extforce:.2f} | "
            "{positive_match_rate:.3f} | {fail_missing}/{fail_extra} | "
            "{stop_reason} | {recommendation} |".format(
                object_name=row.get("object_name", ""),
                duration=float(row.get("duration", 0.0)),
                valid=int(row.get("valid", 0)),
                time=float(row.get("time", 0.0)),
                extforce=float(row.get("extforce", 0.0)),
                positive_match_rate=float(row.get("positive_match_rate", 0.0)),
                fail_missing=int(row.get("fail_missing", 0)),
                fail_extra=int(row.get("fail_extra", 0)),
                stop_reason=row.get("stop_reason", ""),
                recommendation=row.get("recommendation", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommend(row: Dict[str, float | int | str], base_duration: float) -> str:
    if float(row["duration"]) == float(base_duration):
        return "base"
    match_rate = float(row.get("positive_match_rate", 0.0))
    if match_rate >= 0.99:
        return "safe"
    if match_rate >= 0.95:
        return "moderate"
    if match_rate >= 0.90:
        return "aggressive"
    return "risky"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", default="configs/run_YCB_liberhand_right.json"
    )
    parser.add_argument("--run-script", type=Path, default=Path("run.py"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.12)
    parser.add_argument(
        "--durations", type=float, nargs="+", default=[0.5, 0.4, 0.3, 0.2]
    )
    parser.add_argument("--base-duration", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--objects-file", type=Path)
    parser.add_argument("--max-cap", type=int)
    parser.add_argument("--max-time-sec", type=float)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    durations = [float(duration) for duration in args.durations]
    if float(args.base_duration) not in durations:
        raise ValueError("--base-duration must be included in --durations.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "_configs"
    scale_tag = _scale_tag(float(args.scale))
    objdata_root = Path(base_cfg["data"]["generated_dataset_root"]) / str(
        base_cfg["data"]["objdata_tag"]
    )
    object_names = _select_objects(args, base_cfg)
    (output_root / "objects.txt").write_text(
        "\n".join(object_names) + "\n", encoding="utf-8"
    )
    config_paths = {
        duration: _write_config(
            base_cfg,
            config_dir=config_dir,
            duration=duration,
            max_cap=args.max_cap,
            max_time_sec=args.max_time_sec,
        )
        for duration in durations
    }

    run_dirs: Dict[Tuple[str, float], Path] = {}
    run_stats: Dict[Tuple[str, float], Dict[str, float | int | str]] = {}
    for object_name in object_names:
        for duration in durations:
            print(f"Running {object_name} duration={duration}", flush=True)
            output_dir, stats = _run_one(
                python_exe=sys.executable,
                run_script=args.run_script,
                config_path=config_paths[duration],
                object_name=object_name,
                scale_tag=scale_tag,
                objdata_root=objdata_root,
                output_root=output_root,
                duration=duration,
                force=bool(args.force),
            )
            run_dirs[(object_name, duration)] = output_dir
            run_stats[(object_name, duration)] = stats
            print(
                f"  valid={stats.get('valid')} time={stats.get('time')} "
                f"stop={stats.get('stop_reason')}",
                flush=True,
            )

    rows: List[Dict[str, float | int | str]] = []
    for object_name in object_names:
        base_dir = run_dirs[(object_name, float(args.base_duration))]
        base_time = float(
            run_stats[(object_name, float(args.base_duration))].get("time", 0.0)
        )
        for duration in durations:
            stats = dict(run_stats[(object_name, duration)])
            compare = _compare_outputs(
                base_dir,
                run_dirs[(object_name, duration)],
                decimals=int(args.decimals),
            )
            row: Dict[str, float | int | str] = {
                "object_name": object_name,
                "duration": duration,
                **stats,
                **compare,
            }
            row["time_ratio_vs_base"] = float(row.get("time", 0.0)) / max(
                base_time, 1e-8
            )
            row["recommendation"] = _recommend(row, float(args.base_duration))
            rows.append(row)

    _write_reports(rows, output_root, base_duration=float(args.base_duration))
    print(f"Wrote {output_root / 'summary.csv'}")
    print(f"Wrote {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
