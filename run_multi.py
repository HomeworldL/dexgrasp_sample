#!/usr/bin/env python3
"""Parallel object-scale runner with dataset split index export."""

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    build_logs_dir,
    dataset_tag_from_config,
    list_existing_files,
    load_config,
    relpath_str,
    safe_filename,
)
from utils.utils_sample import grasp_h5_nonempty, grasp_outputs_exist

# 全局进程注册（用于在主线程捕获中断时终止子进程）
_RUN_PROCS = []
_RUN_PROCS_LOCK = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser(description="Run run.py for all objects in dataset in parallel.")
    p.add_argument(
        "-j", "--max-parallel",
        type=int,
        default=4,
        help="最大并行进程数（默认 4）"
    )
    p.add_argument(
        "--script",
        type=str,
        default="run.py",
        help="要调用的脚本（默认 run.py，相对或绝对路径）"
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="运行配置 JSON（默认 configs/run_YCB_liberhand.json）",
    )
    p.add_argument("--force", action="store_true", help="即使已有 grasp.h5 和 grasp.npy 也强制重跑")
    p.add_argument("-v", "--verbose", action="store_true", help="仅透传详细日志给子进程 run.py")
    return p.parse_args()

def _collect_entry_record(
    entry: Dict,
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    output_dir = Path(str(entry["output_dir_abs"])).resolve()
    grasp_h5_path = output_dir / "grasp.h5"
    grasp_npy_path = output_dir / "grasp.npy"
    if not grasp_h5_path.exists():
        return None, f"missing {grasp_h5_path.name}"
    if not grasp_npy_path.exists():
        return None, f"missing {grasp_npy_path.name}"

    render_dir = output_dir / render_subdir
    cam_in_path = render_dir / "cam_in.npy"
    if not cam_in_path.exists():
        return None, f"missing {render_subdir}/cam_in.npy"

    partial_pc_paths = list_existing_files(render_dir, "partial_pc_")
    partial_pc_cam_paths = list_existing_files(render_dir, "partial_pc_cam_")
    cam_ex_paths = list_existing_files(render_dir, "cam_ex_")

    partial_pc_paths = [p for p in partial_pc_paths if not p.name.startswith("partial_pc_cam_")]

    if not partial_pc_paths:
        return None, f"missing {render_subdir}/partial_pc_*.npy"
    if not partial_pc_cam_paths:
        return None, f"missing {render_subdir}/partial_pc_cam_*.npy"
    if not cam_ex_paths:
        return None, f"missing {render_subdir}/cam_ex_*.npy"

    world_suffixes = [p.stem[len("partial_pc_"):] for p in partial_pc_paths]
    cam_suffixes = [p.stem[len("partial_pc_cam_"):] for p in partial_pc_cam_paths]
    ex_suffixes = [p.stem[len("cam_ex_"):] for p in cam_ex_paths]
    if world_suffixes != cam_suffixes or world_suffixes != ex_suffixes:
        return None, f"mismatched render view files under {render_subdir}"

    record = {
        "global_id": int(entry["global_id"]),
        "object_scale_key": str(entry["object_scale_key"]),
        "object_name": str(entry["object_name"]),
        "output_path": relpath_str(output_dir, dataset_dir),
        "coacd_path": relpath_str(Path(str(entry["coacd_abs"])), dataset_dir),
        "mjcf_path": relpath_str(Path(str(entry["mjcf_abs"])), dataset_dir),
        "grasp_h5_path": relpath_str(grasp_h5_path, dataset_dir),
        "grasp_npy_path": relpath_str(grasp_npy_path, dataset_dir),
        "partial_pc_path": [relpath_str(path, dataset_dir) for path in partial_pc_paths],
        "partial_pc_cam_path": [relpath_str(path, dataset_dir) for path in partial_pc_cam_paths],
        "cam_ex_path": [relpath_str(path, dataset_dir) for path in cam_ex_paths],
        "cam_in": relpath_str(cam_in_path, dataset_dir),
        "scale": float(entry["scale"]),
    }
    return record, None


def build_split_records(
    entries: Sequence[Dict],
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    records: List[Dict] = []
    skipped: List[Tuple[str, str]] = []
    for entry in sorted(entries, key=lambda it: int(it["global_id"])):
        record, reason = _collect_entry_record(entry=entry, dataset_dir=dataset_dir, render_subdir=render_subdir)
        if record is None:
            skipped.append((str(entry["object_scale_key"]), str(reason)))
            continue
        records.append(record)
    return records, skipped


def split_records_by_object(records: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    object_names = sorted({str(record["object_name"]) for record in records})
    if len(object_names) <= 1:
        test_objects = set()
    else:
        test_count = max(1, int(round(len(object_names) * 0.2)))
        test_objects = set(object_names[-test_count:])

    train_records = [record for record in records if str(record["object_name"]) not in test_objects]
    test_records = [record for record in records if str(record["object_name"]) in test_objects]
    return train_records, test_records


def filter_nonempty_grasp_records(
    records: Sequence[Dict],
    dataset_dir: Path,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    kept: List[Dict] = []
    removed: List[Tuple[str, str]] = []
    for record in records:
        h5_path = dataset_dir / str(record["grasp_h5_path"])
        ok, reason = grasp_h5_nonempty(h5_path)
        if ok:
            kept.append(record)
            continue
        removed.append((str(record["object_scale_key"]), reason))
    return kept, removed


def write_split_jsons(
    entries: Sequence[Dict],
    dataset_dir: Path,
    render_subdir: str,
) -> Tuple[Path, Path]:
    records, skipped = build_split_records(entries=entries, dataset_dir=dataset_dir, render_subdir=render_subdir)
    train_records, test_records = split_records_by_object(records)
    train_records, empty_train = filter_nonempty_grasp_records(train_records, dataset_dir)
    test_records, empty_test = filter_nonempty_grasp_records(test_records, dataset_dir)
    empty_records = empty_train + empty_test

    train_path = dataset_dir / "train.json"
    test_path = dataset_dir / "test.json"
    train_path.write_text(json.dumps(train_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Wrote dataset splits under {dataset_dir}: "
        f"train={len(train_records)} test={len(test_records)} skipped={len(skipped)} empty={len(empty_records)}"
    )
    for object_scale_key, reason in skipped:
        print(f"[SKIP] {object_scale_key}: {reason}")
    for object_scale_key, reason in empty_records:
        print(f"[EMPTY] {object_scale_key}: {reason}")
    return train_path, test_path


def run_one(
    entry: Dict,
    script: str,
    config_path: str,
    force: bool = False,
    verbose: bool = False,
    python_exe: str = sys.executable,
    logs_dir: Path = Path("logs"),
):
    """启动子进程执行一个 object-scale 任务，返回 (key, returncode, logpath)。"""
    object_scale_key = str(entry["object_scale_key"])
    cmd = [
        python_exe,
        script,
        "-c",
        config_path,
        "--object-scale-key",
        object_scale_key,
        "--coacd-path",
        str(entry["coacd_abs"]),
        "--mjcf-path",
        str(entry["mjcf_abs"]),
        "--output-dir",
        str(entry["output_dir_abs"]),
    ]
    if force:
        cmd.append("--force")
    if verbose:
        cmd.append("-v")
    logpath = logs_dir / f"{safe_filename(object_scale_key)}.log"

    # 启动子进程并注册到全局列表，以便在需要时终止
    with open(logpath, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        with _RUN_PROCS_LOCK:
            _RUN_PROCS.append(proc)
        ret = proc.wait()
        with _RUN_PROCS_LOCK:
            # 移除已完成项（若尚在列表中）
            if proc in _RUN_PROCS:
                _RUN_PROCS.remove(proc)
    return object_scale_key, ret, str(logpath)


def terminate_all_running():
    with _RUN_PROCS_LOCK:
        procs = list(_RUN_PROCS)
    if not procs:
        return
    print(f"Terminating {len(procs)} running child process(es)...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    # 等待短时间后强制 kill
    import time
    time.sleep(0.5)
    with _RUN_PROCS_LOCK:
        procs = list(_RUN_PROCS)
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def main():
    args = parse_args()
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print("Discovering dataset object-scale entries...")
    cfg = load_config(args.config)
    dataset_tag = dataset_tag_from_config(args.config)
    logs_dir = build_logs_dir(args.script, dataset_tag)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ds = DatasetObjects(
        cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(args.verbose),
    )
    entries = sorted(ds.get_entries(), key=lambda it: int(it["global_id"]))
    if not entries:
        print("没有发现任何 object-scale 条目，退出。")
        return

    if not args.force:
        total_entries = len(entries)
        entries = [it for it in entries if not grasp_outputs_exist(str(it["output_dir_abs"]))]
        skipped = total_entries - len(entries)
        if skipped > 0:
            print(f"Pre-skip existing results: {skipped}/{total_entries} entries already have grasp.h5 and grasp.npy.")
        if not entries:
            print("所有 object-scale 条目都已存在 grasp.h5 和 grasp.npy，跳过并行执行，继续构建数据划分。")

    print(f"Found {len(entries)} object-scale entries to run. Running with max parallel = {args.max_parallel}.")
    print(f"Logs directory: {logs_dir}")
    for i, it in enumerate(entries):
        print(f"  [{i}] {it['object_scale_key']}")

    # 并行执行
    futures = []
    if entries:
        try:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
                for entry in entries:
                    futures.append(
                        ex.submit(
                            run_one,
                            entry,
                            args.script,
                            args.config,
                            bool(args.force),
                            bool(args.verbose),
                            sys.executable,
                            logs_dir,
                        )
                    )
                # 等待完成并打印结果（每完成一个就开始下一个是 ThreadPoolExecutor 的默认行为）
                for fut in as_completed(futures):
                    try:
                        object_scale_key, code, logpath = fut.result()
                        status = "OK" if code == 0 else f"FAIL(code={code})"
                        print(f"[DONE] {object_scale_key}: {status} (log: {logpath})")
                    except Exception as exc:
                        print(f"[ERR] task raised exception: {exc}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected — terminating running child processes...")
            terminate_all_running()
            print("Exiting.")
            sys.exit(1)

    print("All jobs finished.")
    dataset_root = Path(cfg.get("output", {}).get("dataset_root", "datasets")).resolve()
    dataset_dir = dataset_root / dataset_tag
    write_split_jsons(
        entries=ds.get_entries(),
        dataset_dir=dataset_dir,
        render_subdir=str(cfg["warp_render"]["output_subdir"]),
    )
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    main()
