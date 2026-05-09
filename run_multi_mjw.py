#!/usr/bin/env python3
"""Parallel object-scale runner for run_mjw.py."""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from src.dataset_objects import DatasetObjects
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    build_logs_dir,
    data_verbose_from_config,
    generated_dataset_root_from_config,
    graspdata_tag_from_config,
    load_config,
    objdata_tag_from_config,
    run_scales_from_config,
    safe_filename,
    use_native_asset_from_config,
)
from utils.utils_sample import global_pc_exists, grasp_outputs_exist

_RUN_PROCS = []
_RUN_PROCS_LOCK = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser(description="Run run_mjw.py for all object-scale entries in parallel.")
    p.add_argument("-j", "--max-parallel", type=int, default=4, help="最大并行进程数（默认 4）")
    p.add_argument("--script", type=str, default="run_mjw.py", help="要调用的脚本（默认 run_mjw.py）")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="运行配置 JSON（默认 configs/run_YCB_liberhand_right.json）",
    )
    p.add_argument("--force", action="store_true", help="即使已存在配置指定的抓取输出也强制重跑")
    p.add_argument("-v", "--verbose", action="store_true", help="仅透传详细日志给子进程 run_mjw.py")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--nconmax", type=int, default=32)
    p.add_argument("--naconmax", type=int, default=16384)
    p.add_argument("--njmax", type=int, default=None)
    p.add_argument("--ccd-iterations", type=int, default=200)
    return p.parse_args()

def run_one(
    entry: Dict,
    script: str,
    config_path: str,
    args: argparse.Namespace,
    python_exe: str = sys.executable,
    logs_dir: Path = Path("logs"),
):
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
        "--batch-size",
        str(int(args.batch_size)),
        "--device",
        str(args.device),
        "--nconmax",
        str(int(args.nconmax)),
        "--naconmax",
        str(int(args.naconmax)),
        "--ccd-iterations",
        str(int(args.ccd_iterations)),
    ]
    if args.njmax is not None:
        cmd.extend(["--njmax", str(int(args.njmax))])
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("-v")

    logpath = logs_dir / f"{safe_filename(object_scale_key)}.log"
    with open(logpath, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        with _RUN_PROCS_LOCK:
            _RUN_PROCS.append(proc)
        ret = proc.wait()
        with _RUN_PROCS_LOCK:
            if proc in _RUN_PROCS:
                _RUN_PROCS.remove(proc)
    return object_scale_key, ret, str(logpath)


def terminate_all_running():
    with _RUN_PROCS_LOCK:
        procs = list(_RUN_PROCS)
    if not procs:
        return
    print(f"Terminating {len(procs)} running child process(es)...")
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    time.sleep(0.5)
    with _RUN_PROCS_LOCK:
        procs = list(_RUN_PROCS)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


def main():
    args = parse_args()
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("Discovering dataset object-scale entries...")

    cfg = load_config(args.config)
    objdata_tag = objdata_tag_from_config(cfg, args.config)
    graspdata_tag = graspdata_tag_from_config(cfg, args.config)
    logs_dir = build_logs_dir(args.script, graspdata_tag)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ds = DatasetObjects(
        scales=run_scales_from_config(cfg),
        objdata_tag=objdata_tag,
        include_native=use_native_asset_from_config(cfg),
        graspdata_tag=graspdata_tag,
        generated_dataset_root=generated_dataset_root_from_config(cfg),
        verbose=data_verbose_from_config(cfg),
    )
    entries = sorted(ds.get_entries(), key=lambda it: int(it["global_id"]))
    if not entries:
        print("没有发现任何 object-scale 条目，退出。")
        return

    h5_name = str(cfg["data"]["h5_name"])
    npy_name = str(cfg["data"]["npy_name"])
    render_subdir = str(cfg["sampling"]["pc_subdir"])
    if not args.force:
        total_entries = len(entries)
        entries = [
            it
            for it in entries
            if not (
                grasp_outputs_exist(
                    str(it["output_dir_abs"]),
                    h5_name=h5_name,
                    npy_name=npy_name,
                )
                and global_pc_exists(
                    str(it["output_dir_abs"]),
                    render_subdir=render_subdir,
                )
            )
        ]
        skipped = total_entries - len(entries)
        if skipped > 0:
            print(
                f"Pre-skip existing results: {skipped}/{total_entries} entries already have "
                f"{h5_name}, {npy_name}, and {render_subdir}/global_pc.npy."
            )
        if not entries:
            print(
                f"所有 object-scale 条目都已存在 {h5_name}、{npy_name} 和 "
                f"{render_subdir}/global_pc.npy，跳过执行。"
            )
            return

    print(
        f"Found {len(entries)} object-scale entries to run. "
        f"Running with max parallel = {args.max_parallel}."
    )
    print(f"Logs directory: {logs_dir}")
    for i, entry in enumerate(entries):
        print(f"  [{i}] {entry['object_scale_key']}")

    futures = []
    executor = None
    try:
        executor = ThreadPoolExecutor(max_workers=args.max_parallel)
        for entry in entries:
            futures.append(
                executor.submit(
                    run_one,
                    entry,
                    args.script,
                    args.config,
                    args,
                    sys.executable,
                    logs_dir,
                )
            )
        for future in as_completed(futures):
            try:
                object_scale_key, code, logpath = future.result()
                status = "OK" if code == 0 else f"FAIL(code={code})"
                print(f"[DONE] {object_scale_key}: {status} (log: {logpath})")
            except Exception as exc:
                print(f"[ERR] task raised exception: {exc}")
        executor.shutdown(wait=True, cancel_futures=False)
        executor = None
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected — terminating running child processes...")
        terminate_all_running()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        print("Exiting.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)

    print("All jobs finished.")
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    main()
