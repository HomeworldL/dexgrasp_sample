#!/usr/bin/env python3
"""
并行调用 run_mjx.py（每个 object-scale 一条命令）。
默认并行数为 4。每个 object-scale 的输出写到 logs/<object_scale_key>.log。
"""

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List
import time

from src.dataset_objects import DatasetObjects
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config

_RUN_PROCS = []
_RUN_PROCS_LOCK = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser(description="Run run_mjx.py for all objects in dataset in parallel.")
    p.add_argument(
        "-j",
        "--max-parallel",
        type=int,
        default=4,
        help="最大并行进程数（默认 4）",
    )
    p.add_argument(
        "--script",
        type=str,
        default="run_mjx.py",
        help="要调用的脚本（默认 run_mjx.py，相对或绝对路径）",
    )
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="运行配置 JSON（默认 configs/run_YCB_liberhand.json）",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="打印详细日志并透传给 run_mjx.py")
    return p.parse_args()


def get_all_entries(config_path: str, verbose: bool = False) -> List[Dict]:
    cfg = load_config(config_path)
    ds = DatasetObjects(
        cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=Path(config_path).stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=verbose,
    )
    return sorted(ds.get_entries(), key=lambda it: int(it["global_id"]))


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def run_one(
    entry: Dict,
    script: str,
    config_path: str,
    verbose: bool = False,
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
    ]
    if verbose:
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
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("Discovering dataset object-scale entries...")
    entries = get_all_entries(args.config, verbose=bool(args.verbose))
    if not entries:
        print("没有发现任何 object-scale 条目，退出。")
        return

    print(f"Found {len(entries)} object-scale entries. Running with max parallel = {args.max_parallel}.")
    if args.verbose:
        for i, it in enumerate(entries):
            print(f"  [{i}] {it['object_scale_key']}")

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            for entry in entries:
                futures.append(
                    executor.submit(
                        run_one,
                        entry,
                        args.script,
                        args.config,
                        bool(args.verbose),
                        sys.executable,
                        logs_dir,
                    )
                )
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
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    main()
