#!/usr/bin/env python3
"""
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
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config

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
    p.add_argument("-v", "--verbose", action="store_true", help="打印详细日志并透传给 run.py")
    return p.parse_args()


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def _grasp_outputs_exist(entry: Dict) -> bool:
    out_dir = Path(str(entry["output_dir_abs"]))
    return (out_dir / "grasp.h5").exists() and (out_dir / "grasp.npy").exists()


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
        "--scale",
        str(float(entry.get("scale"))),
        "--object-id",
        str(entry.get("object_id", entry.get("object_name", ""))),
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
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print("Discovering dataset object-scale entries...")
    cfg = load_config(args.config)
    ds = DatasetObjects(
        cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag_from_config(args.config),
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(args.verbose),
    )
    entries = sorted(ds.get_entries(), key=lambda it: int(it["global_id"]))
    if not entries:
        print("没有发现任何 object-scale 条目，退出。")
        return

    if not args.force:
        total_entries = len(entries)
        entries = [it for it in entries if not _grasp_outputs_exist(it)]
        skipped = total_entries - len(entries)
        if skipped > 0:
            print(f"Pre-skip existing results: {skipped}/{total_entries} entries already have grasp.h5 and grasp.npy.")
        if not entries:
            print("所有 object-scale 条目都已存在 grasp.h5 和 grasp.npy，退出。")
            return

    print(f"Found {len(entries)} object-scale entries. Running with max parallel = {args.max_parallel}.")
    if args.verbose:
        for i, it in enumerate(entries):
            print(f"  [{i}] {it['object_scale_key']}")

    # 并行执行
    futures = []
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
    print(f"Time Stamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    main()
