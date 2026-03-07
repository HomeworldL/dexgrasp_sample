#!/usr/bin/env python3
"""
parallel_run.py

并行调用 run.py（每个对象一条命令：python run.py -o <obj_name>）
默认并行数为 8。每个对象的输出写到 logs/<obj>.log。

用法:
  python parallel_run.py            # 使用默认并行 8
  python parallel_run.py -j 4       # 并行 4 个
  python parallel_run.py --script ./run.py  # 指定 run 脚本路径
"""

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import time
from src.dataset_objects import DatasetObjects, resolve_dataset_root
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config

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
    return p.parse_args()


def get_all_object_names(config_path: str) -> List[str]:
    """
    返回 DatasetObjects 中按 id 排序的所有 object name 列表。
    如果你的项目里 DatasetObjects 定义在不同模块，按需修改下面的导入路径。
    """
    cfg = load_config(config_path)
    ds = DatasetObjects(
        resolve_dataset_root(cfg["dataset"].get("root")),
        dataset_names=list(cfg["dataset"].get("include", [])),
        shapenet_scale_range=tuple(cfg["dataset"].get("shapenet_scale_range", [0.06, 0.15])),
        shapenet_scale_seed=int(cfg["seed"]),
    )
    id2name = ds.id2name  # dict[int->str]
    # 按 key 排序并返回 name 列表
    names = [name for _id, name in sorted(id2name.items(), key=lambda kv: kv[0])]
    return names


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def run_one(
    obj_name: str,
    script: str,
    config_path: str,
    python_exe: str = sys.executable,
    logs_dir: Path = Path("logs"),
):
    """
    启动子进程执行: python_exe script -o obj_name
    把 stdout/stderr 写入 logs/<obj_name>.log
    返回 (obj_name, returncode, logpath)
    """
    cmd = [python_exe, script, "-c", config_path, "-o", obj_name]
    logpath = logs_dir / f"{safe_filename(obj_name)}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

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
    return obj_name, ret, str(logpath)


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

    print("Discovering dataset objects...")
    names = get_all_object_names(args.config)
    if not names:
        print("没有发现任何对象（ds.id2name 为空），退出。")
        return

    print(f"Found {len(names)} objects. Running with max parallel = {args.max_parallel}.")
    for i, nm in enumerate(names):
        print(f"  [{i}] {nm}")

    # 并行执行
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            for name in names:
                futures.append(ex.submit(run_one, name, args.script, args.config))
            # 等待完成并打印结果（每完成一个就开始下一个是 ThreadPoolExecutor 的默认行为）
            for fut in as_completed(futures):
                try:
                    obj_name, code, logpath = fut.result()
                    status = "OK" if code == 0 else f"FAIL(code={code})"
                    print(f"[DONE] {obj_name}: {status} (log: {logpath})")
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
