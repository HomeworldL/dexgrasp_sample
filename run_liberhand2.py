#!/usr/bin/env python3
"""Compatibility wrapper for liberhand2 runs.

This script forwards to run.py with a liberhand2-oriented default config.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grasp sampling using liberhand2 default config.")
    parser.add_argument("-c", "--config", default="configs/run_YCB_liberhand2.json", help="JSON config path")
    parser.add_argument("-i", "--obj-id", type=int, default=None, help="Global object id")
    parser.add_argument("-o", "--obj", type=str, default=None, help="Object name override")
    args = parser.parse_args()

    run_py = str((Path(__file__).parent / "run.py").resolve())
    cmd = [sys.executable, run_py, "-c", args.config]
    if args.obj_id is not None:
        cmd.extend(["-i", str(args.obj_id)])
    if args.obj is not None:
        cmd.extend(["-o", args.obj])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
