#!/usr/bin/env python3
"""Compare grasp outputs by matched rows instead of row positions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

GRASP_KEYS = (
    "qpos_init",
    "qpos_approach",
    "qpos_prepared",
    "qpos_grasp",
    "qpos_squeeze",
)


def _resolve_payload_path(path: Path, name: str) -> Path:
    if path.is_dir():
        return path / name
    return path


def _load_grasp(path: Path) -> Dict[str, np.ndarray]:
    path = _resolve_payload_path(path, "grasp.npy")
    if path.suffix == ".npy":
        payload = np.load(path, allow_pickle=True).item()
        return {key: np.asarray(payload[key]) for key in GRASP_KEYS}
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as hf:
            return {key: np.asarray(hf[key][:]) for key in GRASP_KEYS}
    raise ValueError(f"Unsupported grasp path: {path}")


def _load_fail(path: Path) -> Dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "grasp_fail.npy"
    if not path.exists():
        return {
            "qpos_fail": np.zeros((0, 0), dtype=np.float32),
            "failure_stage": np.asarray([], dtype=object),
        }
    if path.suffix == ".npy":
        payload = np.load(path, allow_pickle=True).item()
        return {
            "qpos_fail": np.asarray(payload["qpos_fail"]),
            "failure_stage": np.asarray(payload["failure_stage"]),
        }
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as hf:
            return {
                "qpos_fail": np.asarray(hf["qpos_fail"][:]),
                "failure_stage": np.asarray(hf["failure_stage"][:]),
            }
    raise ValueError(f"Unsupported failure path: {path}")


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
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidate_index = _build_index(candidate_rows, decimals)
    matches: List[Tuple[int, int]] = []
    missing: List[int] = []

    for base_idx, base_row in enumerate(base_rows):
        key = _row_key(base_row, decimals)
        candidate_matches = candidate_index.get(key)
        if candidate_matches:
            matches.append((base_idx, candidate_matches.pop(0)))
        else:
            missing.append(base_idx)

    extra = [
        candidate_idx
        for candidate_matches in candidate_index.values()
        for candidate_idx in candidate_matches
    ]
    return matches, missing, extra


def _max_abs_for_matches(
    base: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    matches: Iterable[Tuple[int, int]],
) -> Dict[str, float]:
    match_list = list(matches)
    if not match_list:
        return {key: float("nan") for key in GRASP_KEYS}
    base_idx = np.asarray([pair[0] for pair in match_list], dtype=int)
    candidate_idx = np.asarray([pair[1] for pair in match_list], dtype=int)
    return {
        key: float(np.max(np.abs(base[key][base_idx] - candidate[key][candidate_idx])))
        for key in GRASP_KEYS
    }


def _fail_counter(fail_payload: Dict[str, np.ndarray], decimals: int) -> Counter:
    qpos_fail = fail_payload["qpos_fail"]
    failure_stage = fail_payload["failure_stage"]
    counter: Counter = Counter()
    for qpos, stage in zip(qpos_fail, failure_stage):
        if isinstance(stage, bytes):
            stage_key = stage.decode("utf-8")
        else:
            stage_key = str(stage)
        counter[(stage_key, _row_key(qpos, decimals))] += 1
    return counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two grasp output directories/files by matching rows with "
            "qpos_prepared instead of comparing row positions."
        )
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--match-key",
        default="qpos_prepared",
        choices=GRASP_KEYS,
        help="Dataset used to identify the same grasp candidate.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Decimal places used for row-key quantization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = _load_grasp(args.base)
    candidate = _load_grasp(args.candidate)

    matches, missing, extra = _match_rows(
        base[args.match_key],
        candidate[args.match_key],
        decimals=int(args.decimals),
    )
    max_abs = _max_abs_for_matches(base, candidate, matches)

    base_fail = _load_fail(args.base)
    candidate_fail = _load_fail(args.candidate)
    base_fail_counter = _fail_counter(base_fail, int(args.decimals))
    candidate_fail_counter = _fail_counter(candidate_fail, int(args.decimals))
    fail_missing = base_fail_counter - candidate_fail_counter
    fail_extra = candidate_fail_counter - base_fail_counter

    print(f"base_rows={base[args.match_key].shape[0]}")
    print(f"candidate_rows={candidate[args.match_key].shape[0]}")
    print(f"matched_rows={len(matches)}")
    print(f"missing_from_candidate={len(missing)}")
    print(f"extra_in_candidate={len(extra)}")
    for key in GRASP_KEYS:
        print(f"max_abs_diff[{key}]={max_abs[key]}")
    print(f"fail_base_rows={base_fail['qpos_fail'].shape[0]}")
    print(f"fail_candidate_rows={candidate_fail['qpos_fail'].shape[0]}")
    print(f"fail_missing_from_candidate={sum(fail_missing.values())}")
    print(f"fail_extra_in_candidate={sum(fail_extra.values())}")


if __name__ == "__main__":
    main()
