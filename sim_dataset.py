#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from src.mj_ho import MjHO
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    hand_profile_from_config,
    load_config,
    object_profile_from_config,
    resolve_split_manifest_path,
    anchor_params_from_config,
)
from utils.utils_seed import set_seed


LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_qpos_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported qpos dtype '{dtype_name}'. Expected 'float32' or 'float64'.")


def _load_qpos_grasp(grasp_h5_path: Path, qpos_dtype: np.dtype) -> Dict[str, np.ndarray]:
    if not grasp_h5_path.exists():
        raise FileNotFoundError(f"Configured grasp HDF5 not found: {grasp_h5_path}")
    with h5py.File(grasp_h5_path, "r") as handle:
        required = ["qpos_init", "qpos_prepared", "qpos_grasp", "qpos_squeeze"]
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"Missing datasets {missing} in {grasp_h5_path}")
        arrays = {key: np.asarray(handle[key][:], dtype=qpos_dtype) for key in required}
    sizes = {key: int(value.shape[0]) for key, value in arrays.items()}
    if min(sizes.values()) <= 0:
        raise ValueError(f"One or more grasp datasets are empty in {grasp_h5_path}: {sizes}")
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Mismatched grasp dataset lengths in {grasp_h5_path}: {sizes}")
    return arrays


def evaluate_dataset_manifest(
    run_config_path: str,
    split: str,
    seed: Optional[int] = None,
    qpos_dtype_name: str = "float32",
    visualize: bool = False,
) -> Dict:
    cfg = load_config(run_config_path)
    eval_seed = int(cfg["seed"] if seed is None else seed)
    set_seed(eval_seed)
    qpos_dtype = _resolve_qpos_dtype(qpos_dtype_name)

    manifest_file = resolve_split_manifest_path(cfg, run_config_path, split)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_file}")

    dataset_root = manifest_file.parent.resolve()
    items = _load_json(manifest_file)
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_sim_cfg = dict(extforce_cfg)
    extforce_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("grip_delta", None)
    anchor_params = anchor_params_from_config(cfg)
    hand_profile = hand_profile_from_config(cfg)
    object_profile = object_profile_from_config(cfg)

    summary_items: List[Dict] = []
    skipped_items: List[Dict] = []
    total_success = 0
    total_attempts = 0

    LOGGER.info(
        "Evaluating manifest=%s split=%s items=%d qpos_dtype=%s",
        manifest_file,
        split,
        len(items),
        qpos_dtype.name,
    )

    for item in items:
        grasp_h5_path = dataset_root / str(item["grasp_h5_path"])
        mjcf_path = dataset_root / str(item["mjcf_path"])
        object_scale_key = str(item["object_scale_key"])
        object_name = str(item["object_name"])

        try:
            grasp_arrays = _load_qpos_grasp(grasp_h5_path, qpos_dtype=qpos_dtype)
            qpos_eval = grasp_arrays["qpos_squeeze"]
            if qpos_eval.shape[0] <= 0:
                raise ValueError("no grasps selected for evaluation")
            mjho_collision = MjHO(
                {"name": object_name, "xml_abs": str(mjcf_path.resolve())},
                cfg["hand"]["xml_path"],
                anchor_params=anchor_params,
                hand_profile=hand_profile,
                object_profile=object_profile,
                object_fixed=True,
                visualize=visualize,
            )
            mjho_valid = MjHO(
                {"name": object_name, "xml_abs": str(mjcf_path.resolve())},
                cfg["hand"]["xml_path"],
                anchor_params=anchor_params,
                hand_profile=hand_profile,
                object_profile=object_profile,
                object_fixed=False,
                visualize=visualize,
            )
        except Exception as exc:
            skipped = {"object_scale_key": object_scale_key, "reason": str(exc)}
            skipped_items.append(skipped)
            LOGGER.warning("Skip %s: %s", object_scale_key, exc)
            continue

        success_count = 0
        attempt_details: List[Dict] = []
        for grasp_idx, grasp_qpos in enumerate(qpos_eval):
            try:
                prepared_joints = grasp_arrays["qpos_prepared"][grasp_idx][7:].copy()
                prepared_qpos = mjho_collision.build_pregrasp_qpos(
                    grasp_qpos.copy(),
                    prepared_joints,
                )
                # Pre-check stored init/prepared states before running extforce validation.
                init_qpos = grasp_arrays["qpos_init"][grasp_idx]
                mjho_collision.reset()
                mjho_collision.set_hand_qpos(init_qpos.copy())
                if visualize:
                    mjho_collision._render_viewer()
                if mjho_collision.is_contact():
                    total_attempts += 1
                    attempt_details.append(
                        {
                            "grasp_index": int(grasp_idx),
                            "success": False,
                            "failure_stage": "qpos_init_contact",
                        }
                    )
                    continue
                success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
                    grasp_qpos.copy(),
                    prepared_qpos.copy(),
                    visualize=visualize,
                    **extforce_sim_cfg,
                )
                success_flag = bool(success)
                success_count += int(success_flag)
                total_success += int(success_flag)
                total_attempts += 1
                attempt_details.append(
                    {
                        "grasp_index": int(grasp_idx),
                        "success": success_flag,
                        "pos_delta": float(pos_delta),
                        "angle_delta": float(angle_delta),
                    }
                )
            except Exception as exc:
                total_attempts += 1
                attempt_details.append(
                    {
                        "grasp_index": int(grasp_idx),
                        "success": False,
                        "error": str(exc),
                    }
                )
                LOGGER.warning("Grasp eval failed for %s idx=%d: %s", object_scale_key, grasp_idx, exc)

        item_summary = {
            "object_scale_key": object_scale_key,
            "object_name": object_name,
            "grasp_count": int(qpos_eval.shape[0]),
            "validated_qpos_key": "qpos_squeeze",
            "success_count": int(success_count),
            "success_rate": float(success_count / max(1, qpos_eval.shape[0])),
            "attempts": attempt_details,
        }
        summary_items.append(item_summary)
        LOGGER.info(
            "[%s] success=%d/%d rate=%.4f",
            object_scale_key,
            success_count,
            int(qpos_eval.shape[0]),
            item_summary["success_rate"],
        )

    return {
        "manifest_path": str(manifest_file),
        "split": split,
        "qpos_dtype": qpos_dtype.name,
        "total_items": len(items),
        "evaluated_items": len(summary_items),
        "skipped_items": skipped_items,
        "total_success": int(total_success),
        "total_attempts": int(total_attempts),
        "success_rate": float(total_success / max(1, total_attempts)),
        "items": summary_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate existing dataset grasps from train/test.json using MjHO.sim_under_extforce."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Cast stored qpos arrays to this dtype before simulation. Mainline uses float32.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open MuJoCo viewer for pre-contact checks and extforce simulation.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    summary = evaluate_dataset_manifest(
        run_config_path=args.config,
        split=args.split,
        seed=args.seed,
        qpos_dtype_name=args.dtype,
        visualize=bool(args.visualize),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[sim_dataset] split={summary['split']} "
        f"dtype={summary['qpos_dtype']} "
        f"success_rate={summary['success_rate']:.6f} "
        f"success={summary['total_success']}/{summary['total_attempts']} "
        f"evaluated_items={summary['evaluated_items']} skipped_items={len(summary['skipped_items'])}"
    )


if __name__ == "__main__":
    main()
