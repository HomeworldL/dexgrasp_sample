#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from run import set_seed
from src.mj_ho import MjHO
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config


LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(
    run_cfg: Dict,
    run_config_path: str,
    split: str,
    manifest_path: Optional[str],
) -> Path:
    if manifest_path:
        return Path(manifest_path).resolve()
    dataset_root = Path(run_cfg.get("output", {}).get("dataset_root", "datasets")).resolve()
    dataset_tag = dataset_tag_from_config(run_config_path)
    return dataset_root / dataset_tag / f"{split}.json"


def _load_qpos_grasp(grasp_h5_path: Path) -> Dict[str, np.ndarray]:
    if not grasp_h5_path.exists():
        raise FileNotFoundError(f"grasp.h5 not found: {grasp_h5_path}")
    with h5py.File(grasp_h5_path, "r") as handle:
        required = ["qpos_init", "qpos_prepared", "qpos_grasp"]
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"Missing datasets {missing} in {grasp_h5_path}")
        arrays = {key: np.asarray(handle[key][:], dtype=np.float32) for key in required}
    sizes = {key: int(value.shape[0]) for key, value in arrays.items()}
    if min(sizes.values()) <= 0:
        raise ValueError(f"One or more grasp datasets are empty in {grasp_h5_path}: {sizes}")
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Mismatched grasp dataset lengths in {grasp_h5_path}: {sizes}")
    return arrays


def _build_mjho(cfg: Dict, mjcf_abs_path: Path, object_name: str, object_fixed: bool, visualize: bool) -> MjHO:
    return MjHO(
        {"name": str(object_name), "xml_abs": str(mjcf_abs_path.resolve())},
        cfg["hand"]["xml_path"],
        target_body_params=cfg["hand"]["target_body_params"],
        object_fixed=object_fixed,
        visualize=visualize,
    )


def _has_contact(mjho: MjHO, hand_qpos: np.ndarray, visualize: bool) -> bool:
    mjho.reset()
    mjho.set_hand_qpos(hand_qpos.copy())
    if visualize:
        mjho._render_viewer()
    return bool(mjho.is_contact())


def evaluate_dataset_manifest(
    run_config_path: str,
    split: str,
    manifest_path: Optional[str] = None,
    seed: Optional[int] = None,
    max_grasps_per_object_scale: Optional[int] = None,
    visualize: bool = False,
) -> Dict:
    cfg = load_config(run_config_path)
    eval_seed = int(cfg["seed"] if seed is None else seed)
    set_seed(eval_seed)

    manifest_file = _resolve_manifest_path(cfg, run_config_path, split, manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_file}")

    dataset_root = manifest_file.parent.resolve()
    items = _load_json(manifest_file)
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_cfg.pop("visualize", None)

    summary_items: List[Dict] = []
    skipped_items: List[Dict] = []
    total_success = 0
    total_attempts = 0

    LOGGER.info("Evaluating manifest=%s split=%s items=%d", manifest_file, split, len(items))

    for item in items:
        grasp_h5_path = dataset_root / str(item["grasp_h5_path"])
        mjcf_path = dataset_root / str(item["mjcf_path"])
        object_scale_key = str(item["object_scale_key"])
        object_name = str(item["object_name"])

        try:
            grasp_arrays = _load_qpos_grasp(grasp_h5_path)
            if max_grasps_per_object_scale is not None:
                keep = max(0, int(max_grasps_per_object_scale))
                grasp_arrays = {key: value[:keep] for key, value in grasp_arrays.items()}
            qpos_grasp = grasp_arrays["qpos_grasp"]
            if qpos_grasp.shape[0] <= 0:
                raise ValueError("no grasps selected for evaluation")
            mjho_collision = _build_mjho(cfg, mjcf_path, object_name, object_fixed=True, visualize=visualize)
            mjho_valid = _build_mjho(cfg, mjcf_path, object_name, object_fixed=False, visualize=visualize)
        except Exception as exc:
            skipped = {"object_scale_key": object_scale_key, "reason": str(exc)}
            skipped_items.append(skipped)
            LOGGER.warning("Skip %s: %s", object_scale_key, exc)
            continue

        success_count = 0
        attempt_details: List[Dict] = []
        for grasp_idx, grasp_qpos in enumerate(qpos_grasp):
            try:
                init_qpos = grasp_arrays["qpos_init"][grasp_idx]
                prepared_qpos = grasp_arrays["qpos_prepared"][grasp_idx]
                if _has_contact(mjho_collision, init_qpos, visualize=visualize):
                    total_attempts += 1
                    attempt_details.append(
                        {
                            "grasp_index": int(grasp_idx),
                            "success": False,
                            "failure_stage": "qpos_init_contact",
                        }
                    )
                    continue
                if _has_contact(mjho_collision, prepared_qpos, visualize=visualize):
                    total_attempts += 1
                    attempt_details.append(
                        {
                            "grasp_index": int(grasp_idx),
                            "success": False,
                            "failure_stage": "qpos_prepared_contact",
                        }
                    )
                    continue
                success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
                    grasp_qpos.copy(),
                    visualize=visualize,
                    **extforce_cfg,
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
            "grasp_count": int(qpos_grasp.shape[0]),
            "success_count": int(success_count),
            "success_rate": float(success_count / max(1, qpos_grasp.shape[0])),
            "attempts": attempt_details,
        }
        summary_items.append(item_summary)
        LOGGER.info(
            "[%s] success=%d/%d rate=%.4f",
            object_scale_key,
            success_count,
            int(qpos_grasp.shape[0]),
            item_summary["success_rate"],
        )

    return {
        "manifest_path": str(manifest_file),
        "split": split,
        "total_items": len(items),
        "evaluated_items": len(summary_items),
        "skipped_items": skipped_items,
        "total_success": int(total_success),
        "total_attempts": int(total_attempts),
        "success_rate": float(total_success / max(1, total_attempts)),
        "max_grasps_per_object_scale": None if max_grasps_per_object_scale is None else int(max_grasps_per_object_scale),
        "items": summary_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing dataset grasps from train/test.json using MjHO.sim_under_extforce."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-grasps-per-object-scale",
        type=int,
        default=None,
        help="Optional cap for grasps evaluated per object-scale.",
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
        manifest_path=args.manifest_path,
        seed=args.seed,
        max_grasps_per_object_scale=args.max_grasps_per_object_scale,
        visualize=bool(args.visualize),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[eval_dataset] split={summary['split']} "
        f"success_rate={summary['success_rate']:.6f} "
        f"success={summary['total_success']}/{summary['total_attempts']} "
        f"evaluated_items={summary['evaluated_items']} skipped_items={len(summary['skipped_items'])}"
    )


if __name__ == "__main__":
    main()
