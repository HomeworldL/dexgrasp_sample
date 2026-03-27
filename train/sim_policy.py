#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.mj_ho import MjHO
from train.train_policy import (
    EXPERIMENT,
    FlattenedGraspDataset,
    denormalize_target,
    load_model_from_checkpoint,
    qpos_from_target,
)
from utils.utils_file import load_config
from utils.utils_seed import set_seed


LOGGER = logging.getLogger(__name__)


def default_checkpoint_path() -> Path:
    return Path(str(EXPERIMENT["artifact_dir"])).resolve() / "checkpoint_best.pt"


def evaluate_policy_checkpoint(
    checkpoint_path: str | Path,
    visualize: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(int(EXPERIMENT["seed"]))
    device = str(EXPERIMENT["device"])
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    dataset = FlattenedGraspDataset(
        manifest_path=str(EXPERIMENT["manifest_path"]),
        point_count=int(EXPERIMENT["point_count"]),
        seed=int(EXPERIMENT["seed"]),
        grasp_selection_mode=str(EXPERIMENT["grasp_selection_mode"]),
    )
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)

    cfg = load_config(str(EXPERIMENT["run_config_path"]))
    extforce_sim_cfg = dict(cfg.get("extforce", {}))
    extforce_sim_cfg.pop("visualize", None)
    extforce_sim_cfg.pop("grip_delta", None)

    env_cache: Dict[str, MjHO] = {}
    per_object = defaultdict(lambda: {"success": 0, "attempts": 0})
    attempts: List[Dict[str, object]] = []
    total_success = 0
    total_attempts = 0

    sample_indices = list(range(len(dataset)))
    if limit is not None:
        sample_indices = sample_indices[: int(limit)]
    object_groups: Dict[str, List[int]] = {}
    object_names: Dict[str, str] = {}
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        object_scale_key = str(sample["object_scale_key"])
        if object_scale_key not in object_groups:
            object_groups[object_scale_key] = []
            object_names[object_scale_key] = str(sample["object_name"])
        object_groups[object_scale_key].append(sample_index)

    LOGGER.info(
        "Sim policy replay on seen train views: samples=%d objects=%d checkpoint=%s",
        len(sample_indices),
        len(object_groups),
        checkpoint_path,
    )

    model.eval()
    with torch.no_grad():
        for object_scale_key, grouped_indices in object_groups.items():
            object_name = object_names[object_scale_key]
            sample = dataset[grouped_indices[0]]
            mjcf_path = Path(dataset.dataset_root) / str(sample["mjcf_path"])
            if object_scale_key not in env_cache:
                env_cache[object_scale_key] = MjHO(
                    {"name": object_name, "xml_abs": str(mjcf_path.resolve())},
                    cfg["hand"]["xml_path"],
                    target_body_params=cfg["hand"]["target_body_params"],
                    object_fixed=False,
                    visualize=visualize,
                )

            for sample_index in grouped_indices:
                sample = dataset[sample_index]
                points = torch.from_numpy(dataset.points[sample_index]).unsqueeze(0).to(device)
                normalized_prediction = model(points).squeeze(0).cpu().numpy().astype(np.float32)
                target_prediction = denormalize_target(normalized_prediction, target_mean, target_std)
                predicted_qpos = qpos_from_target(target_prediction)

                success, pos_delta, angle_delta = env_cache[object_scale_key].sim_under_extforce(
                    predicted_qpos.copy(),
                    visualize=visualize,
                    **extforce_sim_cfg,
                )
                success_flag = bool(success)
                total_success += int(success_flag)
                total_attempts += 1
                per_object[object_scale_key]["success"] += int(success_flag)
                per_object[object_scale_key]["attempts"] += 1
                attempts.append(
                    {
                        "sample_index": int(sample_index),
                        "object_scale_key": object_scale_key,
                        "object_name": object_name,
                        "view_index": int(sample["view_index"]),
                        "grasp_index": int(sample["grasp_index"]),
                        "success": success_flag,
                        "pos_delta": float(pos_delta),
                        "angle_delta": float(angle_delta),
                    }
                )

            object_success = int(per_object[object_scale_key]["success"])
            object_attempts = int(per_object[object_scale_key]["attempts"])
            print(
                f"[sim_policy][{object_scale_key}] "
                f"success_rate={object_success / max(1, object_attempts):.6f} "
                f"success={object_success}/{object_attempts}"
            )

    per_object_items = []
    for object_scale_key in sorted(per_object):
        stats = per_object[object_scale_key]
        per_object_items.append(
            {
                "object_scale_key": object_scale_key,
                "object_name": object_names[object_scale_key],
                "success_count": int(stats["success"]),
                "attempts": int(stats["attempts"]),
                "success_rate": float(stats["success"] / max(1, stats["attempts"])),
            }
        )

    summary = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "sample_count": int(total_attempts),
        "success_count": int(total_success),
        "success_rate": float(total_success / max(1, total_attempts)),
        "items": per_object_items,
        "attempts": attempts,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[sim_policy] success_rate={summary['success_rate']:.6f} "
        f"success={summary['success_count']}/{summary['sample_count']} "
        f"objects={len(summary['items'])}"
    )
    return summary


def main() -> None:
    evaluate_policy_checkpoint(default_checkpoint_path())


if __name__ == "__main__":
    main()
