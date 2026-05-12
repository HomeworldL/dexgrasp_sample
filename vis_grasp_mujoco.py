"""Replay stored grasp states with MuJoCo extforce validation.

It loads qpos_prepared and qpos_squeeze for one object-scale directory, then
runs sim_under_extforce with viewer rendering for interactive inspection.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from src.mj_ho import MjHO
from utils.utils_file import (
    DEFAULT_RUN_CONFIG_PATH,
    data_generated_dataset_root_cfg,
    graspdata_tag_cfg,
    hand_anchor_params_cfg,
    hand_profile_cfg,
    hand_root_stabilization_cfg,
    load_run_config,
    object_profile_cfg,
    objdata_tag_cfg,
)
from utils.utils_seed import set_seed

LOGGER = logging.getLogger(__name__)


def _hold_viewer(mjho: MjHO) -> None:
    while mjho._viewer_alive():
        mjho._render_viewer()
        time.sleep(1.0 / 60.0)


def _resolve_qpos_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(
        f"Unsupported qpos dtype '{dtype_name}'. Expected 'float32' or 'float64'."
    )


def _load_grasp_arrays(
    grasp_dir: Path,
    qpos_dtype: np.dtype,
    grasp_h5_name: str,
    grasp_npy_name: str,
) -> Dict[str, np.ndarray]:
    grasp_npy_path = grasp_dir / str(grasp_npy_name)
    grasp_h5_path = grasp_dir / str(grasp_h5_name)
    required_keys = ("qpos_prepared", "qpos_squeeze")

    if grasp_npy_path.exists():
        payload = np.load(grasp_npy_path, allow_pickle=True).item()
        missing = [key for key in required_keys if key not in payload]
        if not missing:
            return {
                key: np.asarray(payload[key], dtype=qpos_dtype) for key in required_keys
            }

    if not grasp_h5_path.exists():
        raise FileNotFoundError(
            f"Neither {grasp_npy_path} nor {grasp_h5_path} contains usable grasp arrays."
        )

    with h5py.File(grasp_h5_path, "r") as handle:
        missing = [key for key in required_keys if key not in handle]
        if missing:
            raise KeyError(f"Missing datasets {missing} in {grasp_h5_path}")
        return {
            key: np.asarray(handle[key][:], dtype=qpos_dtype) for key in required_keys
        }


def visualize_extforce_grasps(
    cfg: Dict,
    config_path: str,
    object_scale_dir: Path,
    qpos_dtype_name: str,
) -> int:
    qpos_dtype = _resolve_qpos_dtype(qpos_dtype_name)
    asset_dir, grasp_dir = _resolve_asset_and_grasp_dirs(
        cfg=cfg,
        config_path=config_path,
        object_scale_dir=object_scale_dir,
    )
    grasp_arrays = _load_grasp_arrays(
        grasp_dir,
        qpos_dtype=qpos_dtype,
        grasp_h5_name=str(cfg["data"]["h5_name"]),
        grasp_npy_name=str(cfg["data"]["npy_name"]),
    )
    qpos_squeeze = grasp_arrays["qpos_squeeze"]
    qpos_prepared_raw = grasp_arrays["qpos_prepared"]

    if qpos_squeeze.shape[0] <= 0:
        raise ValueError(f"No grasps found in {grasp_dir}")
    if qpos_squeeze.shape[0] != qpos_prepared_raw.shape[0]:
        raise ValueError(
            f"Mismatched grasp counts in {grasp_dir}: "
            f"qpos_squeeze={qpos_squeeze.shape[0]}, qpos_prepared={qpos_prepared_raw.shape[0]}"
        )

    object_name = asset_dir.parent.name
    object_mjcf_path = (asset_dir / "object.xml").resolve()
    if not object_mjcf_path.exists():
        raise FileNotFoundError(f"Object MJCF not found: {object_mjcf_path}")

    root_stabilization = hand_root_stabilization_cfg(cfg)
    mjho_valid = MjHO(
        {"name": object_name, "xml_abs": str(object_mjcf_path)},
        cfg["hand"]["xml_path"],
        anchor_params=hand_anchor_params_cfg(cfg),
        hand_profile=hand_profile_cfg(cfg),
        object_profile=object_profile_cfg(cfg),
        root_stabilization=root_stabilization,
        object_fixed=False,
        visualize=True,
    )
    extforce_cfg = dict(cfg.get("extforce", {}))
    extforce_cfg.pop("visualize", None)
    extforce_cfg.pop("grip_delta", None)

    success_count = 0
    for grasp_idx in range(qpos_squeeze.shape[0]):
        qpos_target = qpos_squeeze[grasp_idx].copy()
        prepared_joints = qpos_prepared_raw[grasp_idx][7:].copy()
        qpos_prepared = mjho_valid.build_pregrasp_qpos(qpos_target, prepared_joints)
        success, pos_delta, angle_delta = mjho_valid.sim_under_extforce(
            qpos_target,
            qpos_prepared,
            visualize=True,
            **extforce_cfg,
        )
        success_flag = bool(success)
        success_count += int(success_flag)
        LOGGER.info(
            "[%s] grasp_idx=%d success=%s pos_delta=%.6f angle_delta=%.6f",
            asset_dir.name,
            grasp_idx,
            success_flag,
            float(pos_delta),
            float(angle_delta),
        )

    LOGGER.info(
        "[%s] extforce replay finished success=%d/%d",
        asset_dir.name,
        success_count,
        int(qpos_squeeze.shape[0]),
    )
    _hold_viewer(mjho_valid)
    return success_count


def _resolve_asset_and_grasp_dirs(
    cfg: Dict,
    config_path: str,
    object_scale_dir: Path,
) -> tuple[Path, Path]:
    object_scale_dir = object_scale_dir.resolve()
    grasp_h5_name = str(cfg["data"]["h5_name"])
    grasp_npy_name = str(cfg["data"]["npy_name"])
    has_object_xml = (object_scale_dir / "object.xml").exists()
    has_grasp = (object_scale_dir / grasp_h5_name).exists() or (
        object_scale_dir / grasp_npy_name
    ).exists()
    dataset_root = Path(data_generated_dataset_root_cfg(cfg)).resolve()

    if has_object_xml:
        asset_dir = object_scale_dir
        if has_grasp:
            return asset_dir, asset_dir
        object_name = asset_dir.parent.name
        scale_tag = asset_dir.name
        grasp_dir = (
            dataset_root / graspdata_tag_cfg(cfg, config_path) / object_name / scale_tag
        )
        return asset_dir, grasp_dir

    scale_tag = object_scale_dir.name
    object_name = object_scale_dir.parent.name
    asset_dir = (
        dataset_root / objdata_tag_cfg(cfg, config_path) / object_name / scale_tag
    )
    return asset_dir, object_scale_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all stored dataset grasps under MuJoCo extforce replay."
    )
    parser.add_argument(
        "--object-scale-dir",
        type=str,
        required=True,
        help="Object-scale asset dir under objdata_* or grasp-output dir under graspdata_*.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Cast stored qpos arrays to this dtype before simulation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="JSON config path.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable info logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    cfg = load_run_config(args.config)
    set_seed(int(cfg["seed"]))
    visualize_extforce_grasps(
        cfg=cfg,
        config_path=args.config,
        object_scale_dir=Path(args.object_scale_dir).resolve(),
        qpos_dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
