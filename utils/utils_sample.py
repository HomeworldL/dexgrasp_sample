from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.sample import sample_grasp_frames


def encode_h5_str(value: str) -> np.bytes_:
    return np.bytes_(str(value))


def parse_object_scale_key(object_scale_key: str) -> Tuple[str, Optional[float]]:
    if "__" not in object_scale_key:
        return object_scale_key, None
    object_name, suffix = object_scale_key.split("__", 1)
    if not suffix.startswith("scale"):
        return object_name, None
    digits = suffix[len("scale") :]
    if not digits.isdigit():
        return object_name, None
    return object_name, float(int(digits)) / 1000.0


def grasp_outputs_exist(output_dir_abs: str) -> bool:
    out_dir = Path(output_dir_abs)
    return (out_dir / "grasp.h5").exists() and (out_dir / "grasp.npy").exists()


def grasp_h5_nonempty(h5_path: Path) -> Tuple[bool, str]:
    try:
        with h5py.File(h5_path, "r") as hf:
            if "qpos_grasp" not in hf:
                return False, "missing qpos_grasp dataset"
            ds_grasp = hf["qpos_grasp"]
            if len(ds_grasp.shape) == 0:
                return False, "qpos_grasp is scalar"
            if int(ds_grasp.shape[0]) <= 0:
                return False, "qpos_grasp has zero rows"
    except Exception as exc:
        return False, f"failed to read grasp.h5 ({exc})"
    return True, ""


def write_grasp_npy_from_h5(h5_path: Path, npy_path: Path) -> None:
    payload: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as hf:
        for key in ("qpos_init", "qpos_approach", "qpos_prepared", "qpos_grasp"):
            if key not in hf:
                raise KeyError(f"Missing dataset '{key}' in {h5_path}")
            payload[key] = np.asarray(hf[key][:], dtype=np.float32)
    np.save(npy_path, payload, allow_pickle=True)


def compose_rot_grasp_to_palm(cfg: Dict) -> np.ndarray:
    transform_cfg = cfg["hand"]["transform"]
    base = np.asarray(transform_cfg["base_rot_grasp_to_palm"], dtype=float)
    extra = transform_cfg["extra_euler"]
    extra_rot = R.from_euler(extra["axis"], float(extra["degrees"]), degrees=True).as_matrix()
    return (base @ extra_rot).T


def sample_frames_from_points(cfg: Dict, pts: np.ndarray, norms: np.ndarray) -> np.ndarray:
    sampling_cfg = cfg["sampling"]
    transforms = sample_grasp_frames(
        pts,
        norms,
        Nd=int(sampling_cfg["Nd"]),
        rot_n=int(sampling_cfg["rot_n"]),
        d_min=float(sampling_cfg["d_min"]),
        d_max=float(sampling_cfg["d_max"]),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=sampling_cfg["max_points"],
    )
    transforms_np = transforms.cpu().numpy()
    del transforms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transforms_np


def build_pose_candidates(cfg: Dict, transforms_np: np.ndarray) -> np.ndarray:
    rot_grasp_to_palm = compose_rot_grasp_to_palm(cfg)
    rotation_matrices = transforms_np[:, :3, :3] @ rot_grasp_to_palm
    positions = transforms_np[:, :3, 3]
    quaternions = R.from_matrix(rotation_matrices).as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)
    return np.concatenate([positions, quaternions], axis=1).astype(np.float32)


def make_qpos_triplets(cfg: Dict, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float32)
    approach_joints = np.asarray(cfg["hand"]["approach_joints"], dtype=np.float32)

    q_expanded = np.tile(prepared_joints, (pose.shape[0], 1)).astype(np.float32)
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1).astype(np.float32)

    n = qpos_prepared_sample.shape[0]
    qpos_approach_sample = qpos_prepared_sample.copy()
    qpos_approach_sample[:, 7:] = np.tile(approach_joints, (n, 1))

    shift_local = np.asarray(cfg["hand"]["shift_local"], dtype=float)
    positions = qpos_prepared_sample[:, :3]
    quats_wxyz = qpos_prepared_sample[:, 3:7]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    offset_world = R.from_quat(quats_xyzw).apply(shift_local)

    qpos_init_sample = qpos_prepared_sample.copy()
    qpos_init_sample[:, :3] = positions + offset_world
    qpos_init_sample[:, 7:] = np.tile(approach_joints, (n, 1))

    return qpos_init_sample, qpos_approach_sample, qpos_prepared_sample
