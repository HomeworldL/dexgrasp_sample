from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from utils.utils_sample import ARRAY_DTYPE, make_qpos_triplets


def _require(cfg: Dict, key: str):
    if key not in cfg:
        raise ValueError(f"pose_preopt.{key} is required when pose_preopt is enabled.")
    return cfg[key]


def _skew(values: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(values[..., 0])
    x, y, z = values[..., 0], values[..., 1], values[..., 2]
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / theta
    k = _skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(
        rotvec.shape[:-1] + (3, 3)
    )
    theta_mat = theta[..., None]
    return eye + torch.sin(theta_mat) * k + (1.0 - torch.cos(theta_mat)) * (k @ k)


def extract_anchor_template(
    mjho, prepared_joints: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return anchor positions and configured axes in the hand-root frame."""
    qpos = np.concatenate(
        [
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
            np.asarray(prepared_joints, dtype=float).reshape(-1),
        ]
    )
    mjho.set_hand_qpos(qpos)
    root_pos = np.asarray(qpos[:3], dtype=float)
    root_rot = R.from_quat(qpos[3:7][[1, 2, 3, 0]]).as_matrix()
    root_rot_t = root_rot.T
    anchor_pos_world = np.vstack(
        [np.asarray(mjho.data.xpos[bid], dtype=float) for bid in mjho.anchor_body_ids]
    )
    anchor_axes_world = []
    for bid, axis_local in zip(mjho.anchor_body_ids, mjho.anchor_plane_axes):
        xmat = np.asarray(mjho.data.xmat[bid], dtype=float).reshape(3, 3)
        axis_world = xmat @ np.asarray(axis_local, dtype=float)
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
        anchor_axes_world.append(axis_world)
    anchor_axes_world = np.vstack(anchor_axes_world)
    anchor_pos_local = (root_rot_t @ (anchor_pos_world - root_pos).T).T
    anchor_axes_local = (root_rot_t @ anchor_axes_world.T).T
    return (
        anchor_pos_local.astype(ARRAY_DTYPE, copy=False),
        anchor_axes_local.astype(ARRAY_DTYPE, copy=False),
    )


def optimize_pose_candidates(
    cfg: Dict,
    pose: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    anchor_pos_local: np.ndarray,
    anchor_axes_local: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float | int]]:
    preopt_cfg = dict(cfg.get("pose_preopt", {}))
    if not bool(preopt_cfg.get("enabled", False)):
        return pose, {"enabled": 0, "elapsed": 0.0, "count": int(pose.shape[0])}

    if pose.size == 0:
        return pose, {"enabled": 1, "elapsed": 0.0, "count": 0}

    device_name = str(_require(preopt_cfg, "device"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = torch.float32
    batch_size = max(int(_require(preopt_cfg, "batch_size")), 1)
    steps = max(int(_require(preopt_cfg, "steps")), 1)
    lr_pos = float(_require(preopt_cfg, "lr_pos"))
    lr_rot = float(_require(preopt_cfg, "lr_rot"))
    target_dist = float(_require(preopt_cfg, "target_dist"))
    max_pos_delta = float(_require(preopt_cfg, "max_pos_delta"))
    max_rot_rad = math.radians(float(_require(preopt_cfg, "max_rot_deg")))
    distance_weight = float(_require(preopt_cfg, "distance_weight"))
    balance_weight = float(_require(preopt_cfg, "balance_weight"))
    normal_weight = float(_require(preopt_cfg, "normal_weight"))
    tangent_weight = float(preopt_cfg.get("tangent_weight", 0.0))
    reg_pos_weight = float(_require(preopt_cfg, "reg_pos_weight"))
    reg_rot_weight = float(_require(preopt_cfg, "reg_rot_weight"))
    loss_mode = str(preopt_cfg.get("loss_mode", "unsigned")).strip().lower()
    if loss_mode not in {"unsigned", "signed"}:
        raise ValueError("pose_preopt.loss_mode must be one of ['unsigned', 'signed'].")

    start = time.perf_counter()
    points_t = torch.as_tensor(points, dtype=dtype, device=device)
    normals_t = torch.as_tensor(normals, dtype=dtype, device=device)
    normals_t = torch.nn.functional.normalize(normals_t, dim=-1, eps=1e-8)
    anchor_local_t = torch.as_tensor(anchor_pos_local, dtype=dtype, device=device)
    axis_local_t = torch.as_tensor(anchor_axes_local, dtype=dtype, device=device)
    pose_out = np.asarray(pose, dtype=ARRAY_DTYPE).copy()

    for start_i in range(0, pose.shape[0], batch_size):
        end_i = min(start_i + batch_size, pose.shape[0])
        pose_batch = np.asarray(pose[start_i:end_i], dtype=np.float32)
        base_pos = torch.as_tensor(pose_batch[:, :3], dtype=dtype, device=device)
        base_quat = pose_batch[:, 3:7]
        base_rot_np = (
            R.from_quat(base_quat[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
        )
        base_rot = torch.as_tensor(base_rot_np, dtype=dtype, device=device)
        delta_pos = torch.zeros_like(base_pos, requires_grad=True)
        delta_rot = torch.zeros((pose_batch.shape[0], 3), dtype=dtype, device=device)
        delta_rot.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [delta_pos], "lr": lr_pos},
                {"params": [delta_rot], "lr": lr_rot},
            ]
        )

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            delta_rot_m = _rotvec_to_matrix(delta_rot)
            rot = delta_rot_m @ base_rot
            anchors = torch.einsum("bij,aj->bai", rot, anchor_local_t)
            anchors = anchors + base_pos[:, None, :] + delta_pos[:, None, :]
            dists_all = torch.cdist(anchors.reshape(-1, 3), points_t)
            min_dist, min_idx = torch.min(dists_all, dim=1)
            min_dist = min_dist.reshape(pose_batch.shape[0], -1)
            nearest_points = points_t[min_idx].reshape(pose_batch.shape[0], -1, 3)
            nearest_normals = normals_t[min_idx].reshape(pose_batch.shape[0], -1, 3)
            if loss_mode == "signed":
                offset = anchors - nearest_points
                signed_dist = torch.sum(offset * nearest_normals, dim=-1)
                tangent = offset - signed_dist[..., None] * nearest_normals
                loss = distance_weight * torch.mean((signed_dist - target_dist) ** 2)
                loss = loss + balance_weight * torch.mean(torch.var(signed_dist, dim=1))
                loss = loss + tangent_weight * torch.mean(torch.sum(tangent**2, dim=-1))
            else:
                loss = distance_weight * torch.mean((min_dist - target_dist) ** 2)
                loss = loss + balance_weight * torch.mean(torch.var(min_dist, dim=1))
            loss = loss + reg_pos_weight * torch.mean(delta_pos**2)
            loss = loss + reg_rot_weight * torch.mean(delta_rot**2)
            if normal_weight > 0.0:
                axes = torch.einsum("bij,aj->bai", rot, axis_local_t)
                axes = torch.nn.functional.normalize(axes, dim=-1, eps=1e-8)
                normal_loss = 1.0 - torch.sum((-axes) * nearest_normals, dim=-1)
                loss = loss + normal_weight * torch.mean(normal_loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta_pos.clamp_(min=-max_pos_delta, max=max_pos_delta)
                rot_norm = torch.linalg.norm(delta_rot, dim=-1, keepdim=True)
                scale = torch.clamp(max_rot_rad / rot_norm.clamp_min(1e-8), max=1.0)
                delta_rot.mul_(scale)

        with torch.no_grad():
            delta_rot_m = _rotvec_to_matrix(delta_rot)
            rot = delta_rot_m @ base_rot
            pos = base_pos + delta_pos
            rot_np = rot.detach().cpu().numpy()
            pos_np = pos.detach().cpu().numpy()
            quat_xyzw = R.from_matrix(rot_np).as_quat().astype(np.float32)
            quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
            pose_out[start_i:end_i, :3] = pos_np
            pose_out[start_i:end_i, 3:7] = quat_wxyz

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - start
    return pose_out.astype(ARRAY_DTYPE, copy=False), {
        "enabled": 1,
        "elapsed": float(elapsed),
        "count": int(pose.shape[0]),
        "steps": int(steps),
        "batch_size": int(batch_size),
    }


def optimize_qpos_triplets(
    cfg: Dict,
    mjho,
    pose: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float | int]]:
    prepared_joints = np.asarray(cfg["hand"]["prepared_joints"], dtype=ARRAY_DTYPE)
    anchor_pos_local, anchor_axes_local = extract_anchor_template(mjho, prepared_joints)
    pose_opt, profile = optimize_pose_candidates(
        cfg,
        pose,
        points,
        normals,
        anchor_pos_local,
        anchor_axes_local,
    )
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose_opt)
    return pose_opt, qpos_init, qpos_approach, qpos_prepared, profile
