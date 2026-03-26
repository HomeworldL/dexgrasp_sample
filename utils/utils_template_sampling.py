from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.mj_ho import RobotKinematics


@dataclass(frozen=True)
class TemplateAnchorSet:
    positions: np.ndarray
    normals: np.ndarray
    contact_weights: np.ndarray
    distance_weights: np.ndarray


def build_template_anchor_set(
    hand_xml_path: str,
    prepared_joints: np.ndarray,
    target_body_params: Dict[str, list[float]],
) -> TemplateAnchorSet:
    rk = RobotKinematics(hand_xml_path)
    qpos = np.zeros((rk.model.nq,), dtype=np.float64)
    qpos[:7] = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qpos[7:] = np.asarray(prepared_joints, dtype=np.float64)
    rk.forward_kinematics(qpos)

    body_name_to_id = {}
    for body_id in range(rk.model.nbody):
        body_name = rk.model.body(body_id).name
        if body_name is not None:
            body_name_to_id[str(body_name)] = body_id

    positions = []
    normals = []
    contact_weights = []
    distance_weights = []
    for body_name, weights in target_body_params.items():
        if body_name not in body_name_to_id:
            raise ValueError(f"Body name '{body_name}' not found in RobotKinematics model.")
        body_id = body_name_to_id[body_name]
        positions.append(np.asarray(rk.data.xpos[body_id], dtype=np.float64))
        body_xmat = np.asarray(rk.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        normal = body_xmat[:, 2].copy()
        normal /= max(np.linalg.norm(normal), 1e-8)
        normals.append(normal)
        contact_weights.append(float(weights[0]))
        distance_weights.append(float(weights[1]))

    return TemplateAnchorSet(
        positions=np.asarray(positions, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
        contact_weights=np.asarray(contact_weights, dtype=np.float64),
        distance_weights=np.asarray(distance_weights, dtype=np.float64),
    )


def _solve_weighted_rigid_transform_row(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source and target must have shape (B, N, 3).")
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape.")
    if weights.ndim != 1 or weights.shape[0] != source.shape[1]:
        raise ValueError("weights must have shape (N,).")

    weights = weights.to(device=source.device, dtype=source.dtype).view(1, -1, 1)
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    source_centroid = (source * weights).sum(dim=1, keepdim=True) / weight_sum
    target_centroid = (target * weights).sum(dim=1, keepdim=True) / weight_sum
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    cov = torch.matmul((source_centered * weights).transpose(1, 2), target_centered)

    u, _, vh = torch.linalg.svd(cov, full_matrices=False)
    eye = torch.eye(3, dtype=source.dtype, device=source.device).unsqueeze(0).repeat(source.shape[0], 1, 1)
    det_sign = torch.det(torch.matmul(u, vh))
    eye[:, 2, 2] = torch.where(det_sign < 0.0, -1.0, 1.0)
    rot = torch.matmul(torch.matmul(u, eye), vh)
    trans = target_centroid[:, 0, :] - torch.matmul(source_centroid[:, 0, :].unsqueeze(1), rot).squeeze(1)
    return rot, trans


def _as_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def refine_pose_candidates_with_template_alignment(
    cfg: Dict,
    hand_xml_path: str,
    target_body_params: Dict[str, list[float]],
    pose_candidates: np.ndarray,
    obj_points: np.ndarray,
    obj_normals: np.ndarray,
) -> np.ndarray:
    template_cfg = cfg.get("template_sampling")
    if not template_cfg or not bool(template_cfg["enabled"]):
        return np.asarray(pose_candidates, dtype=np.float64)
    if pose_candidates.shape[0] == 0:
        return np.asarray(pose_candidates, dtype=np.float64)

    anchor_set = build_template_anchor_set(
        hand_xml_path=hand_xml_path,
        prepared_joints=np.asarray(cfg["hand"]["prepared_joints"], dtype=np.float64),
        target_body_params=target_body_params,
    )
    keep_topk = min(int(template_cfg["keep_topk"]), int(pose_candidates.shape[0]))
    chunk_size = int(template_cfg["batch_size"])
    opt_steps = int(template_cfg["opt_steps"])
    normal_offset = float(template_cfg["normal_offset"])
    device = _as_device(template_cfg.get("device"))

    points_t = torch.as_tensor(obj_points, dtype=torch.float32, device=device)
    normals_t = torch.as_tensor(obj_normals, dtype=torch.float32, device=device)
    normals_t = normals_t / torch.linalg.norm(normals_t, dim=1, keepdim=True).clamp_min(1e-8)
    anchor_pos_t = torch.as_tensor(anchor_set.positions, dtype=torch.float32, device=device)
    anchor_nrm_t = torch.as_tensor(anchor_set.normals, dtype=torch.float32, device=device)
    contact_w_t = torch.as_tensor(anchor_set.contact_weights, dtype=torch.float32, device=device)
    distance_w_t = torch.as_tensor(anchor_set.distance_weights, dtype=torch.float32, device=device)
    solve_weights = torch.cat([distance_w_t, contact_w_t], dim=0)

    pose_np = np.asarray(pose_candidates, dtype=np.float64)
    quats_xyzw = pose_np[:, [4, 5, 6, 3]]
    hand_rot_np = R.from_quat(quats_xyzw).as_matrix().astype(np.float32)
    hand_pos_t = torch.as_tensor(pose_np[:, :3], dtype=torch.float32, device=device)
    hand_rot_t = torch.as_tensor(hand_rot_np, dtype=torch.float32, device=device)

    refined_pose_chunks = []
    refined_score_chunks = []
    for start in range(0, pose_np.shape[0], chunk_size):
        end = min(start + chunk_size, pose_np.shape[0])
        obj_rot_chunk = hand_rot_t[start:end].clone()
        obj_t_chunk = -torch.matmul(hand_pos_t[start:end].unsqueeze(1), obj_rot_chunk).squeeze(1)

        matched_points_obj = None
        matched_normals_obj = None
        for _ in range(opt_steps):
            anchor_obj = torch.matmul(
                anchor_pos_t.unsqueeze(0) - obj_t_chunk.unsqueeze(1),
                obj_rot_chunk.transpose(1, 2),
            )
            pairwise = torch.cdist(anchor_obj, points_t.unsqueeze(0).expand(end - start, -1, -1))
            nearest_idx = torch.argmin(pairwise, dim=-1)
            matched_points_obj = points_t[nearest_idx]
            matched_normals_obj = normals_t[nearest_idx]

            source = torch.cat(
                [
                    matched_points_obj,
                    matched_points_obj + (normal_offset * matched_normals_obj),
                ],
                dim=1,
            )
            target = torch.cat(
                [
                    anchor_pos_t.unsqueeze(0).expand(end - start, -1, -1),
                    anchor_pos_t.unsqueeze(0).expand(end - start, -1, -1)
                    - (normal_offset * anchor_nrm_t.unsqueeze(0).expand(end - start, -1, -1)),
                ],
                dim=1,
            )
            obj_rot_chunk, obj_t_chunk = _solve_weighted_rigid_transform_row(
                source=source,
                target=target,
                weights=solve_weights,
            )

        if matched_points_obj is None or matched_normals_obj is None:
            raise RuntimeError("Template alignment produced no matched object points.")

        matched_points_hand = torch.matmul(matched_points_obj, obj_rot_chunk) + obj_t_chunk.unsqueeze(1)
        matched_normals_hand = torch.matmul(matched_normals_obj, obj_rot_chunk)
        pos_err = torch.linalg.norm(matched_points_hand - anchor_pos_t.unsqueeze(0), dim=-1)
        normal_dot = torch.sum(matched_normals_hand * (-anchor_nrm_t.unsqueeze(0)), dim=-1).clamp(-1.0, 1.0)
        normal_err = 1.0 - normal_dot
        score = ((pos_err * distance_w_t.unsqueeze(0)) + (normal_err * contact_w_t.unsqueeze(0))).sum(dim=1)
        score = score / (distance_w_t.sum() + contact_w_t.sum()).clamp_min(1e-8)

        hand_pos_chunk = -torch.matmul(obj_t_chunk.unsqueeze(1), obj_rot_chunk.transpose(1, 2)).squeeze(1)
        hand_quat_xyzw = R.from_matrix(obj_rot_chunk.detach().cpu().numpy().astype(np.float64)).as_quat()
        hand_quat_wxyz = np.roll(hand_quat_xyzw, shift=1, axis=1)
        refined_pose = np.concatenate(
            [
                hand_pos_chunk.detach().cpu().numpy().astype(np.float64),
                hand_quat_wxyz.astype(np.float64),
            ],
            axis=1,
        )
        refined_pose_chunks.append(refined_pose)
        refined_score_chunks.append(score.detach().cpu().numpy().astype(np.float64))

    all_pose = np.concatenate(refined_pose_chunks, axis=0)
    all_score = np.concatenate(refined_score_chunks, axis=0)
    keep_idx = np.argsort(all_score)[:keep_topk]
    return all_pose[keep_idx].astype(np.float64, copy=False)
