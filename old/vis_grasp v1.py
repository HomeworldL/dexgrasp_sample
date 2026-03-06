import os
import torch
import math
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

torch.random.manual_seed(0)


def sample_grasp_frames(
    points,
    normals,
    d_list: Sequence[float],
    rot_n: int = 8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    max_points: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized sampling of grasp frames.

    Args:
        points: (N,3) array-like (numpy or torch)
        normals: (N,3) array-like (numpy or torch)
        d_list: list/tuple/1D-array of offsets along normal (same units as points), e.g. [0.005, 0.015]
        rot_n: number of rotations around local +Z (normal) to sample (evenly spaced)
        device: torch.device or None (defaults to cpu)
        dtype: torch dtype
        max_points: if given, randomly sample at most this many base points before expanding offsets/rotations
        eps: small number for numerical stability

    Returns:
        transforms: torch.Tensor of shape (M,4,4) with dtype and device as given.
                    Each 4x4 is [R | t; 0 0 0 1], where z-axis = input normal, x/y are chosen tangents.
    """

    # convert to torch tensors
    if device is None:
        device = torch.device("cpu")
    pts = torch.as_tensor(points, dtype=dtype, device=device)
    nrm = torch.as_tensor(normals, dtype=dtype, device=device)

    if pts.dim() != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shape (N,3)")
    if nrm.dim() != 2 or nrm.shape[1] != 3:
        raise ValueError("normals must be shape (N,3)")
    if pts.shape[0] != nrm.shape[0]:
        raise ValueError("points and normals must have same first dimension")

    N = pts.shape[0]

    # optional subsampling to limit total frames
    if (max_points is not None) and (max_points < N):
        perm = torch.randperm(N, device=device)[:max_points]
        pts = pts[perm]
        nrm = nrm[perm]
        N = pts.shape[0]

    # sanitize normals: replace NaN with z-axis and normalize
    nan_mask = torch.isnan(nrm).any(dim=1)
    if nan_mask.any():
        nrm[nan_mask] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    nrm_len = torch.linalg.norm(nrm, dim=1, keepdim=True).clamp_min(eps)
    z = nrm / nrm_len  # shape (N,3)

    # choose a helper vector that is not parallel to z
    # we'll use [0,1,0] except for those too close to it, then choose [1,0,0]
    candidate = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    alt = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    dot_with_candidate = (z * candidate).sum(dim=1).abs()  # (N,)
    use_alt = dot_with_candidate > 0.99  # nearly parallel -> switch helper
    helper = candidate.unsqueeze(0).repeat(N, 1)
    helper[use_alt] = alt

    # x = normalize(cross(helper, z)); y = cross(z, x)
    x = torch.cross(helper, z, dim=1)
    x_len = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)
    x = x / x_len
    y = torch.cross(z, x, dim=1)

    # assemble frame rotation matrices: columns are x, y, z
    # R_frame: (N, 3, 3)
    R_frame = torch.stack([x, -y, -z], dim=2)  # (N, 3, 3) ; stacking as columns

    # prepare offsets: (K,)
    d_arr = torch.as_tensor(list(d_list), dtype=dtype, device=device).reshape(
        -1
    )  # (K,)
    K = d_arr.shape[0]

    # compute origins for each (point, offset): origins_pk (N, K, 3)
    # expand pts and z to (N,1,3)
    origins_pk = pts.unsqueeze(1) + z.unsqueeze(1) * d_arr.view(1, K, 1)  # (N, K, 3)

    # prepare rotation matrices around local z: Rz (rot_n, 3, 3)
    angles = (2.0 * math.pi) * torch.arange(rot_n, device=device, dtype=dtype) / rot_n
    c = torch.cos(angles)
    s = torch.sin(angles)
    # each Rz_i = [[c,-s,0],[s,c,0],[0,0,1]]
    Rz = torch.zeros((rot_n, 3, 3), device=device, dtype=dtype)
    Rz[:, 0, 0] = c
    Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s
    Rz[:, 1, 1] = c
    Rz[:, 2, 2] = 1.0

    # compute R_total for each point and each rotation: R_total_pr (N, rot_n, 3, 3)
    # R_total = R_frame @ Rz  (matrix multiplication: (N,3,3) x (rot_n,3,3) -> (N,rot_n,3,3))
    # we'll use broadcasting: R_frame.unsqueeze(1) @ Rz.unsqueeze(0)
    R_total_pr = torch.matmul(R_frame.unsqueeze(1), Rz.unsqueeze(0))  # (N, rot_n, 3, 3)

    # expand to include offsets K: we want (N, K, rot_n, 3, 3)
    R_total_pkr = R_total_pr.unsqueeze(1).expand(-1, K, -1, -1, -1)  # (N,K,rot_n,3,3)
    origins_pkr = origins_pk.unsqueeze(2).expand(-1, -1, rot_n, -1)  # (N,K,rot_n,3)

    # flatten to list dimension M = N * K * rot_n
    Np, Kp, Rn = R_total_pkr.shape[0], R_total_pkr.shape[1], R_total_pkr.shape[2]
    M = Np * Kp * Rn
    R_flat = R_total_pkr.reshape(M, 3, 3)
    t_flat = origins_pkr.reshape(M, 3)

    # build homogeneous transforms (M,4,4)
    transforms = (
        torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(M, 1, 1)
    )  # (M,4,4)
    transforms[:, :3, :3] = R_flat
    transforms[:, :3, 3] = t_flat

    return transforms


if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")

    obj_name = "035_power_drill"  # 002_master_chef_can 035_power_drill
    obj_name = ds.id2name[0]
    obj_info = ds.get_info(obj_name)
    # print(obj_info)

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")
    mesh_coacd = ds.get_mesh(obj_name, "coacd")
    convex_pieces = ds.get_mesh(obj_name, "convex_pieces")

    pcs_sample_poisson, norms_sample_poisson = ds.get_point_cloud(
        obj_name, n_points=1024, method="poisson"
    )

    d_list = [0.005, 0.015, 0.025]  # 单位与点云单位一致（例如米）
    rot_n = 8

    ts = time.time()
    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        d_list=d_list,
        rot_n=rot_n,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=None,
    )
    print("sample_grasp_frames time:", time.time() - ts)

    print("sampled frames:", transforms.shape)  # (M,4,4)

    transforms_np = transforms.cpu().numpy()

    # trans to hand coord
    rot_grasp_to_palm = (
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        @ R.from_euler("y", 45, degrees=True).as_matrix()
    ).T

    # N,4,4 -> N, 7 pos quat
    rotation_matrices = transforms_np[:, :3, :3] @ rot_grasp_to_palm

    positions = transforms_np[:, :3, 3]
    rot = R.from_matrix(rotation_matrices)
    quaternions = rot.as_quat()
    quaternions = np.roll(quaternions, shift=1, axis=1)
    pose = np.concatenate(
        [
            positions,
            quaternions,
        ],
        axis=1,
    )

    xml_path = os.path.join(
        os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
    )
    target_body_params = {}
    for finger in range(1, 6):
        for digit, value in [("3", (0.6, 1.0)), ("4_end", (0.3, 0.2))]:
            key = f"hand_right_f{finger}{digit}"
            target_body_params[key] = value
    mjho = MjHO(obj_info, xml_path, target_body_params=target_body_params)
    mjho._set_obj_pts_norms(pcs_sample_poisson, norms_sample_poisson)
    rk = RobotKinematics(xml_path)

    # q = np.array(
    #     [
    #         0,
    #         0.3,
    #         0.3,
    #         0.3,
    #         0,
    #         0.3,
    #         0.3,
    #         0.3,
    #         0,
    #         0.3,
    #         0.3,
    #         0.3,
    #         0,
    #         0.3,
    #         0.3,
    #         0.3,
    #         1.6,
    #         0.0,
    #         0.3,
    #         0.3,
    #     ]
    # )
    q = np.array(
        [
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.5,
            0.5,
            0.5,
            1.6,
            0.0,
            0.3,
            0.3,
        ]
    )

    q_expanded = np.tile(q, (pose.shape[0], 1))
    qpos_init_prepared = np.concatenate([pose, q_expanded], axis=1)
    print(f"qpos_init_prepared shape: {qpos_init_prepared.shape}")  # (M, 26)

    collision_flags = []
    valid_flags = []
    for i in tqdm(range(qpos_init_prepared.shape[0]), desc="碰撞检测"):
        mjho.set_hand_qpos(qpos_init_prepared[i])

        hand_qpos = mjho.get_hand_qpos()
        col0 = rk.get_posed_meshes(hand_qpos, kind="collision")

        is_contact = mjho.is_contact()
        collision_flags.append(is_contact)
        if not is_contact:
            # 抓取，获取抓取点，计算FC指标
            results = mjho.sim_grasp()
            ho_contact, hh_contact = mjho.get_contact_info()

            contact_pos = [h["contact_pos"] for h in ho_contact]
            contact_pos = np.array(contact_pos)

            break
            # # 计算FC
            # fc =
            # if fc > 0.5:
            #     valid_flags.append(True)

    collision_flags = np.array(collision_flags)

    qpos_init_processed = qpos_init_prepared[~collision_flags]
    print(f"初始采样构型数: {transforms.shape[0]}")
    print(f"处理后无碰撞构型数: {qpos_init_processed.shape[0]}")
    print(f"碰撞率: {(np.sum(collision_flags) / transforms.shape[0]):.2%}")

    print(f"ho_contact: {ho_contact}")
    print(f"contact_pos: {contact_pos}")
    # exit()

    final_hand_qpos, history = results
    tip_positions = np.concatenate(history["tip_positions"])
    pts_target = np.concatenate(history["pts_target"])
    print(f"tip_positions shape: {tip_positions.shape}")
    print(f"pts_target shape: {pts_target.shape}")
    # print(f"history tip_positions: {history['tip_positions']}")
    # exit()

    hand_qpos = mjho.get_hand_qpos()
    col = rk.get_posed_meshes(hand_qpos, kind="collision")

    # vis
    meshes_for_vis = {
        "mesh_base": mesh_base,
        "mesh_coacd": mesh_coacd,
        "convex_pieces": convex_pieces,
        "mesh_hand_col": col,
        "mesh_hand_col0": col0,
    }

    pointclouds_for_vis = {
        "pcs_sample_poisson": (pcs_sample_poisson, None),
        "tip_positions": (tip_positions, None),
        "pts_target": (pts_target, None),
        "contact_pos": (contact_pos, None),
    }

    num_samples = min(50, pose.shape[0])
    random_indices = np.random.choice(pose.shape[0], size=num_samples, replace=False)
    frame_for_vis = pose[random_indices]

    # visualize_with_viser(meshes_for_vis, pointclouds_for_vis, frame_for_vis)
    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
