import os
import torch
import math
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

torch.random.manual_seed(0)


def sample_grasp_frames(
    points,
    normals,
    d_max: float = 0.08,
    Nd: int = 4,
    rot_n: int = 8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    max_points: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized sampling of grasp frames.

    For each input point we uniformly sample Nd offsets in [0, d_max] along the point normal,
    and for each offset we generate rot_n rotations around the local z (normal).
    The function returns a tensor of homogeneous transforms with shape (M,4,4),
    where M = N_selected * Nd * rot_n.

    Args:
        points: (N,3) array-like (numpy or torch)
        normals: (N,3) array-like (numpy or torch)
        d_max: maximum offset distance (meters if points are in meters)
        Nd: number of offsets to sample per point (integer)
        rot_n: number of rotations around local +Z (normal) to sample (evenly spaced)
        device: torch.device or None (defaults to cpu)
        dtype: torch dtype
        max_points: if given, randomly sample at most this many base points before expanding offsets/rotations
        eps: small number for numerical stability

    Returns:
        transforms: torch.Tensor of shape (M,4,4) with dtype and device as given.
                    Each 4x4 is [R | t; 0 0 0 1], where z-axis = input normal, x/y are chosen tangents.
    """
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
    R_frame = torch.stack([x, -y, -z], dim=2)  # (N, 3, 3)

    # sample Nd offsets uniformly in [0, d_max] for each point -> shape (N, Nd)
    # sample on the requested device/dtype (use torch.rand for uniform)
    d_samples = torch.rand((N, Nd), dtype=dtype, device=device) * float(d_max)  # (N, Nd)

    # compute origins for each (point, sampled offset): origins_pk (N, Nd, 3)
    origins_pk = pts.unsqueeze(1) + z.unsqueeze(1) * d_samples.unsqueeze(2)  # (N, Nd, 3)

    # prepare rotation matrices around local z: Rz (rot_n, 3, 3)
    angles = (2.0 * math.pi) * torch.arange(rot_n, device=device, dtype=dtype) / rot_n
    c = torch.cos(angles)
    s = torch.sin(angles)
    Rz = torch.zeros((rot_n, 3, 3), device=device, dtype=dtype)
    Rz[:, 0, 0] = c
    Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s
    Rz[:, 1, 1] = c
    Rz[:, 2, 2] = 1.0

    # compute R_total for each point and each rotation: R_total_pr (N, rot_n, 3, 3)
    R_total_pr = torch.matmul(R_frame.unsqueeze(1), Rz.unsqueeze(0))  # (N, rot_n, 3, 3)

    # expand to include offsets Nd: we want (N, Nd, rot_n, 3, 3)
    R_total_pnr = R_total_pr.unsqueeze(1).expand(-1, Nd, -1, -1, -1)  # (N,Nd,rot_n,3,3)
    origins_pnr = origins_pk.unsqueeze(2).expand(-1, -1, rot_n, -1)    # (N,Nd,rot_n,3)

    # flatten to list dimension M = N * Nd * rot_n
    Np, Nd_p, Rn = R_total_pnr.shape[0], R_total_pnr.shape[1], R_total_pnr.shape[2]
    M = Np * Nd_p * Rn
    R_flat = R_total_pnr.reshape(M, 3, 3)
    t_flat = origins_pnr.reshape(M, 3)

    # build homogeneous transforms (M,4,4)
    transforms = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(M, 1, 1)
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

    ts = time.time()
    transforms = sample_grasp_frames(
        pcs_sample_poisson,
        norms_sample_poisson,
        d_max=0.06,
        Nd=2,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_points=None,
    )
    transforms_np = transforms.cpu().numpy()
    print("Time to sample:", time.time() - ts)
    print("Shape of transforms:", transforms.shape)  # (M,4,4)

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

    # joints
    q = np.array([0, 0.5, 0.4, 0.4] * 2 + [0, 0.4, 0.4, 0.4] * 2 + [1.6, 0.0, 0.3, 0.3])

    q_expanded = np.tile(q, (pose.shape[0], 1))
    qpos_prepared_sample = np.concatenate([pose, q_expanded], axis=1)
    print(f"Shape of qpos_prepared: {qpos_prepared_sample.shape}")  # (M, 26)

    qpos_prepared_list = []
    qpos_grasp_list = []
    ho_contact_list = []
    hh_contact_list = []
    history_list = []
    for i in tqdm(range(qpos_prepared_sample.shape[0]), desc="碰撞检测"):
        mjho.set_hand_qpos(qpos_prepared_sample[i])

        hand_qpos = mjho.get_hand_qpos()
        col0 = rk.get_posed_meshes(hand_qpos, kind="collision")

        is_contact = mjho.is_contact()
        if not is_contact:
            # sim_grasp
            qpos_grasp, history = mjho.sim_grasp(visualize=True)
            ho_contact, hh_contact = mjho.get_contact_info()
            break

            qpos_prepared_list.append(qpos_prepared_sample[i])
            # qpos_grasp_list.append(qpos_grasp)
            # ho_contact_list.append(ho_contact)
            # hh_contact_list.append(hh_contact)
            # history_list.append(history)

    
    print(f"Num of samples: {transforms.shape[0]}")
    print(f"Num of no-collision samples: {len(qpos_prepared_list)}")
    print(f"Rate of no-collision: {len(qpos_prepared_list) / transforms.shape[0]}")

    idx = 50
    # print(f"idx: {idx}")
    # print(f"qpos_prepared: {qpos_prepared_list[idx]}")
    # print(f"qpos_grasp: {qpos_grasp_list[idx]}")
    # print(f"ho_contact: {ho_contact_list[idx]}")
    # print(f"hh_contact: {hh_contact_list[idx]}")
    # print(f"history: {history_list[idx]}")
    # exit()


    # visualize
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

    frame_origin = np.array(qpos_prepared_list)[:,:3]
    
    pointclouds_for_vis = {
        "pcs_sample_poisson": (pcs_sample_poisson, None),
        "frame_origin": (frame_origin, None),
    }

    # visualize_with_viser(meshes_for_vis, pointclouds_for_vis, frame_for_vis)
    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
