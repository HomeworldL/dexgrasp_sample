import torch
import math
from typing import Sequence, Optional
import numpy as np

torch.random.manual_seed(0)


def sample_grasp_frames(
    points,
    normals,
    d_min: float = 0.015,
    d_max: float = 0.05,
    Nd: int = 4,
    rot_n: int = 8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    max_points: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized sampling of grasp frames.

    For each input point we uniformly sample Nd offsets in [d_min, d_max] along the point normal,
    and for each offset we generate rot_n rotations around the local z (normal).
    The function returns a tensor of homogeneous transforms with shape (M,4,4),
    where M = N_selected * Nd * rot_n.

    Args:
        points: (N,3) array-like (numpy or torch)
        normals: (N,3) array-like (numpy or torch)
        d_min: minimum offset distance (meters if points are in meters)
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
    d_samples = d_min + torch.rand((N, Nd), dtype=dtype, device=device) * (
        d_max - d_min
    )  # (N, Nd)

    # compute origins for each (point, sampled offset): origins_pk (N, Nd, 3)
    origins_pk = pts.unsqueeze(1) + z.unsqueeze(1) * d_samples.unsqueeze(
        2
    )  # (N, Nd, 3)

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
    origins_pnr = origins_pk.unsqueeze(2).expand(-1, -1, rot_n, -1)  # (N,Nd,rot_n,3)

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


def farthest_point_sampling(points, k, seed=None):
    """
    Simple FPS implementation (numpy).
    points: (N,3)
    returns: indices of selected points length k
    """
    N = points.shape[0]
    if k >= N:
        return np.arange(N, dtype=int)
    rng = np.random.default_rng(seed)
    indices = np.empty(k, dtype=int)
    # start with a random point
    indices[0] = rng.integers(N)
    # distances to nearest selected point
    dists = np.full(N, np.inf)
    # update distances iteratively
    selected = points[indices[0:1]]  # shape (1,3)
    for i in range(1, k):
        # compute squared distances to newest selected point
        new_d = np.sum((points - selected[-1]) ** 2, axis=1)
        # update min distances
        dists = np.minimum(dists, new_d)
        # pick farthest
        idx = np.argmax(dists)
        indices[i] = idx
        selected = points[indices[: i + 1]]  # append; small overhead but ok
    return indices


def downsample_fps(pcs, norms, k, seed=None):
    idx = farthest_point_sampling(pcs, k, seed=seed)
    return pcs[idx], norms[idx], idx
