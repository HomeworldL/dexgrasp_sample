import math
import random
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# Keep deterministic behavior aligned with global seed defaults.
random.seed(0)
np.random.seed(0)


def sample_grasp_frames(
    points,
    normals,
    d_min: float = 0.015,
    d_max: float = 0.05,
    Nd: int = 4,
    rot_n: int = 8,
    dtype=jnp.float32,
    max_points: Optional[int] = None,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """JAX implementation of grasp frame sampling. Returns numpy array (M,4,4)."""
    pts = np.asarray(points, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be shape (N,3)")
    if nrm.ndim != 2 or nrm.shape[1] != 3:
        raise ValueError("normals must be shape (N,3)")
    if pts.shape[0] != nrm.shape[0]:
        raise ValueError("points and normals must have same first dimension")

    n_points = pts.shape[0]
    if (max_points is not None) and (max_points < n_points):
        perm = np.random.permutation(n_points)[:max_points]
        pts = pts[perm]
        nrm = nrm[perm]
        n_points = pts.shape[0]

    pts_j = jnp.asarray(pts, dtype=dtype)
    nrm_j = jnp.asarray(nrm, dtype=dtype)

    nan_mask = jnp.isnan(nrm_j).any(axis=1)
    default_n = jnp.tile(jnp.asarray([[0.0, 0.0, 1.0]], dtype=dtype), (n_points, 1))
    nrm_j = jnp.where(nan_mask[:, None], default_n, nrm_j)

    z = nrm_j / jnp.clip(jnp.linalg.norm(nrm_j, axis=1, keepdims=True), a_min=eps)

    candidate = jnp.asarray([0.0, 1.0, 0.0], dtype=dtype)
    alt = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)
    use_alt = jnp.abs(jnp.sum(z * candidate[None, :], axis=1)) > 0.99
    helper = jnp.where(use_alt[:, None], alt[None, :], candidate[None, :])

    x = jnp.cross(helper, z)
    x = x / jnp.clip(jnp.linalg.norm(x, axis=1, keepdims=True), a_min=eps)
    y = jnp.cross(z, x)

    # (N,3,3), columns are x,-y,-z
    r_frame = jnp.stack([x, -y, -z], axis=2)

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    key = jax.random.PRNGKey(int(seed))
    d_samples = d_min + jax.random.uniform(key, shape=(n_points, Nd), dtype=dtype) * (d_max - d_min)

    origins_pk = pts_j[:, None, :] + z[:, None, :] * d_samples[:, :, None]

    angles = (2.0 * math.pi) * jnp.arange(rot_n, dtype=dtype) / float(rot_n)
    c = jnp.cos(angles)
    s = jnp.sin(angles)
    rz = jnp.zeros((rot_n, 3, 3), dtype=dtype)
    rz = rz.at[:, 0, 0].set(c)
    rz = rz.at[:, 0, 1].set(-s)
    rz = rz.at[:, 1, 0].set(s)
    rz = rz.at[:, 1, 1].set(c)
    rz = rz.at[:, 2, 2].set(1.0)

    r_total_pr = jnp.matmul(r_frame[:, None, :, :], rz[None, :, :, :])

    r_total_pnr = jnp.broadcast_to(r_total_pr[:, None, :, :, :], (n_points, Nd, rot_n, 3, 3))
    origins_pnr = jnp.broadcast_to(origins_pk[:, :, None, :], (n_points, Nd, rot_n, 3))

    m = n_points * Nd * rot_n
    r_flat = jnp.reshape(r_total_pnr, (m, 3, 3))
    t_flat = jnp.reshape(origins_pnr, (m, 3))

    transforms = jnp.tile(jnp.eye(4, dtype=dtype)[None, :, :], (m, 1, 1))
    transforms = transforms.at[:, :3, :3].set(r_flat)
    transforms = transforms.at[:, :3, 3].set(t_flat)
    return np.asarray(transforms, dtype=np.float32)


def farthest_point_sampling(points: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Simple FPS implementation (numpy)."""
    n = points.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    indices = np.empty(k, dtype=int)
    indices[0] = rng.integers(n)
    dists = np.full(n, np.inf)
    last_selected = points[indices[0]]
    for i in range(1, k):
        new_d = np.sum((points - last_selected) ** 2, axis=1)
        dists = np.minimum(dists, new_d)
        idx = int(np.argmax(dists))
        indices[i] = idx
        last_selected = points[idx]
    return indices


def downsample_fps(pcs: np.ndarray, norms: np.ndarray, k: int, seed: Optional[int] = None):
    idx = farthest_point_sampling(pcs, k, seed=seed)
    return pcs[idx], norms[idx], idx
