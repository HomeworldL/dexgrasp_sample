import os
from typing import Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def sample_surface_o3d(
    obj_path: str,
    n_points: int = 4096,
    method: str = "poisson",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample surface points and normals from a mesh OBJ using Open3D."""
    if o3d is None:
        raise RuntimeError("open3d is required for sampling. Install open3d (`pip install open3d`).")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ path not found: {obj_path}")

    mesh_o3d = o3d.io.read_triangle_mesh(obj_path)
    if not mesh_o3d.has_triangles():
        raise RuntimeError(f"Loaded mesh has no triangles: {obj_path}")

    if not mesh_o3d.has_vertex_normals():
        mesh_o3d.compute_vertex_normals()

    if method == "uniform":
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
    elif method == "poisson":
        try:
            pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
        except Exception:
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
    else:
        raise ValueError("Unknown sampling method: choose 'uniform' or 'poisson'")

    pts = np.asarray(pcd.points, dtype=np.float32)
    norms = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(pts)
    return pts, norms


def preview_pointcloud_with_normals(pts: np.ndarray, norms: np.ndarray) -> None:
    if o3d is None:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(norms)
    o3d.visualization.draw_geometries([pcd])
