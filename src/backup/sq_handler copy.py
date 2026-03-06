# src/sq_handler.py
import math
from typing import Sequence, Dict, Any, Tuple, Optional
import numpy as np
import torch
import trimesh
import open3d as o3d
from utils.utils_vis import generate_ncolors

def _ensure_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)

class SQHandler:
    """
    Minimal SQHandler that takes:
      - points_dict: {'s1': np.ndarray([N,3]), 's2': ...}
      - ems_params_dict: {'s1': {'shape':(2,), 'scale':(3,), 'rotation':(3,3), 'translation':(3,)}, ...}
    Rotation is expected to already be a 3x3 matrix.
    """

    def __init__(self, points_dict: Dict[str, np.ndarray], ems_params_dict: Dict[str, Dict[str, Any]], name: Optional[str]=None):
        # preserve order of keys from points_dict
        self.keys = list(points_dict.keys())
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.name = name

        # store point clouds per primitive
        self.points_dict = {k: _ensure_numpy(points_dict[k]) for k in self.keys}

        # ems params dict (may contain None for some keys)
        self.ems_params = {}
        for k in self.keys:
            p = ems_params_dict.get(k, None)
            if p is None:
                self.ems_params[k] = None
            else:
                self.ems_params[k] = {
                    "shape": _ensure_numpy(p.get("shape")),
                    "scale": _ensure_numpy(p.get("scale")),
                    "rotation": _ensure_numpy(p.get("rotation")),   # already 3x3
                    "translation": _ensure_numpy(p.get("translation")),
                }

        # build arrays for quick access (order matches self.keys)
        scales = []
        rotations = []
        translations = []
        exponents = []
        exist = []
        for k in self.keys:
            params = self.ems_params[k]
            if params is None:
                scales.append(np.zeros(3))
                rotations.append(np.eye(3))
                translations.append(np.zeros(3))
                exponents.append(np.zeros(2))
                exist.append(0.0)
            else:
                scales.append(params["scale"])
                rotations.append(params["rotation"])
                translations.append(params["translation"])
                exponents.append(params["shape"])
                exist.append(1.0)
        self.scale = np.vstack(scales) if len(scales)>0 else np.zeros((0,3))
        self.rotation = np.stack(rotations, axis=0) if len(rotations)>0 else np.zeros((0,3,3))
        self.translation = np.vstack(translations) if len(translations)>0 else np.zeros((0,3))
        self.exponents = np.vstack(exponents) if len(exponents)>0 else np.zeros((0,2))
        self.exist = np.asarray(exist)

        # colors for visualization
        self.P = len(self.keys)
        self.colors = generate_ncolors(self.P)

    # -------------------------
    # param grid & mesh generation
    # -------------------------
    def _param_grid(self, a: Sequence[float], e: Sequence[float], nu: int, nv: int):
        a = np.asarray(a, dtype=float)
        e = np.asarray(e, dtype=float)
        u = np.linspace(-np.pi, np.pi, nu, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, nv, endpoint=True)
        uu, vv = np.meshgrid(u, v, indexing='xy')
        def f(s, m): return np.sign(np.sin(s)) * (np.abs(np.sin(s)) ** m)
        def g(s, m): return np.sign(np.cos(s)) * (np.abs(np.cos(s)) ** m)
        gv = g(vv, e[0])
        gu = g(uu, e[1])
        fu = f(uu, e[1])
        x = a[0] * gv * gu
        y = a[1] * gv * fu
        z = a[2] * f(vv, e[0])
        pts = np.stack([x, y, z], axis=-1)  # (nv,nu,3)
        return pts.reshape(-1, 3), (nv, nu)

    def _normals_grid(self, pts_flat: np.ndarray, shape_vu: Tuple[int,int]):
        nv, nu = shape_vu
        pts_grid = pts_flat.reshape(nv, nu, 3)
        normals = np.zeros_like(pts_grid)
        for iv in range(nv):
            for iu in range(nu):
                iu2 = (iu + 1) % nu
                iu0 = (iu - 1) % nu
                iv2 = min(iv + 1, nv - 1)
                iv0 = max(iv - 1, 0)
                du = pts_grid[iv, iu2] - pts_grid[iv, iu0]
                dv = pts_grid[iv2, iu] - pts_grid[iv0, iu]
                n = np.cross(du, dv)
                norm = np.linalg.norm(n)
                normals[iv, iu] = (n / norm) if norm > 0 else np.array([0.0,0.0,1.0])
        return normals.reshape(-1, 3)

    def _apply_transform(self, pts_local: np.ndarray, rot: np.ndarray, trans: np.ndarray):
        R = np.asarray(rot, dtype=float)
        t = np.asarray(trans, dtype=float)
        return (R @ pts_local.T).T + t[np.newaxis, :]

    def _superquadric_mesh(self, scale, exponents, rotation, translation, res_u=64, res_v=32):
        pts_local, shape_vu = self._param_grid(scale, exponents, nu=res_u, nv=res_v)
        nv, nu = shape_vu
        faces = []
        for iv in range(nv - 1):
            for iu in range(nu - 1):
                i00 = iv * nu + iu
                i01 = iv * nu + (iu + 1)
                i10 = (iv + 1) * nu + iu
                i11 = (iv + 1) * nu + (iu + 1)
                faces.append([i00, i01, i10])
                faces.append([i10, i01, i11])
            # wrap
            i_last = iv * nu + (nu - 1)
            i_first = iv * nu + 0
            i_next_last = (iv + 1) * nu + (nu - 1)
            i_next_first = (iv + 1) * nu + 0
            faces.append([i_last, i_first, i_next_last])
            faces.append([i_next_last, i_first, i_next_first])
        verts_world = self._apply_transform(pts_local, rotation, translation)
        return verts_world, np.asarray(faces, dtype=np.int64)

    def get_mesh(self, idx: int, resolution: int = 64, colors: bool = True, alpha: float = 1.0) -> trimesh.Trimesh:
        """
        Build and return a single primitive mesh by index (idx).
        - idx: primitive index (0-based)
        - resolution: u-resolution; v-resolution uses max(4, resolution//2)
        - colors: whether to assign per-primitive color
        - alpha: transparency in [0,1]
        """
        # --- compute primitive mesh vertices & faces in world frame ---
        scale = np.asarray(self.scale[idx], dtype=float)
        exps = np.asarray(self.exponents[idx], dtype=float)
        rot = np.asarray(self.rotation[idx], dtype=float) if self.rotation is not None else np.eye(3)
        trans = np.asarray(self.translation[idx], dtype=float) if self.translation is not None else np.zeros(3)

        res_u = int(resolution)
        res_v = max(4, int(resolution // 2))
        verts, faces = self._superquadric_mesh(scale, exps, rot, trans, res_u=res_u, res_v=res_v)

        # --- colors (RGBA uint8) per vertex/face if requested ---
        if colors:
            col = self.colors[idx].astype(np.uint8)  # (3,)
            a8 = np.array([int(np.clip(alpha, 0.0, 1.0) * 255.0)], dtype=np.uint8)
            rgba = np.concatenate([col, a8], axis=0)  # (4,)
            vcols = np.tile(rgba[np.newaxis, :], (verts.shape[0], 1)).astype(np.uint8)
            fcols = np.tile(rgba[np.newaxis, :], (faces.shape[0], 1)).astype(np.uint8)
            return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcols, face_colors=fcols)
        else:
            return trimesh.Trimesh(vertices=verts, faces=faces)


    def get_meshes(self, resolution: int = 64, colors: bool = True, alpha: float = 1.0) -> trimesh.Trimesh:
        """
        Build and return a single merged trimesh composed of all primitives.
        This reuses get_mesh(idx, ... ) for each primitive and concatenates results.
        """
        mesh_list = []
        for idx in range(self.P):
            mesh = self.get_mesh(idx, resolution=resolution, colors=colors, alpha=alpha)
            mesh_list.append(mesh)

        # if only one primitive, return it directly
        if len(mesh_list) == 1:
            return mesh_list[0]

        # concatenate all trimesh objects into a single trimesh
        merged = trimesh.util.concatenate(mesh_list)
        return merged


    # -------------------------
    # pointcloud / segmentation
    # -------------------------
    def get_segmented_pc(self):
        # build colored pointcloud from points_dict, color by primitive key order
        all_pts = []
        all_cols = []
        for idx, k in enumerate(self.keys):
            pts = self.points_dict.get(k)
            if pts is None:
                continue
            all_pts.append(pts)
            col = self.colors[idx].astype(np.float32) / 255.0
            all_cols.append(np.tile(col[np.newaxis, :], (pts.shape[0], 1)))
        if len(all_pts) == 0:
            pc_o3d = o3d.geometry.PointCloud()
            return pc_o3d
        pts_cat = np.vstack(all_pts)
        cols_cat = np.vstack(all_cols)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pts_cat)
        pc_o3d.colors = o3d.utility.Vector3dVector(cols_cat)
        return pc_o3d

    def get_segmented_pcs(self):
        out = []
        for k in self.keys:
            pts = self.points_dict.get(k)
            if pts is None:
                out.append(np.zeros((0,3)))
            else:
                out.append(np.asarray(pts, dtype=float))
        return out

    # -------------------------
    # sampling / offsets
    # -------------------------
    def sample_surface(self, key: str, angular_resolution_deg: float = 6.0):
        idx = self.key_to_idx[key]
        a = self.scale[idx]
        e = self.exponents[idx]
        rot = self.rotation[idx]
        tr = self.translation[idx]
        deg = max(1.0, float(angular_resolution_deg))
        nu = max(8, int(math.ceil(360.0 / deg)))
        nv = max(4, int(math.ceil(180.0 / deg)))
        pts_local, shape_vu = self._param_grid(a, e, nu=nu, nv=nv)
        normals_local = self._normals_grid(pts_local, shape_vu)
        pts_world = self._apply_transform(pts_local, rot, tr)
        normals_world = (np.asarray(rot) @ normals_local.T).T
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        normals_world = normals_world / norms
        return pts_world, normals_world

    def sample_offset_surfaces(self, key: str, d_list: Sequence[float], angular_resolution_deg: float = 6.0):
        pts, norms = self.sample_surface(key, angular_resolution_deg=angular_resolution_deg)
        out = {}
        for d in d_list:
            out[float(d)] = (pts + norms * float(d), norms.copy())
        return out

    def sample_all_primitives_offset(self, d_list: Sequence[float], angular_resolution_deg: float = 6.0):
        result_pts = {float(d): [] for d in d_list}
        result_norms = {float(d): [] for d in d_list}
        for k in self.keys:
            idx = self.key_to_idx[k]
            if self.exist[idx] <= 0.5:
                continue
            off = self.sample_offset_surfaces(k, d_list, angular_resolution_deg=angular_resolution_deg)
            for d, (pts, norms) in off.items():
                result_pts[float(d)].append(pts)
                result_norms[float(d)].append(norms)
        for d in list(result_pts.keys()):
            if len(result_pts[d])>0:
                result_pts[d] = np.vstack(result_pts[d])
                result_norms[d] = np.vstack(result_norms[d])
            else:
                result_pts[d] = np.zeros((0,3))
                result_norms[d] = np.zeros((0,3))
        return result_pts, result_norms

    # -------------------------
    # occupancy
    # -------------------------
    def get_occupancy_numpy(self, points: np.ndarray, key: str):
        pts = np.asarray(points, dtype=float)
        single = False
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
            single = True
        idx = self.key_to_idx[key]
        a = np.asarray(self.scale[idx], dtype=float)
        ex = np.asarray(self.exponents[idx], dtype=float)
        rot = np.asarray(self.rotation[idx], dtype=float)
        trans = np.asarray(self.translation[idx], dtype=float)
        pts_local = (rot.T @ (pts - trans).T).T
        ex0 = max(1e-6, float(ex[0])); ex1 = max(1e-6, float(ex[1]))
        x_s = (pts_local[:,0] / a[0])**2
        y_s = (pts_local[:,1] / a[1])**2
        z_s = (pts_local[:,2] / a[2])**2
        a1 = np.power(x_s, 1.0/ex0)
        a2 = np.power(y_s, 1.0/ex1)
        a_comb = np.power(a1 + a2, ex0 / (ex1 + ex0))
        b = np.power(z_s, 1.0/ex0)
        impl = a_comb + b - 1.0
        inside = impl < 0.0
        return bool(inside[0]) if single else inside

    def get_occupancy_torch(self, points: torch.Tensor, key: str):
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32)
        idx = self.key_to_idx[key]
        a = torch.tensor(self.scale[idx], dtype=torch.float32)
        ex = torch.tensor(self.exponents[idx], dtype=torch.float32)
        rot = torch.tensor(self.rotation[idx], dtype=torch.float32)
        trans = torch.tensor(self.translation[idx], dtype=torch.float32)
        pts_local = (rot.t() @ (points - trans).t()).t()
        ex0 = torch.clamp(ex[0], min=1e-6); ex1 = torch.clamp(ex[1], min=1e-6)
        x_s = (pts_local[:,0] / a[0])**2
        y_s = (pts_local[:,1] / a[1])**2
        z_s = (pts_local[:,2] / a[2])**2
        a1 = torch.pow(x_s, 1.0/ex0)
        a2 = torch.pow(y_s, 1.0/ex1)
        a_comb = torch.pow(a1 + a2, ex0 / (ex1 + ex0))
        b = torch.pow(z_s, 1.0/ex0)
        impl = a_comb + b - 1.0
        return impl < 0.0

    def points_in_any_primitive(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        N = pts.shape[0]
        inside_any = np.zeros((N,), dtype=bool)
        for k in self.keys:
            idx = self.key_to_idx[k]
            if self.exist[idx] <= 0.5:
                continue
            inside_any |= self.get_occupancy_numpy(pts, k)
        return inside_any
