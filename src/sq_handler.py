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
      - params_dict: {'s1': {'shape':(2,), 'scale':(3,), 'rotation':(3,3), 'translation':(3,)}, ...}
    Rotation is expected to already be a 3x3 matrix.
    """

    def __init__(self, points_dict: Dict[str, np.ndarray], params_dict: Dict[str, Dict[str, Any]], name: Optional[str]=None):
        # preserve order of keys from points_dict
        self.keys = list(points_dict.keys())
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.name = name

        # store point clouds per primitive
        self.points_dict = {k: _ensure_numpy(points_dict[k]) for k in self.keys}
        self.points = [self.points_dict[k] for k in self.keys]
        # print(self.points.shape)
        # print(f"points_dict: {points_dict}")
        # print(f"params_dict: {params_dict}")

        # ems params dict (may contain None for some keys)
        self.params_dict = {}
        for k in self.keys:
            p = params_dict.get(k, None)
            self.params_dict[k] = {
                "shape": _ensure_numpy(p.get("shape")),
                "scale": _ensure_numpy(p.get("scale")),
                "rotation": _ensure_numpy(p.get("rotation")),   # already 3x3
                "translation": _ensure_numpy(p.get("translation")),
            }

        # build arrays for quick access (order matches self.keys)
        exponents = []
        scales = []
        rotations = []
        translations = []
        for k in self.keys:
            params = self.params_dict[k]
            exponents.append(params["shape"])
            scales.append(params["scale"])
            rotations.append(params["rotation"])
            translations.append(params["translation"])
        self.exponents = np.vstack(exponents)
        self.scale = np.vstack(scales)
        self.rotation = np.stack(rotations, axis=0)
        self.translation = np.vstack(translations)
        
        # colors for visualization
        self.P = len(self.keys)
        self.colors = generate_ncolors(self.P)


    def print_info(self):
        print(f"Loaded SQHandler with {self.P} primitives.")
        print(f"Exponents: {self.exponents}")
        print(f"Scales: {self.scale}")
        print(f"Rotations: {self.rotation}")
        print(f"Translations: {self.translation}")

    # -------------------------
    # pointcloud / segmentation
    # -------------------------
    def get_segmented_pc(self, index, colors=True):
        if index > self.P-1:
            raise ValueError(f"Index {index} is out of range [0, {self.P-1}]")

        pts = np.asarray(self.points[index])
        # If pts is 1D of length 3, convert to (1,3)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        # Force dtype to float64 for Open3D
        pts = pts.astype(np.float64)

        if colors:
            col = np.asarray(self.colors[index], dtype=np.float64) / 255.0
            # make color array matching points
            cols = np.tile(col[np.newaxis, :], (pts.shape[0], 1))
            return pts, cols
        else:
            return pts, None
    
    def get_segmented_pcs(self, colors=True):
        pcs = []
        cols = []
        
        for i in range(self.P):
            pcs.append(self.get_segmented_pc(i, colors)[0])
            cols.append(self.get_segmented_pc(i, colors)[1])

            print(f"pcs[{i}]: {pcs[i].shape}")

        pcs = np.concatenate(pcs)
        if colors:
            cols = np.concatenate(cols)
            return pcs, cols
        else:
            return pcs, None

    # -------------------------
    # param grid & mesh generation
    # -------------------------
    def get_meshes(self, resolution: int = 100, colors=True):
        meshes = []
        for i in range(self.P):
            try:
                mesh = self.get_mesh(i, resolution, colors)
                meshes.append(mesh)
            except Exception as e:
                print(f"Error generating mesh for index {i}: {e}")
                meshes.append(None)

        merged = trimesh.util.concatenate(meshes)
        return merged

    def get_mesh(self, index, resolution: int = 100, colors=True):
        if index > self.P-1:
            raise ValueError(f"Index {index} is out of range [0, {self.P-1}]")

        vertices = []
        faces = []
        if colors:
            v_colors = []
            f_colors = []

        mesh = self._superquadric_mesh(
            self.scale[index], self.exponents[index],
            self.rotation[index], self.translation[index], resolution
        )
        
        vertices_cur, faces_cur = mesh

        vertices.append(vertices_cur)
        faces.append(faces_cur) 

        if colors:
            cur_color = self.colors[index]
            v_colors.append(np.ones((vertices_cur.shape[0],3)) * cur_color) 
            f_colors.append(np.ones((faces_cur.shape[0],3)) * cur_color)


        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        if colors:
            v_colors = np.concatenate(v_colors)/255.0
            f_colors = np.concatenate(f_colors)/255.0
            mesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors, vertex_colors=v_colors)
        else:
            mesh = trimesh.Trimesh(vertices, faces)
                
        return mesh

    def _superquadric_mesh(self, scale, exponents, rotation, translation, N):
        def f(o, m):
                return np.sign(np.sin(o)) * np.abs(np.sin(o))**m
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o))**m
        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = (np.repeat(v, N))
        if np.linalg.det(rotation) < 0:
            u = u[::-1]
        triangles = []

        x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
        y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
        z = scale[2] * f(v, exponents[0])
        # Set poles to zero to account for numerical instabilities in f and g due to ** operator
        x[:N] = 0.0
        x[-N:] = 0.0
        vertices =  np.concatenate([np.expand_dims(x, 1),
                                    np.expand_dims(y, 1),
                                    np.expand_dims(z, 1)], axis=1)
        vertices =  (rotation @ vertices.T).T + translation  

        triangles = []
        for i in range(N-1):
            for j in range(N-1):
                triangles.append([i*N+j, i*N+j+1, (i+1)*N+j])
                triangles.append([(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)])
        # Connect first and last vertex in each row
        for i in range(N - 1):
            triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
            triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])

        triangles.append([(N-1)*N+(N-1), (N-1)*N, (N-1)])
        triangles.append([(N-1), (N-1)*N, 0])

        return np.array(vertices), np.array(triangles)
    
    # -------------------------
    # sampling
    # -------------------------
    def sample_surface_uv(self, index, Nu=36, Nv=36, d=0):
        # params
        a = np.array(self.scale[index], dtype=float) + float(d)  # scale + offset
        e = np.array(self.exponents[index], dtype=float)
        R = np.array(self.rotation[index], dtype=float)
        t = np.array(self.translation[index], dtype=float)

        # param grid
        u = np.linspace(-np.pi, np.pi, Nu, endpoint=True)
        v = np.linspace(-np.pi / 2.0, np.pi / 2.0, Nv, endpoint=True)
        uu, vv = np.meshgrid(u, v, indexing='xy')  # shape (Nv, Nu)

        # helper f,g
        def f(s, m):
            return np.sign(np.sin(s)) * (np.abs(np.sin(s)) ** m)

        def g(s, m):
            return np.sign(np.cos(s)) * (np.abs(np.cos(s)) ** m)

        # local coordinates (shape Nv x Nu)
        gv = g(vv, e[0])
        gu = g(uu, e[1])
        fu = f(uu, e[1])
        fv = f(vv, e[0])

        x = a[0] * gv * gu
        y = a[1] * gv * fu
        z = a[2] * fv

        pts_local = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (Nv*Nu, 3)

        # compute normals by finite differences using grid (safe & simple)
        # reshape to grid for neighbor diffs
        pts_grid = pts_local.reshape(Nv, Nu, 3)
        normals_grid = np.zeros_like(pts_grid)
        for iv in range(Nv):
            iv0 = max(iv - 1, 0)
            iv1 = min(iv + 1, Nv - 1)
            for iu in range(Nu):
                iu0 = (iu - 1) % Nu
                iu1 = (iu + 1) % Nu
                du = pts_grid[iv, iu1] - pts_grid[iv, iu0]
                dv = pts_grid[iv1, iu] - pts_grid[iv0, iu]
                n = np.cross(du, dv)
                nnorm = np.linalg.norm(n)
                if nnorm == 0:
                    normals_grid[iv, iu] = np.array([0.0, 0.0, 1.0])
                else:
                    normals_grid[iv, iu] = n / nnorm

        normals_local = normals_grid.reshape(-1, 3)

        # transform to world
        pts_world = (R @ pts_local.T).T + t[np.newaxis, :]
        normals_world = (R @ normals_local.T).T
        nrm = np.linalg.norm(normals_world, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        normals_world = normals_world / nrm

        return pts_world, normals_world
    
    def _dtheta(self, theta: float, arclength: float, threshold: float, scale: Sequence[float], sigma: float) -> float:
        """
        Compute dtheta for a given theta according to the MATLAB dtheta function.
        Returns the incremental angle dt (single float).
        """
        # use numpy for trig but keep scalar math
        if theta < threshold:
            # small-theta branch
            dt = abs((arclength / scale[1] + (theta ** sigma)) ** (1.0 / sigma) - (theta))
        elif theta < (np.pi / 2.0 - threshold):
            # mid-range branch
            c = np.cos(theta)
            s = np.sin(theta)
            numerator = (c ** 2) * (s ** 2)
            denom = (scale[0] ** 2) * (np.abs(c) ** (2.0 * sigma)) * (s ** 4) + \
                    (scale[1] ** 2) * (np.abs(s) ** (2.0 * sigma)) * (c ** 4)
            denom = max(denom, 1e-16)
            dt = (arclength / sigma) * np.sqrt(numerator / denom)
        else:
            # near pi/2 branch (symmetric to small-theta)
            dt = abs((arclength / scale[0] + (np.pi / 2.0 - theta) ** (sigma)) ** (1.0 / sigma) - (np.pi / 2.0 - theta))
        return float(dt)

    def _uniformSampledSuperellipse(self, sigma: float, scale: Sequence[float], arclength: float) -> np.ndarray:
        """
        Port of MATLAB uniformSampledSuperellipse (simplified: returns theta array only).
        Produces theta values spanning [0, pi/2] using the two-phase integration method.
        """
        threshold = 1e-2
        num_limit = 10000

        # first ascending phase from 0 up to near pi/4
        theta1 = np.zeros(num_limit, dtype=np.float64)
        theta1[0] = 0.0
        m = 1
        for m in range(1, num_limit):
            dt = self._dtheta(theta1[m - 1], arclength, threshold, scale, sigma)
            theta_temp = theta1[m - 1] + dt
            if theta_temp > (np.pi / 4.0):
                break
            theta1[m] = theta_temp
        theta1 = theta1[:m]  # keep 0..m-1

        # second descending phase from pi/2 down to near pi/4
        theta2 = np.zeros(num_limit, dtype=np.float64)
        theta2[0] = np.pi / 2.0
        m2 = 1
        for m2 in range(1, num_limit):
            dt = self._dtheta(theta2[m2 - 1], arclength, threshold, scale, sigma)
            theta_temp = theta2[m2 - 1] - dt
            if theta_temp <= (np.pi / 4.0):
                break
            theta2[m2] = theta_temp
        theta2 = theta2[:m2]

        # concatenate like MATLAB: theta = [theta1, flip(theta2)]
        if theta2.size > 0:
            theta = np.concatenate([theta1, theta2[::-1]])
        else:
            theta = theta1.copy()

        return theta
    
    def _angle2points(self, theta: np.ndarray, scale: Sequence[float], sigma: float) -> np.ndarray:
        """
        Map angles to 2D superellipse points (MATLAB angle2points port).
        Inputs:
            theta: (N,) array
            scale: [s1, s2]
            sigma: exponent
        Returns:
            2 x N numpy array, first row = x, second row = y (same layout as MATLAB)
        """
        th = np.asarray(theta)
        # sign(cos) * |cos|^sigma
        c = np.sign(np.cos(th)) * (np.abs(np.cos(th)) ** sigma)
        s = np.sign(np.sin(th)) * (np.abs(np.sin(th)) ** sigma)
        x = scale[0] * c
        y = scale[1] * s
        return np.vstack((x, y))

    def sample_surface(self, index: int) -> np.ndarray:
        """
        Sample points on the superquadric primitive at `index` using the spherical-product
        method adapted from your MATLAB code. Returns points in world coordinates.
        """
        if index < 0 or index >= self.P:
            raise ValueError(f"Index {index} out of range [0, {self.P-1}]")

        # retrieve primitive parameters
        exps = np.asarray(self.exponents[index], dtype=np.float64)  # [eps1, eps2]
        scale = np.asarray(self.scale[index], dtype=np.float64)     # [a1, a2, a3]
        R = np.asarray(self.rotation[index], dtype=np.float64)      # 3x3
        t = np.asarray(self.translation[index], dtype=np.float64)   # (3,)

        eps1 = max(0.1, float(exps[0]))
        eps2 = max(0.1, float(exps[1]))
        a1, a2, a3 = float(scale[0]), float(scale[1]), float(scale[2])

        # arclength same as MATLAB example: min(a)/5
        arclength = max(0.005, float(np.min(scale) / 5.0))

        # 1) sample latitude (theta1) and compute point1 using angle2points
        theta1 = self._uniformSampledSuperellipse(eps1, [(a1 + a2) / 2.0, a3], arclength)
        if theta1.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        
        # print(f"theta1: {theta1}")
        # print(f"theta1 shape: {theta1.shape}")
        # exit()

        # point1: 2 x N (x factor , z factor*scale) as MATLAB angle2points(theta1, [1, a3], eps1)
        point1 = self._angle2points(theta1, [1.0, a3], eps1)  # first row scales x, second row scales z

        # 2) for each latitude, sample longitude (theta2) and get point2 using angle2points
        point_upper_list = []
        for i in range(point1.shape[1]):
            # denom used to scale arclength for longitude sampling
            denom = point1[0, i]
            if denom <= 1e-12:
                denom = 1e-12
            theta2 = self._uniformSampledSuperellipse(eps2, [a1, a2], arclength / denom)
            if theta2.size == 0:
                continue

            # point2: 2 x M  -> angle2points(theta2, [a1,a2], eps2)
            point2 = self._angle2points(theta2, [a1, a2], eps2)  # shape (2,M)

            # build point_temp (3 x M): [point2 * point1(1,i); ones * point1(2,i)]
            M = point2.shape[1]
            X_block = (point2[0, :] * point1[0, i]).reshape(1, M)
            Y_block = (point2[1, :] * point1[0, i]).reshape(1, M)
            Z_block = (np.ones(M) * point1[1, i]).reshape(1, M)
            point_temp = np.vstack((X_block, Y_block, Z_block))  # (3, M)

            num_pt = M
            # Mirror / symmetry blocks following MATLAB logic but avoid creating exact duplicates
            blocks = [point_temp]

            # block: [-pt_x(1 : num_pt-1); pt_y(1 : num_pt-1); pt_z(1 : num_pt-1)]
            if num_pt - 1 >= 1:
                blocks.append(np.vstack((-point_temp[0, :num_pt-1],
                                          point_temp[1, :num_pt-1],
                                          point_temp[2, :num_pt-1])))

            # block: [-pt_x(2:end); -pt_y(2:end); pt_z(2:end)]
            if num_pt >= 2:
                blocks.append(np.vstack((-point_temp[0, 1:],
                                          -point_temp[1, 1:],
                                          point_temp[2, 1:])))

            # block: [pt_x(2:num_pt-1); -pt_y(2:num_pt-1); pt_z(2:num_pt-1)]
            if num_pt - 1 >= 2:
                blocks.append(np.vstack((point_temp[0, 1:num_pt-1],
                                          -point_temp[1, 1:num_pt-1],
                                          point_temp[2, 1:num_pt-1])))

            # concatenate horizontally, filter out empty blocks
            point_upper_i = np.concatenate([b for b in blocks if b.size > 0], axis=1)  # (3, Ki)
            point_upper_list.append(point_upper_i)

        if len(point_upper_list) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # MATLAB: close the seam by setting last latitude cell to its first column
        point_upper_list[-1] = point_upper_list[-1][:, :1].copy()

        # assemble full local points:
        first_block = point_upper_list[0]
        middle_blocks = np.concatenate(point_upper_list[1:], axis=1)
        # mirror middle_blocks across z for lower hemisphere
        lower_blocks = np.vstack((middle_blocks[0, :], middle_blocks[1, :], -middle_blocks[2, :]))
        full_local = np.concatenate([first_block, middle_blocks, lower_blocks], axis=1)

        # deduplicate: use rounding + unique to remove seam duplicates (adjust decimals if needed)
        pts_local = np.unique(np.round(full_local, decimals=10), axis=1)

        # normals local

        # transform to world coordinates
        pts_world = (R @ pts_local).T + t.reshape((1, 3))

        
        # normals world


        return pts_world, None

    def sample_surfaces(self):
        pass