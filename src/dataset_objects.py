# src/object_model.py
"""
DatasetObjects

Manage a processed dataset folder (e.g., assets/YCB/ycb_datasets).
 - Build an ordered index of objects and their important files (inertia.obj, simplified.obj, urdf/xml, convex pieces).
 - Sample surface points with Open3D.
 - Load mesh via trimesh.
 - Generic external command runner.
 - SuperDec integration: compute superquadric decomposition for one object or batch for entire dataset.
 - Save SuperDec parameters into JSON.
 - Visualize mesh / sampled pointcloud / convex pieces / SuperDec meshes using viser (fallback open3d).

Code comments and docstrings are in English.
"""

import sys
import os

sys.path.append(os.path.realpath("."))

from collections import OrderedDict
from pathlib import Path

import json
import subprocess
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import trimesh
import open3d as o3d
from utils.utils_file import *
from utils.utils_vis import generate_ncolors
from src.EMS.EMS_recovery import EMS_recovery
from scipy.spatial.transform import Rotation as R

o3d.utility.random.seed(0)
np.random.seed(0)

# Try to import SuperDec dependencies (used only when computing superquadrics)
_superdec_available = False
try:
    import torch
    from omegaconf import OmegaConf
    from superdec.superdec import SuperDec

    # dataloader utilities used for normalize/denormalize
    from superdec.data.dataloader import (
        normalize_points,
        denormalize_outdict,
        denormalize_points,
    )
    from sq_handler import SQHandler

    _superdec_available = True
except Exception:
    _superdec_available = False


class DatasetObjects:
    """
    Represent a dataset root (e.g. assets/YCB/ycb_datasets) and provide utilities.

    Attributes:
        root (str): absolute path to dataset root
        index (OrderedDict): map object_name -> object_info_dict
            object_info_dict keys: name, folder_abs, inertia_abs, simplified_abs, manifold_abs,
                                   coacd_abs, urdf_abs, xml_abs, meshes: {abs:[], rel:[]}, superdec: optional
    """

    def __init__(self, dataset_root: str):
        """
        Initialize dataset index by scanning immediate subfolders (ordered by leading number).
        """
        self.root = os.path.abspath(dataset_root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.index: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._build_index()

        # SuperDec runtime members (initialized on demand)
        self._sd_model = None
        self._sd_device = None
        self._sd_configs = None
        self._sd_ckpt_path = None

    # -------------------------
    # Index building
    # -------------------------
    def _build_index(self) -> None:
        """Scan dataset root and fill self.index with ordered object entries."""
        entries = [
            d
            for d in sorted(os.listdir(self.root))
            if os.path.isdir(os.path.join(self.root, d))
        ]

        # sort by leading number if present
        def leading_num(s: str):
            import re

            m = re.match(r"^(\d+)", s)
            return int(m.group(1)) if m else float("inf")

        entries.sort(key=lambda s: (leading_num(s), s.lower()))

        for name in entries:
            folder = os.path.join(self.root, name)
            info = {"name": name, "folder_abs": os.path.abspath(folder)}

            # pick candidates for main files
            def _first_exists(*rels):
                for r in rels:
                    p = os.path.join(folder, r)
                    if os.path.exists(p):
                        return os.path.abspath(p)
                return None

            info["inertia_abs"] = _first_exists("inertia.obj")
            info["simplified_abs"] = _first_exists("simplified.obj")
            info["manifold_abs"] = _first_exists("manifold.obj")
            info["coacd_abs"] = _first_exists("coacd.obj", "decomposed.obj")
            info["urdf_abs"] = _first_exists(f"{name}.urdf")
            info["xml_abs"] = _first_exists(f"{name}.xml")
            # find meshes (convex pieces) under common subfolders
            meshes = []
            for md in ("meshes", "convex_parts", "parts"):
                mdp = os.path.join(folder, md)
                if os.path.isdir(mdp):
                    for fn in sorted(os.listdir(mdp)):
                        if fn.lower().endswith(".obj"):
                            meshes.append(os.path.abspath(os.path.join(mdp, fn)))
                    if meshes:
                        break
            info["meshes"] = {
                "abs": meshes,
                "rel": [os.path.relpath(p, folder).replace("\\", "/") for p in meshes],
            }
            # placeholder for superdec npz abs paths
            info["superdec_npz"] = _first_exists("superdec.npz")
            # attach metadata if present (metadata.json or dataset_metadata.json)
            meta = None
            for mname in ("metadata.json", "dataset_metadata.json"):
                mp = os.path.join(folder, mname)
                if os.path.exists(mp):
                    try:
                        with open(mp, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        meta = None
                    break
            info["metadata"] = meta
            self.index[name] = info

    def get_index(self) -> OrderedDict:
        """Return the internal ordered index."""
        return self.index

    def get_info(self, name: str) -> Dict[str, Any]:
        return self.index[name]

    @property
    def id2name(self) -> Dict[int, str]:
        return {i: n for i, n in enumerate(self.index.keys())}

    # -------------------------
    # Mesh loading
    # -------------------------
    def load_mesh(self, mesh_path: Optional[str] = None) -> trimesh.Trimesh:
        """
        Load a mesh via trimesh. If mesh_path is None, prefer inertia_abs, then simplified_abs.
        Return a Trimesh (scene concatenated if necessary).
        """
        mp = mesh_path
        if mp is None:
            # find first available per dataset object? but for load_mesh we expect a path or operate on a single object
            raise ValueError("mesh_path must be provided to load_mesh")
        mesh = trimesh.load(mp, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = (
                trimesh.util.concatenate([g for g in mesh.geometry.values()])
                if hasattr(mesh, "geometry")
                else mesh
            )
        return mesh

    def get_mesh(
        self, name: str, mesh_type: str = "inertia", alpha: float = 1.0
    ) -> trimesh.Trimesh:
        if mesh_type not in (
            "inertia",
            "simplified",
            "manifold",
            "coacd",
            "convex_pieces",
        ):
            raise ValueError(f"Unknown mesh type: {mesh_type}")

        info = self.get_info(name)
        if info is None:
            raise KeyError(f"Object '{name}' not found")

        if mesh_type == "convex_pieces":
            piece_paths = info.get("meshes", {}).get("abs", [])
            if not piece_paths:
                raise FileNotFoundError(f"No convex pieces found for {name}")
            pieces = [self.load_mesh(p) for p in piece_paths]

            # assign random RGB per piece, set alpha later
            alpha_byte = np.clip(int(round(float(alpha) * 255.0)), 0, 255)
            colored_pieces = []
            for piece in pieces:
                n_verts = piece.vertices.shape[0]
                rgb = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                rgba = np.tile(
                    np.concatenate([rgb, np.array([alpha_byte], dtype=np.uint8)]),
                    (n_verts, 1),
                )
                # set vertex colors (overwrite existing)
                piece.visual.vertex_colors = rgba
                colored_pieces.append(piece)

            combined = trimesh.util.concatenate(colored_pieces)
            return combined

        # other mesh types: load and return
        key = mesh_type + "_abs"
        mesh_path = info.get(key)
        if not mesh_path:
            raise FileNotFoundError(f"No path for {mesh_type} mesh of {name}")
        mesh = self.load_mesh(mesh_path)
        # set uniform alpha if requested and mesh has vertex colors or to create them
        if 0.0 <= alpha <= 1.0:
            a_byte = np.clip(int(round(float(alpha) * 255.0)), 0, 255)
            n = mesh.vertices.shape[0]
            # if mesh already has vertex_colors, preserve RGB, replace alpha; else set white with alpha
            if (
                getattr(mesh.visual, "vertex_colors", None) is not None
                and mesh.visual.vertex_colors.size
            ):
                vc = np.asarray(mesh.visual.vertex_colors)
                if vc.shape[1] == 3:
                    vc = np.concatenate(
                        [vc, np.full((vc.shape[0], 1), 255, dtype=np.uint8)], axis=1
                    )
                vc[:, 3] = a_byte
                mesh.visual.vertex_colors = vc
            else:
                mesh.visual.vertex_colors = np.tile(
                    np.array([128, 128, 128, a_byte], dtype=np.uint8), (n, 1)
                )
        return mesh

    # -------------------------
    # Surface sampling (Open3D)
    # -------------------------
    def sample_surface_o3d(
        self,
        obj_path: str,
        n_points: int = 4096,
        method: str = "uniform",
        preview: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample surface points from an OBJ using Open3D.

        Returns:
            points (N,3) numpy array, normals (N,3) numpy array
        """
        if o3d is None:
            raise RuntimeError(
                "open3d is required for sampling. Install open3d (`pip install open3d`)."
            )
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ path not found: {obj_path}")

        mesh_o3d = o3d.io.read_triangle_mesh(obj_path)
        if not mesh_o3d.has_triangles():
            raise RuntimeError(f"Loaded mesh has no triangles: {obj_path}")

        if not mesh_o3d.has_vertex_normals():
            mesh_o3d.compute_vertex_normals()

        ts = time.time()
        if method == "uniform":
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
        elif method == "poisson":
            try:
                pcd = mesh_o3d.sample_points_poisson_disk(
                    number_of_points=n_points, init_factor=5
                )
            except Exception:
                pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
        else:
            raise ValueError("Unknown sampling method: choose 'uniform' or 'poisson'")

        # print(f"Sampled {n_points} points in {time.time() - ts:.3f} seconds")

        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_normals():
            norms = np.asarray(pcd.normals, dtype=np.float32)
        else:
            # fallback: zeros
            norms = np.zeros_like(pts)

        if preview:
            self._preview_pointcloud_with_normals(pts, norms)
        return pts, norms

    def get_point_cloud(self, name: str, n_points: int = 4096, method: str = "poisson"):
        info = self.get_info(name)
        if info is None:
            raise KeyError(f"Object '{name}' not found")

        mesh_path = info.get("inertia_abs")

        return self.sample_surface_o3d(mesh_path, n_points, method)

    def _preview_pointcloud_with_normals(self, points: np.ndarray, normals: np.ndarray):
        """Interactive visualization of point cloud with normals"""

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 方法1：基础预览（带法线）
        print("\n=== 点云预览 ===")
        print("- 红色箭头：法线方向")
        print("- WASD键移动，鼠标拖动旋转视角")
        print("- 按'N'切换法线显示")

        # # 确保法线单位化以便于可视化
        # pcd.normals = o3d.utility.Vector3dVector(normalized_normals)

        # 创建坐标系参考
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        # 可视化
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            point_show_normal=True,
            window_name="点云与法线预览",
        )

    # -------------------------
    # SuperDec integration
    # -------------------------
    def _init_superdec(
        self,
        checkpoints_folder: str,
        checkpoint_file: str,
        device: Optional[str] = None,
        config_name: str = "config.yaml",
        lm_optimization: bool = True,
    ):
        """
        Load SuperDec model into memory. Call once before using compute_superdec_*.
        """
        if not _superdec_available:
            raise RuntimeError(
                "superdec environment not available: make sure `superdec` package and its dependencies are importable."
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sd_device = device
        print(f"SuperDec device: {device}")

        ckp_path = os.path.join(checkpoints_folder, checkpoint_file)
        cfg_path = os.path.join(checkpoints_folder, config_name)
        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f"SuperDec checkpoint not found: {ckp_path}")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"SuperDec config not found: {cfg_path}")

        checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
        with open(cfg_path) as f:
            configs = OmegaConf.load(f)
        model = SuperDec(configs.superdec).to(device)
        model.lm_optimization = False
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # keep config and model
        self._sd_model = model
        self._sd_ckpt_path = ckp_path
        self._sd_configs = configs

    def compute_superdec_single(
        self,
        object_name: str,
        target_n_points: int = 4096,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute SuperDec decomposition for a single object (by object_name key in index).
        Returns a dict with denormalized outdict arrays and optionally mesh/pcs saved.
        The result is stored in self.index[object_name]['superdec'].
        """
        if object_name not in self.index:
            raise KeyError(f"Object not found in dataset index: {object_name}")

        if not _superdec_available:
            raise RuntimeError(
                "superdec integration not available. Ensure superdec package and dependencies installed."
            )

        # init model if not already
        if self._sd_model is None:
            raise RuntimeError(
                "superdec model not initialized. Call _init_superdec first."
            )

        entry = self.index[object_name]
        # choose input mesh path: inertia preferred, else simplified, else manifold
        input_mesh = (
            entry.get("inertia_abs")
            or entry.get("simplified_abs")
            or entry.get("manifold_abs")
        )
        if not input_mesh or not os.path.exists(input_mesh):
            raise FileNotFoundError(f"No input mesh found for {object_name}")

        # sample points using Open3D
        pts, norms = self.sample_surface_o3d(
            input_mesh, n_points=target_n_points, method="uniform"
        )
        # convert to numpy float32
        points_np = np.asarray(pts, dtype=np.float32)

        # normalization - use superdec's normalize_points
        if normalize:
            points_normed, translation, scale = normalize_points(points_np)
        else:
            points_normed = points_np.copy()
            translation = np.zeros(3)
            scale = 1.0

        points_t = (
            torch.from_numpy(points_normed).unsqueeze(0).to(self._sd_device).float()
        )

        # forward
        with torch.no_grad():
            outdict = self._sd_model(points_t)
            # move tensors to cpu numpy
            for key in list(outdict.keys()):
                if isinstance(outdict[key], torch.Tensor):
                    outdict[key] = outdict[key].cpu()
            # denormalize
            translation_arr = np.array([translation])
            scale_arr = np.array([scale])
            # TODO: default z_up false
            outdict_den = denormalize_outdict(
                outdict, translation_arr, scale_arr, z_up=False
            )
            points_den = denormalize_points(
                points_t.cpu(), translation_arr, scale_arr, z_up=False
            )

        # store results in index
        # sd_result = {
        #     "pcd": points_den.cpu().numpy()[0],
        #     "assign_matrix": outdict_den["assign_matrix"].numpy()[0],
        #     "scale": outdict_den["scale"].numpy()[0],
        #     "rotation": outdict_den["rotate"].numpy()[0],
        #     "translation": outdict_den["trans"].numpy()[0],
        #     "exponents": outdict_den["shape"].numpy()[0],
        #     "exist": outdict_den["exist"].numpy()[0],
        # }
        # print(f"pcd: {sd_result['pcd'].shape}")
        # print(f"assign_matrix: {sd_result['assign_matrix'].shape}")
        # print(f"scale: {sd_result['scale'].shape}")
        # print(f"rotation: {sd_result['rotation'].shape}")
        # print(f"translation: {sd_result['translation'].shape}")
        # print(f"exponents: {sd_result['exponents'].shape}")
        # print(f"exist: {sd_result['exist'].shape}")
        # entry["superdec"] = sd_result

        # return sd_result

    def compute_superdec_datasets(
        self,
        target_n_points: int = 4096,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute SuperDec for all objects in the dataset in batches of 16.
        For each object, sample points, normalize individually, batch-forward (B<=16),
        denormalize outputs, then save per-object result to <object_folder>/superdec.npz (overwrite).
        Update self.index[name]['superdec_npz'] with the npz absolute path.

        Returns:
            results: dict mapping object_name -> {"npz": path} or {"error": msg}
        """
        if not _superdec_available:
            raise RuntimeError(
                "superdec integration not available. Ensure superdec package and dependencies installed."
            )

        if self._sd_model is None:
            raise RuntimeError(
                "superdec model not initialized. Call _init_superdec first."
            )

        # collect candidate object names (preserve ordering)
        all_names = list(self.index.keys())
        print(f"Computing SuperDec for {len(all_names)} objects...")
        results: Dict[str, Any] = {}

        BATCH_SIZE = 16

        # helper: convert tensor/other to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.numpy()
            return np.asarray(x)

        # iterate in chunks
        for i0 in range(0, len(all_names), BATCH_SIZE):
            chunk_names = all_names[i0 : i0 + BATCH_SIZE]
            print(f"Computing {i0}-{i0 + BATCH_SIZE} objects...")

            # prepare per-chunk lists
            valid_names = []
            pts_norm_list = []
            translations = []
            scales = []
            orig_N_list = []
            orig_pts_list = []
            info_list = []

            # 1) sample & normalize for each object in chunk
            for name in chunk_names:
                info = self.index[name]
                # choose input mesh path: prefer inertia -> simplified -> manifold
                input_mesh = (
                    info.get("inertia_abs")
                    or info.get("simplified_abs")
                    or info.get("manifold_abs")
                )
                if not input_mesh or not os.path.exists(input_mesh):
                    results[name] = {"error": f"no input mesh found ({input_mesh})"}
                    # ensure no superdec_npz field
                    info["superdec_npz"] = None
                    continue
                try:
                    pts, norms = self.sample_surface_o3d(
                        input_mesh, n_points=target_n_points, method="uniform"
                    )
                    pts_np = np.asarray(pts, dtype=np.float32)
                    orig_N = pts_np.shape[0]
                    if normalize:
                        pts_norm, translation, scale = normalize_points(pts_np)
                    else:
                        pts_norm = pts_np.copy()
                        translation = np.zeros(3, dtype=float)
                        scale = 1.0
                    valid_names.append(name)
                    pts_norm_list.append(np.asarray(pts_norm, dtype=np.float32))
                    translations.append(np.asarray(translation, dtype=float))
                    scales.append(np.asarray(scale, dtype=float))
                    orig_N_list.append(orig_N)
                    orig_pts_list.append(pts_np)
                    info_list.append(info)
                except Exception as e:
                    results[name] = {"error": f"sampling/normalize failed: {str(e)}"}
                    info["superdec_npz"] = None
                    continue

            B = len(valid_names)
            if B == 0:
                # nothing valid in this chunk, continue
                continue

            # 2) make batch [B, N, 3]
            N = int(target_n_points)
            # pad if some pts_norm have fewer points (unlikely because sample_points_uniformly returns N, but safe)
            batch_np = np.stack(
                [
                    p if p.shape[0] == N else np.pad(p, ((0, N - p.shape[0]), (0, 0)))
                    for p in pts_norm_list
                ],
                axis=0,
            )  # (B,N,3)
            points_batch_t = torch.from_numpy(batch_np).to(self._sd_device).float()

            # 3) forward
            with torch.no_grad():
                outdict = self._sd_model(points_batch_t)

            # 4) move outdict tensors to CPU
            for k in list(outdict.keys()):
                if isinstance(outdict[k], torch.Tensor):
                    outdict[k] = outdict[k].cpu()

            # 5) denormalize outdict & points
            translation_arr = np.asarray(translations)  # shape (B,3) or (B,)
            scale_arr = np.asarray(scales)
            outdict_den = denormalize_outdict(
                outdict, translation_arr, scale_arr, z_up=False
            )
            points_den = denormalize_points(
                points_batch_t.cpu(), translation_arr, scale_arr, z_up=False
            )  # torch tensor on CPU

            # convert denorm points to numpy array (B,N,3)
            pts_den_np = to_numpy(points_den)

            # 6) extract arrays from outdict_den and split per object
            assign_matrix_all = to_numpy(outdict_den.get("assign_matrix"))  # (B,N,P)
            scale_all = to_numpy(outdict_den.get("scale"))  # (B,P,3)
            rotate_all = to_numpy(outdict_den.get("rotate"))  # (B,P,3,3)
            trans_all = to_numpy(outdict_den.get("trans"))  # (B,P,3)
            shape_all = to_numpy(outdict_den.get("shape"))  # (B,P,2)
            exist_all = to_numpy(outdict_den.get("exist"))  # (B,P)

            P = int(exist_all.shape[1]) if exist_all is not None else 0
            # exit(0)

            # 7) per-object postprocess + EMS fitting per existing primitive
            for idx_in_chunk, name in enumerate(valid_names):
                info = info_list[idx_in_chunk]
                # print(f"idx_in_chunk={idx_in_chunk} name={name}")
                # print(f"info:{info}")

                try:
                    pcd = pts_den_np[idx_in_chunk]  # (N,3)
                    assign_matrix = (
                        assign_matrix_all[idx_in_chunk]
                        if assign_matrix_all is not None
                        else None
                    )  # (N,P)
                    exist = (
                        exist_all[idx_in_chunk]
                        if exist_all is not None
                        else np.zeros(P)
                    )
                    exist_mask = exist > 0.75
                    exist_indices = np.nonzero(exist_mask)[0].tolist()

                    # print(f"exist_indices={exist_indices}")
                    # print(f"exist_mask={exist_mask}")
                    # print(f"exist={exist}")

                    # prepare primitives dict
                    primitives = {}

                    if assign_matrix is None or len(exist_indices) == 0:
                        # nothing to do: save empty primitives
                        pass
                    else:
                        # segmentation labels by argmax on full assign matrix
                        # assign_matrix shape: (N, P)
                        seg_labels_full = np.argmax(assign_matrix, axis=1).astype(
                            np.int32
                        )  # 0..P-1
                        # iterate through existing primitives
                        local_idx = 0

                        for pid in exist_indices:
                            # segmented points for primitive pid (based on argmax)
                            mask = seg_labels_full == pid
                            seg_pts = pcd[mask]
                            if seg_pts is not None and seg_pts.shape[0] >= 300:
                                key = f"s{local_idx+1}"  # s1, s2, ...
                                net_params = {
                                    "shape": (
                                        shape_all[idx_in_chunk][pid]
                                        if shape_all is not None
                                        else None
                                    ),
                                    "scale": (
                                        scale_all[idx_in_chunk][pid]
                                        if scale_all is not None
                                        else None
                                    ),
                                    "rotation": (
                                        rotate_all[idx_in_chunk][pid]
                                        if rotate_all is not None
                                        else None
                                    ),
                                    "translation": (
                                        trans_all[idx_in_chunk][pid]
                                        if trans_all is not None
                                        else None
                                    ),
                                }

                                # print(f"key: {key}")
                                # print(f"net_params: {net_params}")

                                # run EMS if available and seg points sufficient
                                # EMS_recovery is expected to return (sq_recovered, p)
                                sq_recovered, _ = EMS_recovery(seg_pts)
                                # ensure shape (11,)
                                sq_recovered = np.asarray(sq_recovered).reshape(-1)
                                ems_params = {
                                    "shape": sq_recovered[:2],
                                    "scale": sq_recovered[2:5],
                                    "rotation": R.from_euler(
                                        "ZYX", sq_recovered[5:8]
                                    ).as_matrix(),
                                    "translation": sq_recovered[8:],
                                }
                                # print(f"ems_params: {ems_params}")

                                # store primitive dict
                                primitives[key] = {
                                    "points": seg_pts.astype(np.float32),
                                    "ems_params": ems_params,  # 11-d or None
                                    "net_params": net_params,  # raw net outputs (may be None entries)
                                }
                                local_idx += 1
                            else:
                                pass

                    # save npz into same folder as inertia_abs (or folder_abs)
                    save_dir = None
                    if info.get("inertia_abs") and os.path.exists(
                        info.get("inertia_abs")
                    ):
                        save_dir = os.path.dirname(info.get("inertia_abs"))
                    else:
                        save_dir = info.get("folder_abs", None) or self.root
                    os.makedirs(save_dir, exist_ok=True)
                    npz_path = os.path.join(save_dir, "superdec.npz")

                    # Save the primitives dict into a single npz entry (pickle via numpy)
                    # This will store the whole dict as a pickled object under key 'primitives'
                    np.savez_compressed(npz_path, primitives=primitives)

                    # update index to only store path (no heavy data)
                    info["superdec_npz"] = os.path.abspath(npz_path)
                    results[name] = {"npz": os.path.abspath(npz_path)}
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    results[name] = {"error": str(e)}
                    info["superdec_npz"] = None
                    try:
                        if "npz_path" in locals() and os.path.exists(npz_path):
                            os.remove(npz_path)
                    except Exception:
                        pass

        return results

    def load_superdec_npz(self, object_name: str) -> Dict[str, Any]:
        if object_name not in self.index:
            raise KeyError(f"Object '{object_name}' not found in dataset index.")
        info = self.index[object_name]
        # determine npz path: prefer explicit field, else inertia folder /superdec.npz
        npz_path = info.get("superdec_npz")
        print(f"npz_path={npz_path}")
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Failed to load npz {npz_path}: {e}")

        primitives = data["primitives"].item()
        points_dict = {}
        ems_params_dict = {}
        net_params_dict = {}

        for key, entry in primitives.items():

            pts = entry["points"]
            ems = entry["ems_params"]
            netp = entry["net_params"]

            points_dict[key] = pts
            ems_params_dict[key] = ems
            net_params_dict[key] = netp

        # print(f"points_dict={points_dict}")
        # print(f"ems_params_dict={ems_params_dict}")
        # print(f"net_params_dict={net_params_dict}")
        return points_dict, ems_params_dict, net_params_dict

    # -------------------------
    # Visualization
    # -------------------------
    def visualize_object(self, object_name: str):
        """
        Visualize a single object showing:
        This tries to use viser first; if not available, falls back to open3d (TODO).
        Default up-axis is Z.
        """
        if object_name not in self.index:
            raise KeyError(f"Object not found: {object_name}")
        info = self.index[object_name]

        # prepare base mesh
        mesh_path = (
            info.get("inertia_abs")
            or info.get("simplified_abs")
            or info.get("manifold_abs")
        )
        if mesh_path is None or not os.path.exists(mesh_path):
            raise FileNotFoundError(f"No mesh available to visualize for {object_name}")

        base_mesh = self.load_mesh(mesh_path)

        import viser  # type: ignore

        # Use viser if available
        if viser is not None:
            try:
                server = viser.ViserServer()
                # add base mesh
                server.scene.add_mesh_trimesh("base_mesh", mesh=base_mesh, visible=True)

                # set up axis: default z-up
                server.scene.set_up_direction([0.0, 0.0, 1.0])

                @server.on_client_connect
                def _(client: viser.ClientHandle) -> None:
                    client.camera.position = (0.8, 0.8, 0.8)
                    client.camera.look_at = (0.0, 0.0, 0.0)

                # block keep alive
                try:
                    while True:
                        time.sleep(10.0)
                except KeyboardInterrupt:
                    return
            except Exception as e:
                # fallthrough to open3d
                raise e


# -------------------------
# Example usage (for quick manual test)
# -------------------------
if __name__ == "__main__":
    # Quick demo: scan dataset, print index summary
    ds = DatasetObjects("assets/ycb_datasets")
    idx = ds.get_index()
    print("Found objects:", list(idx.keys()))

    # ds.visualize_object("002_master_chef_can")

    # Example to compute superdec for one object (uncomment & set your checkpoints)
    ds._init_superdec("checkpoints/normalized", "ckpt.pt")
    # res = ds.compute_superdec_single("002_master_chef_can")
    # print("SuperDec result keys:", res.keys())
    # print("SuperDec result:", res)

    res = ds.compute_superdec_datasets()
    # print("SuperDec result keys:", res.keys())
    # print("SuperDec result:", res)
    superdec_data = ds.load_superdec_npz("035_power_drill")
    # print("SuperDec data:", superdec_data)
