"""Unified object index with eager multi-scale asset prebuild."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
try:
    import open3d as o3d
except Exception:
    o3d = None

from src.scale_dataset_builder import DEFAULT_FIXED_SCALES, ScaleDatasetBuilder

DEFAULT_DATASETS = ["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]


def resolve_dataset_root(preferred: Optional[str] = None) -> str:
    candidates: List[Path] = []
    if preferred:
        candidates.append(Path(preferred))

    env_root = os.getenv("DEXGRASP_OBJECTS_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(Path("assets/objects/processed"))

    for cand_dir in candidates:
        if cand_dir.exists() and cand_dir.is_dir():
            return str(cand_dir)

    raise FileNotFoundError(
        "No valid dataset root found. Use assets/objects/processed or set DEXGRASP_OBJECTS_ROOT."
    )


class DatasetObjects:
    """Merged object index over selected datasets + prebuilt multi-scale assets."""

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        dataset_tag: str = "run_YCB_liberhand",
        dataset_output_root: str = "datasets",
        prebuild_scales: bool = True,
        object_mass_kg: float = 0.1,
    ):
        self.root = Path(resolve_dataset_root(dataset_root)).resolve()
        self.dataset_names = list(DEFAULT_DATASETS if dataset_names is None else dataset_names)
        self.dataset_names_norm = {name.lower() for name in self.dataset_names}

        self.scales = [float(s) for s in (DEFAULT_FIXED_SCALES if scales is None else scales)]
        self.dataset_tag = str(dataset_tag)
        self.dataset_output_root = str(dataset_output_root)
        self.prebuild_scales = bool(prebuild_scales)

        self._manifest_success_names = self._load_manifest_success_names()
        self.index: "OrderedDict[str, Dict]" = OrderedDict()
        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self._skip_count = 0
        self._skip_log_limit = 40
        self._scale_builder = ScaleDatasetBuilder(self.dataset_output_root, object_mass_kg=object_mass_kg)

        self._build_index()
        if self.prebuild_scales:
            self._prebuild_scale_assets_for_all()

    @property
    def id2name(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.index.keys())}

    def get_index(self) -> "OrderedDict[str, Dict]":
        return self.index

    def get_info(self, name_or_id: Union[str, int]) -> Dict:
        if isinstance(name_or_id, int):
            name = self.id2name.get(int(name_or_id))
            if name is None:
                raise KeyError(f"Object id '{name_or_id}' out of range. Total={len(self.index)}")
            return self.index[name]

        name = str(name_or_id)
        if name not in self.index:
            raise KeyError(f"Object '{name}' not found under {self.root}")
        return self.index[name]

    def _build_index(self) -> None:
        object_dirs = self._discover_object_dirs(self.root)
        object_dirs.sort(key=lambda p: (p.parent.name.lower(), p.name.lower()))

        for object_dir in object_dirs:
            if not self._is_manifest_success(object_dir):
                continue
            info = self._build_object_info(object_dir)
            if info is None:
                continue
            self.index[info["object_name"]] = info

        if not self.index:
            raise RuntimeError(
                f"No valid object folders under {self.root} with dataset_names={self.dataset_names}."
            )

        for gid, key in enumerate(self.index.keys()):
            self.index[key]["global_id"] = gid

        if self._skip_count > self._skip_log_limit:
            hidden = self._skip_count - self._skip_log_limit
            print(f"[DatasetObjects] ... {hidden} more skipped objects (log capped).")

    def _prebuild_scale_assets_for_all(self) -> None:
        for object_name in self.index.keys():
            info = self.index[object_name]
            recs = self._scale_builder.build_multi_scale_assets(
                self.dataset_tag,
                info,
                self.scales,
                overwrite=False,
                show_progress=True,
            )
            scale_assets: Dict[str, Dict] = {}
            for scale_key, rec in recs.items():
                scale_assets[scale_key] = {
                    "scale": float(rec["scale"]),
                    "coacd_abs": rec["coacd_abs"],
                    "xml_abs": rec["xml_abs"],
                    "convex_parts_abs": list(rec["convex_parts_abs"]),
                }
            info["scale_assets"] = scale_assets

    def _warn_skip(self, dataset_name: str, object_name: str, reason: str) -> None:
        self._skip_count += 1
        if self._skip_count <= self._skip_log_limit:
            print(f"[DatasetObjects] Skip {dataset_name}/{object_name}: {reason}")

    def _discover_object_dirs(self, root_dir: Path) -> List[Path]:
        dataset_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
        object_dirs: List[Path] = []

        for dataset_dir in dataset_dirs:
            if self.dataset_names_norm and dataset_dir.name.lower() not in self.dataset_names_norm:
                continue
            for object_dir in dataset_dir.iterdir():
                if object_dir.is_dir():
                    object_dirs.append(object_dir)
        return object_dirs

    def _load_manifest_success_names(self) -> Dict[str, set]:
        success_map: Dict[str, set] = {}

        for dataset_dir in self.root.iterdir():
            if not dataset_dir.is_dir():
                continue
            if self.dataset_names_norm and dataset_dir.name.lower() not in self.dataset_names_norm:
                continue

            manifest_path = dataset_dir / "manifest.process_meshes.json"
            if not manifest_path.exists():
                continue

            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            objects = data.get("objects")
            if not isinstance(objects, list):
                continue

            success_names = set()
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("process_status", "")).lower() != "success":
                    continue
                object_id = obj.get("object_id")
                name = obj.get("name")
                if isinstance(object_id, str) and object_id:
                    success_names.add(object_id)
                if isinstance(name, str) and name:
                    success_names.add(name)

            success_map[dataset_dir.name.lower()] = success_names

        return success_map

    def _is_manifest_success(self, object_dir: Path) -> bool:
        dataset_name = object_dir.parent.name.lower()
        success_names = self._manifest_success_names.get(dataset_name)
        if success_names is None:
            return True
        return object_dir.name in success_names

    def _build_object_info(self, object_dir: Path) -> Optional[Dict]:
        object_name = object_dir.name
        dataset_name = object_dir.parent.name

        coacd_abs = str((object_dir / "coacd.obj").resolve())
        if not os.path.exists(coacd_abs):
            self._warn_skip(dataset_name, object_name, "missing coacd.obj")
            return None

        visual_path = object_dir / "visual.obj"
        visual_abs = str(visual_path.resolve()) if visual_path.exists() else None

        convex_parts_dir = object_dir / "convex_parts"
        if not convex_parts_dir.is_dir():
            legacy_meshes_dir = object_dir / "meshes"
            convex_parts_dir = legacy_meshes_dir if legacy_meshes_dir.is_dir() else convex_parts_dir

        convex_parts_abs: List[str] = []
        if convex_parts_dir.is_dir():
            convex_parts_abs = [str(p.resolve()) for p in sorted(convex_parts_dir.glob("*.obj"))]

        if not convex_parts_abs:
            self._warn_skip(dataset_name, object_name, "missing convex_parts/*.obj")
            return None

        return {
            "global_id": -1,
            "object_name": object_name,
            "coacd_abs": coacd_abs,
            "convex_parts_abs": convex_parts_abs,
            "visual_abs": visual_abs,
            "scale_assets": {},
        }

    def load_mesh(self, mesh_or_path: Union[str, trimesh.Trimesh]) -> trimesh.Trimesh:
        if isinstance(mesh_or_path, trimesh.Trimesh):
            return mesh_or_path.copy()

        mesh_path = str(mesh_or_path)
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path].copy()

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        self._mesh_cache[mesh_path] = mesh
        return mesh.copy()

    def get_mesh(
        self,
        name_or_id: Union[str, int],
        mesh_type: str = "coacd",
        alpha: float = 1.0,
        scale_key: Optional[str] = None,
    ) -> trimesh.Trimesh:
        info = self.get_info(name_or_id)

        if scale_key:
            rec = info.get("scale_assets", {}).get(scale_key)
            if rec is None:
                raise KeyError(f"scale_key '{scale_key}' not found for object {info['object_name']}")
            if mesh_type == "coacd":
                mesh = self.load_mesh(rec["coacd_abs"])
            elif mesh_type == "convex_parts":
                mesh = trimesh.util.concatenate([self.load_mesh(p) for p in rec["convex_parts_abs"]])
            else:
                raise ValueError(f"Unknown mesh_type={mesh_type}")
        else:
            if mesh_type == "convex_parts":
                mesh = trimesh.util.concatenate([self.load_mesh(p) for p in info["convex_parts_abs"]])
            elif mesh_type == "coacd":
                mesh = self.load_mesh(info["coacd_abs"])
            elif mesh_type == "visual":
                if not info.get("visual_abs"):
                    raise FileNotFoundError(f"No visual_abs for object {info['object_name']}")
                mesh = self.load_mesh(info["visual_abs"])
            else:
                raise ValueError(f"Unknown mesh_type={mesh_type}, expected one of [coacd, convex_parts, visual].")

        if 0.0 <= alpha <= 1.0:
            a = int(round(alpha * 255))
            n = len(mesh.vertices)
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128, a], dtype=np.uint8), (n, 1))
        return mesh

    def sample_surface_mesh(
        self,
        mesh_or_path: Union[str, trimesh.Trimesh],
        n_points: int = 4096,
        method: str = "even",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Compatibility wrapper: prefer Open3D path sampling to keep deterministic
        # behavior with previous pipeline (uniform/poisson).
        if isinstance(mesh_or_path, str):
            if method in ("even", "poisson"):
                o3d_method = "poisson"
            elif method in ("random", "uniform"):
                o3d_method = "uniform"
            else:
                raise ValueError("Unknown sampling method: choose 'even/random/uniform/poisson'")
            return self.sample_surface_o3d(mesh_or_path, n_points=n_points, method=o3d_method)

        mesh = self.load_mesh(mesh_or_path)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        normals = mesh.face_normals[np.asarray(face_idx, dtype=np.int64)]
        return pts.astype(np.float32), normals.astype(np.float32)

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

        _ = ts
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_normals():
            norms = np.asarray(pcd.normals, dtype=np.float32)
        else:
            norms = np.zeros_like(pts)

        if preview:
            self._preview_pointcloud_with_normals(pts, norms)
        return pts, norms

    def _preview_pointcloud_with_normals(self, pts: np.ndarray, norms: np.ndarray) -> None:
        if o3d is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(norms)
        o3d.visualization.draw_geometries([pcd])

    def get_point_cloud(
        self,
        name_or_id: Union[str, int],
        n_points: int = 4096,
        method: str = "poisson",
        scale_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        info = self.get_info(name_or_id)
        if scale_key:
            rec = info.get("scale_assets", {}).get(scale_key)
            if rec is None:
                raise KeyError(f"scale_key '{scale_key}' not found for object {info['object_name']}")
            return self.sample_surface_o3d(rec["coacd_abs"], n_points=n_points, method=method)

        mesh_path = info.get("inertia_abs") or info.get("coacd_abs")
        return self.sample_surface_o3d(mesh_path, n_points=n_points, method=method)
