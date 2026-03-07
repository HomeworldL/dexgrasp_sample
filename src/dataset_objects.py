"""Unified object-dataset loader for dexterous grasp sampling.

Lightweight index only stores paths and small scalar attributes.
Heavy assets (mesh/point cloud) are loaded lazily on demand.
"""

from __future__ import annotations

import hashlib
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

DEFAULT_DATASETS = ["ShapeNetCore", "ShapeNetSem", "DDG", "MSO"]
SHAPENET_DATASETS = {"shapenetcore", "shapenetsem"}
DEFAULT_SHAPENET_SCALE_RANGE = (0.06, 0.15)


def resolve_dataset_root(preferred: Optional[str] = None) -> str:
    """Resolve dataset root. / 解析数据集根目录。

    EN:
    Priority is explicit path -> env var -> default symlink path.

    中文：
    优先级为：显式传入路径 -> 环境变量 -> 默认软链接路径。
    """

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
    """Merged object index with global integer ids. / 多数据集合并对象索引。"""

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
        shapenet_scale_range: Tuple[float, float] = DEFAULT_SHAPENET_SCALE_RANGE,
        shapenet_scale_seed: int = 0,
    ):
        """Build a lightweight merged index. / 构建轻量对象索引。"""

        self.root = Path(resolve_dataset_root(dataset_root)).resolve()
        self.dataset_names = list(DEFAULT_DATASETS if dataset_names is None else dataset_names)
        self.dataset_names_norm = {name.lower() for name in self.dataset_names}
        self.shapenet_scale_range = shapenet_scale_range
        self.shapenet_scale_seed = int(shapenet_scale_seed)

        self._manifest_success_names = self._load_manifest_success_names()
        self.index: "OrderedDict[str, Dict]" = OrderedDict()
        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self._skip_count = 0
        self._skip_log_limit = 40
        self._build_index()

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
        """Scan selected datasets and assign global ids. / 扫描并分配全局id。"""

        object_dirs = self._discover_object_dirs(self.root)
        object_dirs.sort(key=lambda p: (p.parent.name.lower(), p.name.lower()))

        for object_dir in object_dirs:
            if not self._is_manifest_success(object_dir):
                continue

            info = self._build_object_info(object_dir)
            if info is None:
                continue

            name = info["name"]
            self.index[name] = info

        if not self.index:
            raise RuntimeError(
                f"No valid object folders under {self.root} with dataset_names={self.dataset_names}."
            )

        for gid, key in enumerate(self.index.keys()):
            self.index[key]["global_id"] = gid
        if self._skip_count > self._skip_log_limit:
            hidden = self._skip_count - self._skip_log_limit
            print(f"[DatasetObjects] ... {hidden} more skipped objects (log capped).")

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

    def _dataset_should_scale(self, dataset_name: str) -> bool:
        return dataset_name.lower() in SHAPENET_DATASETS

    def _deterministic_scale(self, object_key: str) -> float:
        lo, hi = self.shapenet_scale_range
        if not (lo > 0 and hi >= lo):
            raise ValueError("Invalid shapenet_scale_range. Expect 0 < lo <= hi.")
        key = f"{self.shapenet_scale_seed}:{object_key}"
        digest = hashlib.sha1(key.encode("utf-8")).digest()
        ratio = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return float(lo + (hi - lo) * ratio)

    def _build_object_info(self, object_dir: Path) -> Optional[Dict]:
        """Build flat info dict with strict required files.

        EN:
        Required files are fixed by naming convention. If any required file is
        missing, skip this object and print a warning.

        中文：
        采用固定命名规则。只要任一关键文件缺失，就跳过该物体并给出提示。
        """

        object_name = object_dir.name
        dataset_name = object_dir.parent.name

        inertia_abs = str((object_dir / "inertia.obj").resolve())
        visual_abs = str((object_dir / "visual.obj").resolve())
        manifold_abs = str((object_dir / "manifold.obj").resolve())
        coacd_abs = str((object_dir / "coacd.obj").resolve())
        xml_abs = str((object_dir / f"{object_name}.xml").resolve())

        required_paths = {
            "inertia.obj": inertia_abs,
            "visual.obj": visual_abs,
            "manifold.obj": manifold_abs,
            "coacd.obj": coacd_abs,
            f"{object_name}.xml": xml_abs,
        }
        missing = [k for k, v in required_paths.items() if not os.path.exists(v)]
        if missing:
            self._warn_skip(dataset_name, object_name, f"missing {missing}")
            return None

        convex_parts_dir = object_dir / "convex_parts"
        if not convex_parts_dir.is_dir():
            # legacy compatibility: source folder may still be named meshes
            legacy_meshes_dir = object_dir / "meshes"
            convex_parts_dir = legacy_meshes_dir if legacy_meshes_dir.is_dir() else convex_parts_dir

        convex_parts_abs: List[str] = []
        if convex_parts_dir.is_dir():
            convex_parts_abs = [str(p.resolve()) for p in sorted(convex_parts_dir.glob("*.obj"))]
        if not convex_parts_abs:
            self._warn_skip(dataset_name, object_name, "missing convex parts objs")
            return None

        urdf_path = object_dir / f"{object_name}.urdf"
        urdf_abs = str(urdf_path.resolve()) if urdf_path.exists() else None

        object_scale = (
            self._deterministic_scale(object_name)
            if self._dataset_should_scale(dataset_name)
            else 1.0
        )

        info = {
            "global_id": -1,
            "name": object_name,
            "object_id": object_name,
            "dataset": dataset_name,
            "scale": object_scale,
            "folder_abs": str(object_dir.resolve()),
            "inertia_abs": inertia_abs,
            "visual_abs": visual_abs,
            "simplified_abs": visual_abs,
            "manifold_abs": manifold_abs,
            "coacd_abs": coacd_abs,
            "xml_abs": xml_abs,
            "urdf_abs": urdf_abs,
            "convex_parts_abs": convex_parts_abs,
        }
        return info

    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path].copy()

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        self._mesh_cache[mesh_path] = mesh
        return mesh.copy()

    def get_mesh(self, name_or_id: Union[str, int], mesh_type: str = "inertia", alpha: float = 1.0) -> trimesh.Trimesh:
        """Load mesh lazily by flat keys. / 按扁平字段懒加载网格。"""

        info = self.get_info(name_or_id)

        if mesh_type == "convex_parts":
            meshes = [self.load_mesh(p) for p in info["convex_parts_abs"]]
            mesh = trimesh.util.concatenate(meshes)
        else:
            key = f"{mesh_type}_abs"
            mesh_path = info.get(key)
            if not mesh_path:
                raise FileNotFoundError(f"No mesh for type={mesh_type} object={info['name']}")
            mesh = self.load_mesh(mesh_path)

        if 0.0 <= alpha <= 1.0:
            a = int(round(alpha * 255))
            n = len(mesh.vertices)
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128, a], dtype=np.uint8), (n, 1)
            )
        return mesh

    # -------------------------
    # Surface sampling (Open3D)
    # -------------------------
    def sample_surface_o3d(
        self,
        obj_path: str,
        n_points: int = 4096,
        method: str = "poisson",
        preview: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample surface points from an OBJ using Open3D.

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
                pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Point cloud sampling always uses visual.obj. / 点云统一从 visual.obj 采样。"""

        info = self.get_info(name_or_id)
        mesh_path = info.get("visual_abs")
        if not mesh_path:
            raise FileNotFoundError(f"No visual_abs for object '{info['name']}'")
        return self.sample_surface_o3d(mesh_path, n_points=n_points, method=method)
