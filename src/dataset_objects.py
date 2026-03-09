"""DatasetObjects: manifest-driven object-scale index.

Design notes
- No dataset-root resolver. `dataset_root` must be explicitly passed from config.
- Build object list from `manifest.process_meshes.json` only (`process_status=success`).
- Eager prebuild scaled assets at construction.
- Output info granularity is object-scale (one info per scale).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from tqdm import tqdm

from src.scale_dataset_builder import DEFAULT_FIXED_SCALES, ScaleDatasetBuilder
from utils.utils_pointcloud import preview_pointcloud_with_normals, sample_surface_o3d

DEFAULT_DATASETS = ["YCB"]


class DatasetObjects:
    """Manifest-driven merged dataset index.

    Args:
        dataset_root: Required dataset root path, e.g. ``assets/objects/processed``.
        dataset_names: Dataset include list. Defaults to ``DEFAULT_DATASETS``.
        scales: Fixed scales to prebuild.
        dataset_tag: Output tag, usually config stem.
        dataset_output_root: Root folder for generated scaled assets.
        verbose: Print all skip/build messages when True.

    Info format (one row per object-scale):
        {
            "global_id": int,
            "object_name": str,
            "coacd_abs": str,
            "convex_parts_abs": List[str],
            "scale": float,
            "mjcf_abs": str,
            "output_dir_abs": str,
            "object_scale_key": str,
        }
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        dataset_tag: str = "run_YCB_liberhand",
        dataset_output_root: str = "datasets",
        verbose: bool = False,
    ):
        if not dataset_root:
            raise ValueError("dataset_root must be provided by config and cannot be empty.")

        self.root = Path(dataset_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset_root does not exist: {self.root}")

        self.dataset_names = list(DEFAULT_DATASETS if dataset_names is None else dataset_names)
        if not self.dataset_names:
            raise ValueError("dataset_names cannot be empty.")

        self.scales = [float(s) for s in (DEFAULT_FIXED_SCALES if scales is None else scales)]
        if not self.scales:
            raise ValueError("scales cannot be empty.")

        self.dataset_tag = str(dataset_tag)
        self.dataset_output_root = str(dataset_output_root)
        self.verbose = bool(verbose)

        self._builder = ScaleDatasetBuilder(self.dataset_output_root)

        self.items: List[Dict] = []
        self._key_to_index: Dict[str, int] = {}

        self._build_from_manifests()

        if not self.items:
            raise RuntimeError(
                f"No valid object-scale entries under root={self.root} include={self.dataset_names}."
            )

    @property
    def id2name(self) -> Dict[int, str]:
        return {i: f"{it['object_name']}__{self._scale_to_key(it['scale'])}" for i, it in enumerate(self.items)}

    def get_entries(self) -> List[Dict]:
        return self.items

    def get_info(self, name_or_id: Union[str, int]) -> Dict:
        if isinstance(name_or_id, int):
            idx = int(name_or_id)
            if idx < 0 or idx >= len(self.items):
                raise KeyError(f"Object id '{idx}' out of range. Total={len(self.items)}")
            return self.items[idx]

        key = str(name_or_id)
        if key in self._key_to_index:
            return self.items[self._key_to_index[key]]

        # convenience: allow object name without scale when only one match
        matches = [it for it in self.items if it["object_name"] == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"Object '{key}' has multiple scales. Use obj-id or key 'object__scaleXXX'."
            )
        raise KeyError(f"Object '{key}' not found")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _scale_to_key(scale: float) -> str:
        return f"scale{int(round(float(scale) * 1000)):03d}"

    @staticmethod
    def _to_abs_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (Path.cwd() / p).resolve()

    def _build_from_manifests(self) -> None:
        gid = 0

        for dataset_name in self.dataset_names:
            dataset_dir = self.root / dataset_name
            if not dataset_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            manifest_path = dataset_dir / "manifest.process_meshes.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            objects = manifest.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError(f"Invalid manifest objects list: {manifest_path}")

            default_mass = float(manifest.get("summary", {}).get("default_mass_kg", 0.1))

            obj_iter = objects
            if self.verbose:
                obj_iter = tqdm(objects, desc=f"dataset:{dataset_name}", leave=False)

            for obj in obj_iter:
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("process_status", "")).lower() != "success":
                    continue

                object_name = str(obj.get("object_id") or obj.get("name") or "").strip()
                if not object_name:
                    self._log(f"[DatasetObjects] skip unnamed object in {dataset_name}")
                    continue

                mesh_path_raw = obj.get("mesh_path")
                if not isinstance(mesh_path_raw, str) or not mesh_path_raw:
                    self._log(f"[DatasetObjects] skip {object_name}: missing mesh_path")
                    continue

                object_dir = self._to_abs_path(mesh_path_raw).parent
                coacd_abs = str((object_dir / "coacd.obj").resolve())

                # build input to scale builder (path existence is checked during actual loading/export)
                base_info = {
                    "object_name": object_name,
                    "coacd_abs": coacd_abs,
                }

                mass_kg = float(obj.get("mass_kg", default_mass))
                pm = obj.get("principal_moments")
                if not isinstance(pm, list) or len(pm) != 3:
                    pm = [1e-6, 1e-6, 1e-6]
                principal_moments = [float(pm[0]), float(pm[1]), float(pm[2])]

                scale_assets = self._builder.build_multi_scale_assets(
                    config_stem=self.dataset_tag,
                    object_info=base_info,
                    scales=self.scales,
                    mass_kg=mass_kg,
                    principal_moments=principal_moments,
                    overwrite=False,
                )

                for scale_key, rec in sorted(scale_assets.items()):
                    output_dir_abs = str(Path(rec["xml_abs"]).resolve().parent)
                    info = {
                        "global_id": gid,
                        "object_name": object_name,
                        "coacd_abs": rec["coacd_abs"],
                        "convex_parts_abs": list(rec["convex_parts_abs"]),
                        "scale": float(rec["scale"]),
                        "mjcf_abs": rec["xml_abs"],
                        "output_dir_abs": output_dir_abs,
                        "object_scale_key": f"{object_name}__{scale_key}",
                    }
                    self.items.append(info)
                    self._key_to_index[f"{object_name}__{scale_key}"] = gid
                    gid += 1

    def load_mesh(self, mesh_or_path: Union[str, trimesh.Trimesh]) -> trimesh.Trimesh:
        if isinstance(mesh_or_path, trimesh.Trimesh):
            return mesh_or_path.copy()

        mesh_path = str(mesh_or_path)
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        return mesh.copy()

    def get_mesh(
        self,
        name_or_id: Union[str, int],
        mesh_type: str = "coacd",
        alpha: float = 1.0,
    ) -> trimesh.Trimesh:
        info = self.get_info(name_or_id)

        if mesh_type == "coacd":
            mesh = self.load_mesh(info["coacd_abs"])
        elif mesh_type == "convex_parts":
            mesh = trimesh.util.concatenate([self.load_mesh(p) for p in info["convex_parts_abs"]])
        else:
            raise ValueError("mesh_type must be one of ['coacd','convex_parts']")

        if 0.0 <= alpha <= 1.0:
            a = int(round(alpha * 255))
            n = len(mesh.vertices)
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128, a], dtype=np.uint8), (n, 1))
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
        pts, norms = sample_surface_o3d(obj_path=obj_path, n_points=n_points, method=method)

        if preview:
            self._preview_pointcloud_with_normals(pts, norms)
        return pts, norms

    def _preview_pointcloud_with_normals(self, pts: np.ndarray, norms: np.ndarray) -> None:
        preview_pointcloud_with_normals(pts, norms)

    def get_point_cloud(self, name_or_id: Union[str, int], n_points: int = 4096, method: str = "poisson"):
        info = self.get_info(name_or_id)
        mesh_path = info.get("coacd_abs")
        return self.sample_surface_o3d(mesh_path, n_points, method)
