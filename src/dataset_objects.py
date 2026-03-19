"""DatasetObjects: manifest-driven object-scale index.

Design notes
- No dataset-root resolver. `dataset_root` must be explicitly passed from config.
- Build object list from `manifest.process_meshes.json` only (`process_status=success`).
- Eager prebuild scaled assets at construction.
- Output info granularity is object-scale (one info per scale).
"""

from __future__ import annotations

import json
import time
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
            "object_scale_key": str,
            "object_name": str,
            "output_dir_abs": str,
            "coacd_abs": str,
            "convex_parts_abs": List[str],
            "scale": float,
            "mjcf_abs": str,  
        }
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        dataset_tag: str = "graspdata_YCB_liberhand",
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

    def get_entries(self) -> List[Dict]:
        return self.items
    
    def get_obj_info_by_index(self, obj_id: int) -> Dict:
        if obj_id < 0 or obj_id >= len(self.items):
            raise KeyError(f"Object id '{obj_id}' out of range. Total={len(self.items)}")
        return self.items[obj_id]
    
    def get_obj_info_by_scale_key(self, obj_scale_key: str) -> Dict:
        if obj_scale_key in self._key_to_index:
            return self.items[self._key_to_index[obj_scale_key]]
        raise KeyError(f"Object scale key '{obj_scale_key}' not found.")

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
            dataset_start = time.perf_counter()
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
            expected_entries = len(objects) * len(self.scales)
            print(
                f"[DatasetObjects] indexing dataset={dataset_name} "
                f"objects={len(objects)} scales={len(self.scales)} "
                f"expected_entries<={expected_entries}"
            )

            obj_iter = objects
            if self.verbose:
                obj_iter = tqdm(objects, desc=f"dataset:{dataset_name}", leave=False)

            built_before = gid
            for obj_idx, obj in enumerate(obj_iter, start=1):
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

                scale_assets = {}
                for scale in self.scales:
                    existing_rec = self._existing_scale_assets(object_name=object_name, scale=float(scale))
                    if existing_rec is not None:
                        scale_assets[self._builder.scale_tag(float(scale))] = existing_rec
                        continue
                    try:
                        rec = self._builder.build_scale_assets(
                            config_stem=self.dataset_tag,
                            object_info=base_info,
                            scale=float(scale),
                            mass_kg=mass_kg,
                            principal_moments=principal_moments,
                            overwrite=False,
                        )
                    except Exception as exc:
                        self._log(
                            f"[DatasetObjects] skip {object_name} scale={float(scale):.6f}: "
                            f"{exc}"
                        )
                        continue
                    scale_assets[self._builder.scale_tag(float(scale))] = rec

                for scale_key, rec in sorted(scale_assets.items()):
                    output_dir_abs = str(Path(rec["xml_abs"]).resolve().parent)
                    info = {
                        "global_id": gid,
                        "object_scale_key": f"{object_name}__{scale_key}",
                        "object_name": object_name,
                        "output_dir_abs": output_dir_abs,
                        "coacd_abs": rec["coacd_abs"],
                        "convex_parts_abs": list(rec["convex_parts_abs"]),
                        "mjcf_abs": rec["xml_abs"],
                        "scale": float(rec["scale"]),             
                    }
                    self.items.append(info)
                    self._key_to_index[f"{object_name}__{scale_key}"] = gid
                    gid += 1

                if (
                    not self.verbose
                    and obj_idx % 200 == 0
                    and len(objects) >= 500
                ):
                    elapsed = time.perf_counter() - dataset_start
                    print(
                        f"[DatasetObjects] dataset={dataset_name} progress={obj_idx}/{len(objects)} "
                        f"indexed={gid - built_before} elapsed={elapsed:.1f}s"
                    )

            elapsed = time.perf_counter() - dataset_start
            print(
                f"[DatasetObjects] done dataset={dataset_name} "
                f"indexed={gid - built_before} elapsed={elapsed:.1f}s"
            )

    def _existing_scale_assets(self, object_name: str, scale: float) -> Optional[Dict]:
        scale_tag = self._builder.scale_tag(float(scale))
        scale_dir = self._builder.base_output_dir / self.dataset_tag / object_name / scale_tag
        convex_dir = scale_dir / "convex_parts"
        coacd_path = scale_dir / "coacd.obj"
        xml_path = scale_dir / "object.xml"

        if (not xml_path.exists()) or (not coacd_path.exists()) or (not convex_dir.is_dir()):
            return None

        convex_parts_abs = [str(p.resolve()) for p in sorted(convex_dir.glob("*.obj")) if p.is_file()]
        if not convex_parts_abs:
            return None

        return {
            "object_name": object_name,
            "scale": float(scale),
            "scale_tag": scale_tag,
            "coacd_abs": str(coacd_path.resolve()),
            "convex_parts_abs": convex_parts_abs,
            "xml_abs": str(xml_path.resolve()),
        }

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
        if isinstance(name_or_id, int):
            info = self.get_obj_info_by_index(int(name_or_id))
        else:
            info = self.get_obj_info_by_scale_key(str(name_or_id))

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
        if isinstance(name_or_id, int):
            info = self.get_obj_info_by_index(int(name_or_id))
        else:
            info = self.get_obj_info_by_scale_key(str(name_or_id))
        mesh_path = info.get("coacd_abs")
        return self.sample_surface_o3d(mesh_path, n_points, method)
