"""DatasetObjects: manifest-driven object-scale index over shared-mesh objdata."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from tqdm import tqdm

from src.scale_dataset_builder import ScaleDatasetBuilder
from utils.utils_file import native_tag as _native_tag, scale_tag as _scale_tag
from utils.utils_pointcloud import preview_pointcloud_with_normals, sample_surface_o3d


class DatasetObjects:
    """Objdata-manifest-driven object-scale index.

    Info format (one row per object-scale):
        {
            "global_id": int,
            "object_scale_key": str,
            "object_name": str,
            "asset_dir_abs": str,
            "output_dir_abs": str,
            "coacd_abs": str,
            "mjcf_abs": str,
            "urdf_abs": str,
            "scale_tag": str,
            "scale": Optional[float],
            "is_native": bool,
        }
    """

    def __init__(
        self,
        scales: List[float],
        objdata_tag: str,
        object_names: Optional[List[str]] = None,
        include_native: bool = False,
        graspdata_tag: Optional[str] = None,
        generated_dataset_root: str = "datasets",
        pc_subdir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.scales = [float(s) for s in scales]
        self.include_native = bool(include_native)
        if not self.scales and not self.include_native:
            raise ValueError("At least one scale or include_native=True is required.")
        self.object_names = (
            {str(name).strip() for name in object_names if str(name).strip()}
            if object_names is not None
            else None
        )

        self.objdata_tag = str(objdata_tag)
        self.graspdata_tag = str(graspdata_tag or objdata_tag)
        self.generated_dataset_root = str(generated_dataset_root)
        self.generated_dataset_root_path = Path(self.generated_dataset_root).resolve()
        self.pc_subdir = str(pc_subdir).strip() if pc_subdir else None
        self.verbose = bool(verbose)

        self.items: List[Dict] = []
        self._key_to_index: Dict[str, int] = {}

        self._build_from_manifests()

        if not self.items:
            raise RuntimeError(
                f"No valid object-scale entries under objdata_tag={self.objdata_tag}."
            )

    def get_entries(self) -> List[Dict]:
        return self.items

    def get_obj_info_by_index(self, obj_id: int) -> Dict:
        if obj_id < 0 or obj_id >= len(self.items):
            raise KeyError(
                f"Object id '{obj_id}' out of range. Total={len(self.items)}"
            )
        return self.items[obj_id]

    def get_obj_info_by_scale_key(self, obj_scale_key: str) -> Dict:
        if obj_scale_key in self._key_to_index:
            return self.items[self._key_to_index[obj_scale_key]]
        raise KeyError(f"Object scale key '{obj_scale_key}' not found.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def scale_tag(scale: float) -> str:
        return _scale_tag(scale)

    @staticmethod
    def native_tag() -> str:
        return _native_tag()

    def _build_from_manifests(self) -> None:
        gid = 0
        dataset_start = time.perf_counter()
        manifest_path = (
            self.generated_dataset_root_path
            / self.objdata_tag
            / "manifest.process_meshes.json"
        ).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Objdata manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        objects = manifest.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError(f"Invalid manifest objects list: {manifest_path}")

        expected_entries = len(objects) * (
            len(self.scales) + (1 if self.include_native else 0)
        )
        print(
            f"[DatasetObjects] indexing objdata_tag={self.objdata_tag} "
            f"objects={len(objects)} scales={len(self.scales)} include_native={self.include_native} "
            f"expected_entries<={expected_entries} graspdata_tag={self.graspdata_tag}"
        )

        obj_iter = objects
        if self.verbose:
            obj_iter = tqdm(objects, desc=f"objdata:{self.objdata_tag}", leave=False)

        for obj_idx, obj in enumerate(obj_iter, start=1):
            if not isinstance(obj, dict):
                continue

            object_name = str(obj.get("name") or obj.get("object_id") or "").strip()
            if not object_name:
                self._log("[DatasetObjects] skip unnamed object in objdata manifest")
                continue
            if self.object_names is not None and object_name not in self.object_names:
                continue

            scales_available = {
                str(v).strip()
                for v in obj.get("scales_available", [])
                if str(v).strip()
            }
            asset_recs: List[Dict] = []
            if self.include_native:
                if self.native_tag() not in scales_available:
                    raise FileNotFoundError(
                        f"Missing objdata manifest scale tag for {object_name}: native under {self.objdata_tag}."
                    )
                native_rec = self._existing_native_assets(object_name=object_name)
                if native_rec is None:
                    raise FileNotFoundError(
                        f"Missing objdata asset for {object_name} scale=native under {self.objdata_tag}. "
                        "Rebuild with prepare_object_assets.py --force."
                    )
                asset_recs.append(native_rec)

            for scale in self.scales:
                scale_tag = self.scale_tag(float(scale))
                if scale_tag not in scales_available:
                    raise FileNotFoundError(
                        f"Missing objdata manifest scale tag for {object_name}: {scale_tag} under {self.objdata_tag}."
                    )
                existing_rec = self._existing_scale_assets(
                    object_name=object_name, scale=float(scale)
                )
                if existing_rec is None:
                    raise FileNotFoundError(
                        f"Missing objdata asset for {object_name} scale={float(scale):.6f} "
                        f"under {self.objdata_tag}. Rebuild with prepare_object_assets.py --force."
                    )
                asset_recs.append(existing_rec)

            for rec in sorted(asset_recs, key=lambda item: str(item["scale_tag"])):
                scale_tag = str(rec["scale_tag"])
                asset_dir_abs = Path(rec["xml_abs"]).resolve().parent
                output_dir_abs = (
                    self.generated_dataset_root_path
                    / self.graspdata_tag
                    / object_name
                    / scale_tag
                ).resolve()
                info = {
                    "global_id": gid,
                    "object_scale_key": f"{object_name}__{scale_tag}",
                    "object_name": object_name,
                    "asset_dir_abs": str(asset_dir_abs),
                    "output_dir_abs": str(output_dir_abs),
                    "coacd_abs": rec["coacd_abs"],
                    "mjcf_abs": rec["xml_abs"],
                    "urdf_abs": rec["urdf_abs"],
                    "scale_tag": scale_tag,
                    "scale": rec["scale"],
                    "is_native": bool(rec.get("is_native", False)),
                }
                if self.pc_subdir:
                    info["global_pc_abs"] = str(
                        asset_dir_abs / self.pc_subdir / "global_pc.npy"
                    )
                    info["global_normals_abs"] = str(
                        asset_dir_abs / self.pc_subdir / "global_normals.npy"
                    )
                self.items.append(info)
                self._key_to_index[str(info["object_scale_key"])] = gid
                gid += 1

            if not self.verbose and obj_idx % 200 == 0 and len(objects) >= 500:
                elapsed = time.perf_counter() - dataset_start
                print(
                    f"[DatasetObjects] objdata_tag={self.objdata_tag} progress={obj_idx}/{len(objects)} "
                    f"indexed={gid} elapsed={elapsed:.1f}s"
                )

        elapsed = time.perf_counter() - dataset_start
        print(
            f"[DatasetObjects] done objdata_tag={self.objdata_tag} "
            f"indexed={gid} elapsed={elapsed:.1f}s"
        )

    def _shared_coacd_path(self, object_name: str, is_native: bool) -> Path:
        object_root = self.generated_dataset_root_path / self.objdata_tag / object_name
        shared_subdir = (
            ScaleDatasetBuilder.RAW_MESH_SUBDIR
            if is_native
            else ScaleDatasetBuilder.NORMALIZED_MESH_SUBDIR
        )
        return (object_root / shared_subdir / "coacd.obj").resolve()

    def _shared_mesh_root(self, object_name: str, is_native: bool) -> Path:
        object_root = self.generated_dataset_root_path / self.objdata_tag / object_name
        shared_subdir = (
            ScaleDatasetBuilder.RAW_MESH_SUBDIR
            if is_native
            else ScaleDatasetBuilder.NORMALIZED_MESH_SUBDIR
        )
        return (object_root / shared_subdir).resolve()

    def _existing_asset_by_tag(
        self,
        object_name: str,
        scale_tag: str,
        scale_value: Optional[float],
    ) -> Optional[Dict]:
        scale_dir = (
            self.generated_dataset_root_path
            / self.objdata_tag
            / object_name
            / scale_tag
        )
        xml_path = scale_dir / "object.xml"
        urdf_path = scale_dir / "object.urdf"
        is_native = scale_value is None
        shared_root = self._shared_mesh_root(
            object_name=object_name, is_native=is_native
        )
        coacd_path = shared_root / "coacd.obj"
        manifold_path = shared_root / "manifold.obj"
        convex_dir = shared_root / "convex_parts"

        if (
            (not xml_path.exists())
            or (not coacd_path.exists())
            or (not manifold_path.exists())
            or (not convex_dir.is_dir())
            or (not any(path.is_file() for path in convex_dir.glob("*.obj")))
        ):
            return None

        return {
            "object_name": object_name,
            "scale": scale_value,
            "scale_tag": scale_tag,
            "is_native": is_native,
            "coacd_abs": str(coacd_path.resolve()),
            "xml_abs": str(xml_path.resolve()),
            "urdf_abs": str(urdf_path.resolve()),
        }

    def _existing_scale_assets(self, object_name: str, scale: float) -> Optional[Dict]:
        return self._existing_asset_by_tag(
            object_name=object_name,
            scale_tag=self.scale_tag(float(scale)),
            scale_value=float(scale),
        )

    def _existing_native_assets(self, object_name: str) -> Optional[Dict]:
        return self._existing_asset_by_tag(
            object_name=object_name,
            scale_tag=self.native_tag(),
            scale_value=None,
        )

    @staticmethod
    def _entry_scale(info: Dict, apply_scale: bool = True) -> float:
        if not apply_scale or bool(info.get("is_native", False)):
            return 1.0
        scale = info.get("scale")
        if scale is None:
            return 1.0
        return float(scale)

    def _load_mesh(
        self,
        mesh_or_path: Union[str, Path, trimesh.Trimesh],
        scale: float = 1.0,
    ) -> trimesh.Trimesh:
        if isinstance(mesh_or_path, trimesh.Trimesh):
            mesh = mesh_or_path.copy()
        else:
            mesh_path = str(mesh_or_path)
            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            mesh = mesh.copy()

        scale_value = float(scale)
        if abs(scale_value - 1.0) > 1e-12:
            mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_value
        return mesh

    def _get_shared_mesh_root(self, info: Dict) -> Path:
        asset_dir = Path(str(info["asset_dir_abs"])).resolve()
        shared_subdir = (
            ScaleDatasetBuilder.RAW_MESH_SUBDIR
            if bool(info.get("is_native", False))
            else ScaleDatasetBuilder.NORMALIZED_MESH_SUBDIR
        )
        return (asset_dir.parent / shared_subdir).resolve()

    def _get_shared_coacd_path(self, info: Dict) -> Path:
        return self._get_shared_mesh_root(info) / "coacd.obj"

    def _get_shared_manifold_path(self, info: Dict) -> Path:
        return self._get_shared_mesh_root(info) / "manifold.obj"

    def _get_shared_convex_parts_paths(self, info: Dict) -> List[Path]:
        convex_dir = self._get_shared_mesh_root(info) / "convex_parts"
        return sorted(
            path.resolve() for path in convex_dir.glob("*.obj") if path.is_file()
        )

    def load_entry_mesh(
        self,
        info: Dict,
        kind: str = "coacd",
        apply_scale: bool = True,
    ) -> trimesh.Trimesh:
        scale_value = self._entry_scale(info, apply_scale=apply_scale)
        if kind == "coacd":
            return self._load_mesh(
                self._get_shared_coacd_path(info), scale=scale_value
            )
        if kind == "manifold":
            return self._load_mesh(
                self._get_shared_manifold_path(info), scale=scale_value
            )
        if kind == "convex_parts":
            meshes = [
                self._load_mesh(path, scale=scale_value)
                for path in self._get_shared_convex_parts_paths(info)
            ]
            if not meshes:
                raise FileNotFoundError(
                    f"Missing convex_parts for {info.get('object_scale_key')}: {self._get_shared_mesh_root(info)}"
                )
            return trimesh.util.concatenate(meshes)
        raise ValueError("kind must be one of ['coacd', 'manifold', 'convex_parts']")

    def _sample_surface_o3d(
        self,
        obj_path: str,
        n_points: int = 4096,
        method: str = "uniform",
        preview: bool = False,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts, norms = sample_surface_o3d(
            obj_path=obj_path,
            n_points=n_points,
            method=method,
            scale=float(scale),
        )
        if preview:
            self._preview_pointcloud_with_normals(pts, norms)
        return pts, norms

    def sample_surface_for_entry(
        self,
        info: Dict,
        n_points: int = 4096,
        method: str = "uniform",
        preview: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._sample_surface_o3d(
            obj_path=str(self._get_shared_coacd_path(info)),
            n_points=n_points,
            method=method,
            preview=preview,
            scale=self._entry_scale(info, apply_scale=True),
        )

    def _preview_pointcloud_with_normals(
        self, pts: np.ndarray, norms: np.ndarray
    ) -> None:
        preview_pointcloud_with_normals(pts, norms)
