"""DatasetObjects: manifest-driven object-scale index.

Design notes
- Index from prepared objdata manifest under generated dataset root.
- Object assets live under an object asset tag, separate from grasp outputs.
- Output info granularity is object-scale (one info per scale tag).
- `native` is treated as an optional peer of `scaleXXX` during read-only indexing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from tqdm import tqdm

from utils.utils_pointcloud import preview_pointcloud_with_normals, sample_surface_o3d


class DatasetObjects:
    """Objdata-manifest-driven object-scale index.

    Args:
        scales: Explicit scale list from config.
        objdata_tag: Output tag for object-scale assets, e.g. ``objdata_YCB``.
        include_native: Include ``native`` assets when True.
        graspdata_tag: Output tag for hand-specific grasp artifacts.
        generated_dataset_root: Root folder for generated scaled assets.
        verbose: Print all skip/build messages when True.

    Info format (one row per object-scale):
        {
            "global_id": int,
            "object_scale_key": str,
            "object_name": str,
            "asset_dir_abs": str,
            "output_dir_abs": str,
            "coacd_abs": str,
            "convex_parts_abs": List[str],
            "scale_tag": str,
            "scale": Optional[float],
            "is_native": bool,
            "mjcf_abs": str,
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
        self.verbose = bool(verbose)

        self.items: List[Dict] = []
        self._key_to_index: Dict[str, int] = {}

        self._build_from_manifests()

        if not self.items:
            raise RuntimeError(f"No valid object-scale entries under objdata_tag={self.objdata_tag}.")

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

    def _build_from_manifests(self) -> None:
        gid = 0
        dataset_start = time.perf_counter()
        manifest_path = (
            self.generated_dataset_root_path / self.objdata_tag / "manifest.process_meshes.json"
        ).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Objdata manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        objects = manifest.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError(f"Invalid manifest objects list: {manifest_path}")

        expected_entries = len(objects) * (len(self.scales) + (1 if self.include_native else 0))
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

            scales_available = {str(v).strip() for v in obj.get("scales_available", []) if str(v).strip()}
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
                        "First-time creation or rebuild must use prepare_object_assets.py --force."
                    )
                asset_recs.append(native_rec)

            for scale in self.scales:
                scale_tag = self.scale_tag(float(scale))
                if scale_tag not in scales_available:
                    raise FileNotFoundError(
                        f"Missing objdata manifest scale tag for {object_name}: {scale_tag} under {self.objdata_tag}."
                    )
                existing_rec = self._existing_scale_assets(object_name=object_name, scale=float(scale))
                if existing_rec is None:
                    raise FileNotFoundError(
                        f"Missing objdata asset for {object_name} scale={float(scale):.6f} "
                        f"under {self.objdata_tag}. First-time creation or rebuild must use "
                        "prepare_object_assets.py --force."
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
                    "convex_parts_abs": list(rec["convex_parts_abs"]),
                    "mjcf_abs": rec["xml_abs"],
                    "scale_tag": scale_tag,
                    "scale": rec["scale"],
                    "is_native": bool(rec.get("is_native", False)),
                }
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

    def _existing_asset_by_tag(self, object_name: str, scale_tag: str, scale_value: Optional[float]) -> Optional[Dict]:
        scale_dir = self.generated_dataset_root_path / self.objdata_tag / object_name / scale_tag
        convex_dir = scale_dir / "convex_parts"
        coacd_path = scale_dir / "coacd.obj"
        manifold_path = scale_dir / "manifold.obj"
        xml_path = scale_dir / "object.xml"
        urdf_path = scale_dir / "object.urdf"

        if (
            (not xml_path.exists())
            or (not urdf_path.exists())
            or (not coacd_path.exists())
            or (not manifold_path.exists())
            or (not convex_dir.is_dir())
        ):
            return None

        convex_parts_abs = [str(p.resolve()) for p in sorted(convex_dir.glob("*.obj")) if p.is_file()]
        if not convex_parts_abs:
            return None

        return {
            "object_name": object_name,
            "scale": scale_value,
            "scale_tag": scale_tag,
            "is_native": scale_value is None,
            "coacd_abs": str(coacd_path.resolve()),
            "convex_parts_abs": convex_parts_abs,
            "xml_abs": str(xml_path.resolve()),
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
    def scale_tag(scale: float) -> str:
        return f"scale{int(round(float(scale) * 1000)):03d}"

    @staticmethod
    def native_tag() -> str:
        return "native"

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

    def sample_surface_o3d(
        self,
        obj_path: str,
        n_points: int = 4096,
        method: str = "uniform",
        preview: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
