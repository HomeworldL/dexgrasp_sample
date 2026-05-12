"""Build object assets with object-level shared meshes and scale-level thin dirs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import trimesh

from utils.utils_file import native_tag as _native_tag, scale_tag as _scale_tag


class ScaleDatasetBuilder:
    """Generate shared object meshes and per-scale thin assets."""

    MIN_PART_MAX_EXTENT = 1e-5
    MIN_PART_AREA = 1e-10
    MIN_PART_VOLUME = 1e-12
    MIN_OBJECT_MASS = 1e-10
    MIN_OBJECT_INERTIA = 1e-12
    RAW_MESH_SUBDIR = "meshes"
    NORMALIZED_MESH_SUBDIR = "meshes_normalized"

    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self._raw_cache: Dict[str, Dict] = {}
        self._norm_cache: Dict[str, Dict] = {}

    @staticmethod
    def scale_tag(scale: float) -> str:
        return _scale_tag(scale)

    @staticmethod
    def native_tag() -> str:
        return _native_tag()

    @staticmethod
    def _load_mesh_as_trimesh(mesh_path: str) -> trimesh.Trimesh:
        loaded = trimesh.load(mesh_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(loaded)!r}")
        return mesh.copy()

    def _load_raw_meshes(self, object_info: Dict) -> Dict:
        object_name = object_info["object_name"]
        coacd_abs = str(object_info["coacd_abs"])
        manifold_abs = str(object_info["manifold_abs"])

        cached = self._raw_cache.get(object_name)
        if cached is not None:
            return cached

        merged = self._load_mesh_as_trimesh(coacd_abs)
        convex_meshes = [m.copy() for m in merged.split(only_watertight=False)]
        if not convex_meshes:
            convex_meshes = [merged.copy()]

        manifold_mesh = self._load_mesh_as_trimesh(manifold_abs)
        merged_convex = trimesh.util.concatenate(convex_meshes)
        record = {
            "object_name": object_name,
            "source_coacd_abs": coacd_abs,
            "source_manifold_abs": manifold_abs,
            "raw_convex": convex_meshes,
            "raw_merged": merged_convex,
            "raw_manifold": manifold_mesh.copy(),
            "orig_coacd_volume": self._mesh_volume_safe(merged_convex),
        }
        self._raw_cache[object_name] = record
        return record

    def _load_normalized_meshes(self, object_info: Dict) -> Dict:
        raw = self._load_raw_meshes(object_info)
        object_name = raw["object_name"]
        cached = self._norm_cache.get(object_name)
        if cached is not None:
            return cached

        convex_meshes = raw["raw_convex"]
        merged_convex = raw["raw_merged"]
        center = np.asarray(merged_convex.bounding_box.centroid, dtype=np.float64)
        extent = float(np.max(merged_convex.extents))
        if extent <= 1e-12:
            raise ValueError(f"Invalid object extent for normalization: {object_name}")

        norm_convex: List[trimesh.Trimesh] = []
        for mesh in convex_meshes:
            mc = mesh.copy()
            mc.vertices = (mc.vertices - center) / extent
            norm_convex.append(mc)

        manifold = raw["raw_manifold"].copy()
        manifold.vertices = (manifold.vertices - center) / extent

        rec = {
            "object_name": object_name,
            "norm_convex": norm_convex,
            "norm_manifold": manifold,
            "orig_coacd_volume": float(raw["orig_coacd_volume"]),
            "norm_volume": self._mesh_volume_safe(trimesh.util.concatenate(norm_convex)),
        }
        self._norm_cache[object_name] = rec
        return rec

    @staticmethod
    def _mesh_volume_safe(mesh: trimesh.Trimesh) -> float:
        try:
            vol = float(abs(mesh.volume))
        except Exception:
            vol = 0.0
        if not np.isfinite(vol) or vol <= 1e-16:
            try:
                vol = float(abs(mesh.convex_hull.volume))
            except Exception:
                vol = 0.0
        if not np.isfinite(vol) or vol <= 1e-16:
            raise ValueError("Failed to compute a valid positive mesh volume.")
        return vol

    def _scaled_part_is_valid(self, mesh: trimesh.Trimesh) -> bool:
        if not isinstance(mesh, trimesh.Trimesh):
            return False
        if mesh.vertices is None or mesh.faces is None:
            return False
        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            return False

        extents = np.asarray(mesh.extents, dtype=np.float64)
        if not np.all(np.isfinite(extents)):
            return False
        if float(np.max(extents)) <= self.MIN_PART_MAX_EXTENT:
            return False

        area = float(getattr(mesh, "area", 0.0) or 0.0)
        if (not np.isfinite(area)) or area <= self.MIN_PART_AREA:
            return False

        try:
            volume = self._mesh_volume_safe(mesh)
        except Exception:
            return False
        return bool(np.isfinite(volume) and volume > self.MIN_PART_VOLUME)

    def object_root(self, config_stem: str, object_name: str) -> Path:
        return self.base_output_dir / config_stem / object_name

    def shared_mesh_root(self, config_stem: str, object_name: str, normalized: bool) -> Path:
        subdir = self.NORMALIZED_MESH_SUBDIR if normalized else self.RAW_MESH_SUBDIR
        return self.object_root(config_stem, object_name) / subdir

    def asset_dir(self, config_stem: str, object_name: str, scale_tag: str) -> Path:
        return self.object_root(config_stem, object_name) / scale_tag

    def _convex_parts_dir(self, mesh_root: Path) -> Path:
        return mesh_root / "convex_parts"

    def _shared_meshes_ready(self, mesh_root: Path) -> bool:
        convex_dir = self._convex_parts_dir(mesh_root)
        required = [mesh_root / "coacd.obj", mesh_root / "manifold.obj"]
        if not all(path.exists() for path in required):
            return False
        if not convex_dir.is_dir():
            return False
        return any(path.is_file() for path in convex_dir.glob("*.obj"))

    def _build_object_xml(
        self,
        object_name: str,
        scale_tag: str,
        mass_kg_scaled: float,
        inertia_diag_scaled: np.ndarray,
        convex_rel_paths: List[str],
        mesh_scale: Optional[float],
    ) -> str:
        asset_name = f"{object_name}_{scale_tag}"
        scale_attr = ""
        if mesh_scale is not None:
            scale_attr = f' scale="{mesh_scale:.10f} {mesh_scale:.10f} {mesh_scale:.10f}"'
        mesh_assets = [
            f'    <mesh name="{asset_name}_convex_{i}" file="{rel}"{scale_attr}/>'
            for i, rel in enumerate(convex_rel_paths)
        ]
        geom_lines = [
            (
                f'      <geom name="{asset_name}_collision_{i}" type="mesh" mesh="{asset_name}_convex_{i}" '
                'contype="1" conaffinity="1" group="1" rgba="0.70 0.70 0.70 1"/>'
            )
            for i in range(len(convex_rel_paths))
        ]

        xml_lines = [
            f'<mujoco model="{asset_name}">',
            '  <compiler angle="radian" coordinate="local"/>',
            '  <option gravity="0 0 0"/>',
            '  <asset>',
            *mesh_assets,
            '  </asset>',
            '  <worldbody>',
            f'    <body name="{asset_name}" pos="0 0 0">',
            f'      <freejoint name="{asset_name}_joint"/>',
            (
                f'      <inertial pos="0 0 0" mass="{mass_kg_scaled:.10f}" '
                f'diaginertia="{inertia_diag_scaled[0]:.12e} {inertia_diag_scaled[1]:.12e} {inertia_diag_scaled[2]:.12e}"/>'
            ),
            *geom_lines,
            '    </body>',
            '  </worldbody>',
            '</mujoco>',
        ]
        return "\n".join(xml_lines) + "\n"

    def _build_object_urdf(
        self,
        object_name: str,
        scale_tag: str,
        mass_kg_scaled: float,
        inertia_diag_scaled: np.ndarray,
        mesh_rel_path: str,
        mesh_scale: Optional[float],
    ) -> str:
        model_name = f"{object_name}_{scale_tag}"
        scale_attr = ""
        if mesh_scale is not None:
            scale_attr = f' scale="{mesh_scale:.10f} {mesh_scale:.10f} {mesh_scale:.10f}"'
        urdf_lines = [
            '<?xml version="1.0"?>',
            f'<robot name="{model_name}">',
            "  <static>false</static>",
            '  <link name="base_link">',
            "    <inertial>",
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <mass value="{mass_kg_scaled:.10f}"/>',
            (
                "      <inertia "
                f'ixx="{inertia_diag_scaled[0]:.12e}" ixy="0" ixz="0" '
                f'iyy="{inertia_diag_scaled[1]:.12e}" iyz="0" '
                f'izz="{inertia_diag_scaled[2]:.12e}"/>'
            ),
            "    </inertial>",
            '    <visual name="visual">',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            "      <geometry>",
            f'        <mesh filename="{mesh_rel_path}"{scale_attr}/>',
            "      </geometry>",
            "    </visual>",
            '    <collision name="collision">',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            "      <geometry>",
            f'        <mesh filename="{mesh_rel_path}"{scale_attr}/>',
            "      </geometry>",
            "    </collision>",
            "  </link>",
            "</robot>",
        ]
        return "\n".join(urdf_lines) + "\n"

    def _export_convex_parts(self, meshes: Sequence[trimesh.Trimesh], convex_dir: Path) -> List[Path]:
        convex_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for i, mesh in enumerate(meshes):
            part_path = convex_dir / f"part_{i:03d}.obj"
            mesh.export(part_path)
            paths.append(part_path)
        return paths

    def _valid_mesh_parts(self, meshes: Sequence[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        valid_parts: List[trimesh.Trimesh] = []
        for mesh in meshes:
            mc = mesh.copy()
            if self._scaled_part_is_valid(mc):
                valid_parts.append(mc)
        return valid_parts

    def build_shared_mesh_assets(
        self,
        config_stem: str,
        object_info: Dict,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        object_name = str(object_info["object_name"])
        raw_root = self.shared_mesh_root(config_stem, object_name, normalized=False)
        norm_root = self.shared_mesh_root(config_stem, object_name, normalized=True)

        if (
            not overwrite
            and self._shared_meshes_ready(raw_root)
            and self._shared_meshes_ready(norm_root)
        ):
            return {
                "raw_root_abs": str(raw_root.resolve()),
                "normalized_root_abs": str(norm_root.resolve()),
                "raw_coacd_abs": str((raw_root / "coacd.obj").resolve()),
                "normalized_coacd_abs": str((norm_root / "coacd.obj").resolve()),
            }

        raw = self._load_raw_meshes(object_info)
        norm = self._load_normalized_meshes(object_info)

        raw_root.mkdir(parents=True, exist_ok=True)
        norm_root.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for mesh_root in [raw_root, norm_root]:
                if mesh_root.exists():
                    shutil.rmtree(mesh_root)
                mesh_root.mkdir(parents=True, exist_ok=True)

        raw_valid_parts = self._valid_mesh_parts(raw["raw_convex"])
        if not raw_valid_parts:
            raise ValueError(f"All native convex parts were filtered out as invalid for {object_name}.")
        norm_valid_parts = self._valid_mesh_parts(norm["norm_convex"])
        if not norm_valid_parts:
            raise ValueError(f"All normalized convex parts were filtered out as invalid for {object_name}.")

        shutil.copy2(raw["source_coacd_abs"], raw_root / "coacd.obj")
        shutil.copy2(raw["source_manifold_abs"], raw_root / "manifold.obj")
        self._export_convex_parts(raw_valid_parts, self._convex_parts_dir(raw_root))

        norm_scene = trimesh.Scene()
        for i, mesh in enumerate(norm_valid_parts):
            norm_scene.add_geometry(mesh.copy(), node_name=f"part_{i:03d}", geom_name=f"part_{i:03d}")
        norm_scene.export(str(norm_root / "coacd.obj"))
        norm["norm_manifold"].copy().export(norm_root / "manifold.obj")
        self._export_convex_parts(norm_valid_parts, self._convex_parts_dir(norm_root))

        return {
            "raw_root_abs": str(raw_root.resolve()),
            "normalized_root_abs": str(norm_root.resolve()),
            "raw_coacd_abs": str((raw_root / "coacd.obj").resolve()),
            "normalized_coacd_abs": str((norm_root / "coacd.obj").resolve()),
        }

    def build_native_assets(
        self,
        config_stem: str,
        object_info: Dict,
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict:
        self.build_shared_mesh_assets(config_stem=config_stem, object_info=object_info, overwrite=overwrite)
        raw = self._load_raw_meshes(object_info)

        object_name = raw["object_name"]
        scale_tag = self.native_tag()
        scale_dir = self.asset_dir(config_stem, object_name, scale_tag)
        xml_path = scale_dir / "object.xml"
        urdf_path = scale_dir / "object.urdf"

        if not overwrite and xml_path.exists() and urdf_path.exists():
            return {
                "object_name": object_name,
                "scale": None,
                "scale_tag": scale_tag,
                "is_native": True,
                "coacd_abs": str(
                    (self.shared_mesh_root(config_stem, object_name, normalized=False) / "coacd.obj").resolve()
                ),
                "xml_abs": str(xml_path.resolve()),
                "urdf_abs": str(urdf_path.resolve()),
            }

        scale_dir.mkdir(parents=True, exist_ok=True)

        mass_native = float(mass_kg)
        inertia_native = np.asarray(principal_moments, dtype=np.float64).reshape(3)
        if (
            (not np.isfinite(mass_native))
            or mass_native <= self.MIN_OBJECT_MASS
            or np.any(~np.isfinite(inertia_native))
            or np.any(inertia_native <= self.MIN_OBJECT_INERTIA)
        ):
            raise ValueError(
                f"Native inertial is below valid threshold for {object_name}: "
                f"mass={mass_native:.6e}, inertia={inertia_native.tolist()}"
            )

        raw_root = self.shared_mesh_root(config_stem, object_name, normalized=False)
        convex_paths = sorted(self._convex_parts_dir(raw_root).glob("*.obj"))
        if not convex_paths:
            raise ValueError(f"Missing shared native convex parts for {object_name}.")
        convex_rel_paths = [os.path.relpath(path, scale_dir).replace("\\", "/") for path in convex_paths]
        mesh_rel_path = os.path.relpath(raw_root / "manifold.obj", scale_dir).replace("\\", "/")
        xml_path.write_text(
            self._build_object_xml(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_native,
                inertia_diag_scaled=inertia_native,
                convex_rel_paths=convex_rel_paths,
                mesh_scale=None,
            ),
            encoding="utf-8",
        )
        urdf_path.write_text(
            self._build_object_urdf(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_native,
                inertia_diag_scaled=inertia_native,
                mesh_rel_path=mesh_rel_path,
                mesh_scale=None,
            ),
            encoding="utf-8",
        )

        return {
            "object_name": object_name,
            "scale": None,
            "scale_tag": scale_tag,
            "is_native": True,
            "coacd_abs": str((raw_root / "coacd.obj").resolve()),
            "xml_abs": str(xml_path.resolve()),
            "urdf_abs": str(urdf_path.resolve()),
        }

    def build_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scale: float,
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict:
        self.build_shared_mesh_assets(config_stem=config_stem, object_info=object_info, overwrite=overwrite)
        norm = self._load_normalized_meshes(object_info)

        object_name = norm["object_name"]
        scale_value = float(scale)
        scale_tag = self.scale_tag(scale_value)
        scale_dir = self.asset_dir(config_stem, object_name, scale_tag)
        xml_path = scale_dir / "object.xml"
        urdf_path = scale_dir / "object.urdf"

        if not overwrite and xml_path.exists() and urdf_path.exists():
            return {
                "object_name": object_name,
                "scale": scale_value,
                "scale_tag": scale_tag,
                "is_native": False,
                "coacd_abs": str(
                    (self.shared_mesh_root(config_stem, object_name, normalized=True) / "coacd.obj").resolve()
                ),
                "xml_abs": str(xml_path.resolve()),
                "urdf_abs": str(urdf_path.resolve()),
            }

        scale_dir.mkdir(parents=True, exist_ok=True)

        m0 = float(mass_kg)
        p0 = np.asarray(principal_moments, dtype=np.float64).reshape(3)
        v0 = float(norm["orig_coacd_volume"])
        vn = float(norm["norm_volume"])
        if not np.isfinite(v0) or v0 <= 1e-16:
            raise ValueError(f"Invalid original coacd volume for {object_name}: {v0}")
        if not np.isfinite(vn) or vn <= 1e-16:
            raise ValueError(f"Invalid normalized coacd volume for {object_name}: {vn}")

        volume_ratio = vn / v0
        mass_scaled = m0 * volume_ratio * (scale_value ** 3)
        inertia_scaled = p0 * (volume_ratio ** (5.0 / 3.0)) * (scale_value ** 5)
        if (
            (not np.isfinite(mass_scaled))
            or mass_scaled <= self.MIN_OBJECT_MASS
            or np.any(~np.isfinite(inertia_scaled))
            or np.any(np.asarray(inertia_scaled, dtype=np.float64) <= self.MIN_OBJECT_INERTIA)
        ):
            raise ValueError(
                f"Scaled inertial is below valid threshold for {object_name} at scale={scale_value:.6f}: "
                f"mass={mass_scaled:.6e}, inertia={np.asarray(inertia_scaled).tolist()}"
            )

        norm_root = self.shared_mesh_root(config_stem, object_name, normalized=True)
        convex_paths = sorted(self._convex_parts_dir(norm_root).glob("*.obj"))
        if not convex_paths:
            raise ValueError(f"Missing shared normalized convex parts for {object_name}.")
        convex_rel_paths = [os.path.relpath(path, scale_dir).replace("\\", "/") for path in convex_paths]
        mesh_rel_path = os.path.relpath(norm_root / "manifold.obj", scale_dir).replace("\\", "/")
        xml_path.write_text(
            self._build_object_xml(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_scaled,
                inertia_diag_scaled=inertia_scaled,
                convex_rel_paths=convex_rel_paths,
                mesh_scale=scale_value,
            ),
            encoding="utf-8",
        )
        urdf_path.write_text(
            self._build_object_urdf(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_scaled,
                inertia_diag_scaled=inertia_scaled,
                mesh_rel_path=mesh_rel_path,
                mesh_scale=scale_value,
            ),
            encoding="utf-8",
        )

        return {
            "object_name": object_name,
            "scale": scale_value,
            "scale_tag": scale_tag,
            "is_native": False,
            "coacd_abs": str((norm_root / "coacd.obj").resolve()),
            "xml_abs": str(xml_path.resolve()),
            "urdf_abs": str(urdf_path.resolve()),
        }

    def build_multi_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scales: Sequence[float],
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        self.build_shared_mesh_assets(
            config_stem=config_stem,
            object_info=object_info,
            overwrite=overwrite,
        )
        for scale in scales:
            rec = self.build_scale_assets(
                config_stem=config_stem,
                object_info=object_info,
                scale=float(scale),
                mass_kg=float(mass_kg),
                principal_moments=principal_moments,
                overwrite=overwrite,
            )
            out[str(rec["scale_tag"])] = rec
        return out
