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
    COACD_OBJ_NAME = "coacd.obj"
    MANIFOLD_OBJ_NAME = "manifold.obj"
    OBJECT_XML_NAME = "object.xml"
    OBJECT_URDF_NAME = "object.urdf"
    CONVEX_PARTS_SUBDIR = "convex_parts"
    CONVEX_PART_OBJ_GLOB = "*.obj"
    CONVEX_PART_NAME_TEMPLATE = "part_{index:03d}.obj"
    SCENE_PART_NAME_TEMPLATE = "part_{index:03d}"
    VISUAL_OBJ_NAME = "visual.obj"
    VISUAL_MTL_NAME = "textured_visual.mtl"
    VISUAL_TEXTURE_NAME = "textured_visual.png"
    VISUAL_SIDECAR_NAMES = (VISUAL_MTL_NAME, VISUAL_TEXTURE_NAME)
    VISUAL_ASSET_NAMES = (VISUAL_OBJ_NAME,) + VISUAL_SIDECAR_NAMES
    SHARED_MESH_NAMES = (COACD_OBJ_NAME, MANIFOLD_OBJ_NAME)

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
        source_dir = self._source_dir(object_info)
        coacd_path = self.coacd_path(source_dir)
        manifold_path = self.manifold_path(source_dir)

        cached = self._raw_cache.get(object_name)
        if cached is not None:
            return cached

        merged = self._load_mesh_as_trimesh(str(coacd_path))
        convex_meshes = [m.copy() for m in merged.split(only_watertight=False)]
        if not convex_meshes:
            convex_meshes = [merged.copy()]

        manifold_mesh = self._load_mesh_as_trimesh(str(manifold_path))
        merged_convex = trimesh.util.concatenate(convex_meshes)
        record = {
            "object_name": object_name,
            "source_dir_abs": str(source_dir),
            "source_coacd_abs": str(coacd_path),
            "source_manifold_abs": str(manifold_path),
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
        center, extent = self._normalization_transform(merged_convex, object_name)

        norm_convex: List[trimesh.Trimesh] = []
        for mesh in convex_meshes:
            mc = mesh.copy()
            mc.vertices = self._normalize_vertices(mc.vertices, center, extent)
            norm_convex.append(mc)

        manifold = raw["raw_manifold"].copy()
        manifold.vertices = self._normalize_vertices(manifold.vertices, center, extent)

        rec = {
            "object_name": object_name,
            "norm_convex": norm_convex,
            "norm_manifold": manifold,
            "orig_coacd_volume": float(raw["orig_coacd_volume"]),
            "norm_volume": self._mesh_volume_safe(
                trimesh.util.concatenate(norm_convex)
            ),
            "norm_center": center,
            "norm_extent": extent,
        }
        self._norm_cache[object_name] = rec
        return rec

    @staticmethod
    def _normalization_transform(
        mesh: trimesh.Trimesh, object_name: str
    ) -> tuple[np.ndarray, float]:
        center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
        extent = float(np.max(mesh.extents))
        if extent <= 1e-12:
            raise ValueError(f"Invalid object extent for normalization: {object_name}")
        return center, extent

    @staticmethod
    def _normalize_vertices(
        vertices: np.ndarray, center: np.ndarray, extent: float
    ) -> np.ndarray:
        return (np.asarray(vertices, dtype=np.float64) - center) / float(extent)

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

    def shared_mesh_root(
        self, config_stem: str, object_name: str, normalized: bool
    ) -> Path:
        subdir = self.NORMALIZED_MESH_SUBDIR if normalized else self.RAW_MESH_SUBDIR
        return self.object_root(config_stem, object_name) / subdir

    def asset_dir(self, config_stem: str, object_name: str, scale_tag: str) -> Path:
        return self.object_root(config_stem, object_name) / scale_tag

    @classmethod
    def coacd_path(cls, mesh_root: Path) -> Path:
        return Path(mesh_root) / cls.COACD_OBJ_NAME

    @classmethod
    def manifold_path(cls, mesh_root: Path) -> Path:
        return Path(mesh_root) / cls.MANIFOLD_OBJ_NAME

    @classmethod
    def object_xml_path(cls, asset_dir: Path) -> Path:
        return Path(asset_dir) / cls.OBJECT_XML_NAME

    @classmethod
    def object_urdf_path(cls, asset_dir: Path) -> Path:
        return Path(asset_dir) / cls.OBJECT_URDF_NAME

    @classmethod
    def convex_parts_dir(cls, mesh_root: Path) -> Path:
        return Path(mesh_root) / cls.CONVEX_PARTS_SUBDIR

    @classmethod
    def convex_part_path(cls, convex_dir: Path, index: int) -> Path:
        return Path(convex_dir) / cls.CONVEX_PART_NAME_TEMPLATE.format(index=index)

    @classmethod
    def scene_part_name(cls, index: int) -> str:
        return cls.SCENE_PART_NAME_TEMPLATE.format(index=index)

    @classmethod
    def convex_part_paths(cls, mesh_root: Path) -> list[Path]:
        return sorted(cls.convex_parts_dir(mesh_root).glob(cls.CONVEX_PART_OBJ_GLOB))

    @classmethod
    def shared_mesh_paths(cls, mesh_root: Path) -> list[Path]:
        return [Path(mesh_root) / name for name in cls.SHARED_MESH_NAMES]

    @staticmethod
    def _source_dir(object_info: Dict) -> Path:
        source_dir_raw = object_info.get("source_dir_abs")
        if not source_dir_raw:
            raise ValueError("object_info must provide 'source_dir_abs'.")
        return Path(str(source_dir_raw)).resolve()

    @staticmethod
    def _rel_path_if_exists(path: Path, base_dir: Path) -> Optional[str]:
        if not path.exists():
            return None
        return os.path.relpath(path, base_dir).replace("\\", "/")

    def _shared_visual_rel_paths(
        self, mesh_root: Path, asset_dir: Path
    ) -> tuple[Optional[str], Optional[str]]:
        return (
            self._rel_path_if_exists(mesh_root / self.VISUAL_OBJ_NAME, asset_dir),
            self._rel_path_if_exists(mesh_root / self.VISUAL_TEXTURE_NAME, asset_dir),
        )

    def _shared_meshes_ready(self, mesh_root: Path) -> bool:
        convex_dir = self.convex_parts_dir(mesh_root)
        if not all(path.exists() for path in self.shared_mesh_paths(mesh_root)):
            return False
        if not convex_dir.is_dir():
            return False
        return any(path.is_file() for path in self.convex_part_paths(mesh_root))

    def _shared_visual_assets_ready(self, object_info: Dict, mesh_root: Path) -> bool:
        source_dir = self._source_dir(object_info)
        required_names = [
            name for name in self.VISUAL_ASSET_NAMES if (source_dir / name).exists()
        ]
        return all((mesh_root / name).exists() for name in required_names)

    def _write_obj_with_normalized_vertices(
        self, src_path: Path, dst_path: Path, center: np.ndarray, extent: float
    ) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with src_path.open("r", encoding="utf-8") as src, dst_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        xyz = self._normalize_vertices(
                            np.asarray(
                                [float(parts[1]), float(parts[2]), float(parts[3])],
                                dtype=np.float64,
                            ),
                            center,
                            extent,
                        )
                        rest = " ".join(parts[4:])
                        suffix = f" {rest}" if rest else ""
                        dst.write(f"v {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}{suffix}\n")
                        continue
                dst.write(line)

    def _write_visual_assets(
        self,
        source_dir: Path,
        mesh_root: Path,
        center: Optional[np.ndarray] = None,
        extent: Optional[float] = None,
    ) -> None:
        visual_src = source_dir / self.VISUAL_OBJ_NAME
        visual_dst = mesh_root / self.VISUAL_OBJ_NAME
        if visual_src.exists():
            if center is None or extent is None:
                shutil.copy2(visual_src, visual_dst)
            else:
                self._write_obj_with_normalized_vertices(
                    visual_src,
                    visual_dst,
                    np.asarray(center, dtype=np.float64),
                    float(extent),
                )
        for name in self.VISUAL_SIDECAR_NAMES:
            src_path = source_dir / name
            if src_path.exists():
                shutil.copy2(src_path, mesh_root / name)

    def _build_object_xml(
        self,
        object_name: str,
        scale_tag: str,
        mass_kg_scaled: float,
        inertia_diag_scaled: np.ndarray,
        convex_rel_paths: List[str],
        visual_rel_path: Optional[str],
        texture_rel_path: Optional[str],
        mesh_scale: Optional[float],
    ) -> str:
        asset_name = f"{object_name}_{scale_tag}"
        scale_attr = ""
        if mesh_scale is not None:
            scale_attr = (
                f' scale="{mesh_scale:.10f} {mesh_scale:.10f} {mesh_scale:.10f}"'
            )
        material_asset = []
        visual_geom = []
        if visual_rel_path is not None:
            mesh_assets_prefix = [
                f'    <mesh name="{asset_name}_visual_mesh" file="{visual_rel_path}"{scale_attr}/>'
            ]
            material_attr = ""
            if texture_rel_path is not None:
                material_asset = [
                    f'    <texture name="{asset_name}_texture" type="2d" file="{texture_rel_path}"/>',
                    f'    <material name="{asset_name}_material" texture="{asset_name}_texture"/>',
                ]
                material_attr = f' material="{asset_name}_material"'
            visual_geom = [
                (
                    f'      <geom name="{asset_name}_visual_geom" type="mesh" mesh="{asset_name}_visual_mesh"'
                    f'{material_attr} contype="0" conaffinity="0" group="0"/>'
                )
            ]
        else:
            mesh_assets_prefix = []
        mesh_assets = [
            f'    <mesh name="{asset_name}_convex_{i}" file="{rel}"{scale_attr}/>'
            for i, rel in enumerate(convex_rel_paths)
        ]
        geom_lines = [
            (
                f'      <geom name="{asset_name}_collision_{i}" type="mesh" mesh="{asset_name}_convex_{i}" '
                'contype="1" conaffinity="1" group="3" rgba="0.70 0.70 0.70 1"/>'
            )
            for i in range(len(convex_rel_paths))
        ]

        xml_lines = [
            f'<mujoco model="{asset_name}">',
            '  <compiler angle="radian" coordinate="local"/>',
            '  <option gravity="0 0 0"/>',
            "  <asset>",
            *mesh_assets_prefix,
            *mesh_assets,
            *material_asset,
            "  </asset>",
            "  <worldbody>",
            f'    <body name="{asset_name}" pos="0 0 0">',
            f'      <freejoint name="{asset_name}_joint"/>',
            (
                f'      <inertial pos="0 0 0" mass="{mass_kg_scaled:.10f}" '
                f'diaginertia="{inertia_diag_scaled[0]:.12e} {inertia_diag_scaled[1]:.12e} {inertia_diag_scaled[2]:.12e}"/>'
            ),
            *visual_geom,
            *geom_lines,
            "    </body>",
            "  </worldbody>",
            "</mujoco>",
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
            scale_attr = (
                f' scale="{mesh_scale:.10f} {mesh_scale:.10f} {mesh_scale:.10f}"'
            )
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

    def _build_record(
        self,
        *,
        object_name: str,
        scale,
        scale_tag: str,
        is_native: bool,
        coacd_path: Path,
        xml_path: Path,
        urdf_path: Path,
    ) -> Dict:
        return {
            "object_name": object_name,
            "scale": scale,
            "scale_tag": scale_tag,
            "is_native": is_native,
            "coacd_abs": str(coacd_path.resolve()),
            "xml_abs": str(xml_path.resolve()),
            "urdf_abs": str(urdf_path.resolve()),
        }

    def _shared_mesh_record(self, raw_root: Path, norm_root: Path) -> Dict[str, str]:
        return {
            "raw_root_abs": str(raw_root.resolve()),
            "normalized_root_abs": str(norm_root.resolve()),
            "raw_coacd_abs": str(self.coacd_path(raw_root).resolve()),
            "normalized_coacd_abs": str(self.coacd_path(norm_root).resolve()),
        }

    def _validate_object_inertial(
        self,
        *,
        object_name: str,
        context: str,
        mass_kg: float,
        inertia_diag: np.ndarray,
    ) -> None:
        if (
            (not np.isfinite(mass_kg))
            or mass_kg <= self.MIN_OBJECT_MASS
            or np.any(~np.isfinite(inertia_diag))
            or np.any(inertia_diag <= self.MIN_OBJECT_INERTIA)
        ):
            raise ValueError(
                f"{context} inertial is below valid threshold for {object_name}: "
                f"mass={mass_kg:.6e}, inertia={inertia_diag.tolist()}"
            )

    def _variant_mesh_refs(
        self,
        *,
        object_name: str,
        mesh_root: Path,
        asset_dir: Path,
        mesh_kind: str,
    ) -> tuple[list[str], str, Optional[str], Optional[str]]:
        convex_paths = self.convex_part_paths(mesh_root)
        if not convex_paths:
            raise ValueError(
                f"Missing shared {mesh_kind} convex parts for {object_name}."
            )
        convex_rel_paths = [
            os.path.relpath(path, asset_dir).replace("\\", "/") for path in convex_paths
        ]
        mesh_rel_path = os.path.relpath(
            self.manifold_path(mesh_root), asset_dir
        ).replace("\\", "/")
        visual_rel_path, texture_rel_path = self._shared_visual_rel_paths(
            mesh_root, asset_dir
        )
        return convex_rel_paths, mesh_rel_path, visual_rel_path, texture_rel_path

    def _write_variant_asset_files(
        self,
        *,
        object_name: str,
        scale_tag: str,
        asset_dir: Path,
        mesh_root: Path,
        mesh_kind: str,
        mass_kg: float,
        inertia_diag: np.ndarray,
        mesh_scale: Optional[float],
    ) -> tuple[Path, Path]:
        convex_rel_paths, mesh_rel_path, visual_rel_path, texture_rel_path = (
            self._variant_mesh_refs(
                object_name=object_name,
                mesh_root=mesh_root,
                asset_dir=asset_dir,
                mesh_kind=mesh_kind,
            )
        )
        xml_path = self.object_xml_path(asset_dir)
        urdf_path = self.object_urdf_path(asset_dir)
        xml_path.write_text(
            self._build_object_xml(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_kg,
                inertia_diag_scaled=inertia_diag,
                convex_rel_paths=convex_rel_paths,
                visual_rel_path=visual_rel_path,
                texture_rel_path=texture_rel_path,
                mesh_scale=mesh_scale,
            ),
            encoding="utf-8",
        )
        urdf_path.write_text(
            self._build_object_urdf(
                object_name=object_name,
                scale_tag=scale_tag,
                mass_kg_scaled=mass_kg,
                inertia_diag_scaled=inertia_diag,
                mesh_rel_path=mesh_rel_path,
                mesh_scale=mesh_scale,
            ),
            encoding="utf-8",
        )
        return xml_path, urdf_path

    def _export_convex_parts(
        self, meshes: Sequence[trimesh.Trimesh], convex_dir: Path
    ) -> List[Path]:
        convex_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for i, mesh in enumerate(meshes):
            part_path = self.convex_part_path(convex_dir, i)
            mesh.export(part_path)
            paths.append(part_path)
        return paths

    def _valid_mesh_parts(
        self, meshes: Sequence[trimesh.Trimesh]
    ) -> List[trimesh.Trimesh]:
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
            and self._shared_visual_assets_ready(object_info, raw_root)
            and self._shared_visual_assets_ready(object_info, norm_root)
        ):
            return self._shared_mesh_record(raw_root, norm_root)

        raw = self._load_raw_meshes(object_info)
        norm = self._load_normalized_meshes(object_info)
        source_dir = Path(str(raw["source_dir_abs"])).resolve()

        raw_root.mkdir(parents=True, exist_ok=True)
        norm_root.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for mesh_root in [raw_root, norm_root]:
                if mesh_root.exists():
                    shutil.rmtree(mesh_root)
                mesh_root.mkdir(parents=True, exist_ok=True)

        raw_valid_parts = self._valid_mesh_parts(raw["raw_convex"])
        if not raw_valid_parts:
            raise ValueError(
                f"All native convex parts were filtered out as invalid for {object_name}."
            )
        norm_valid_parts = self._valid_mesh_parts(norm["norm_convex"])
        if not norm_valid_parts:
            raise ValueError(
                f"All normalized convex parts were filtered out as invalid for {object_name}."
            )

        shutil.copy2(raw["source_coacd_abs"], self.coacd_path(raw_root))
        shutil.copy2(raw["source_manifold_abs"], self.manifold_path(raw_root))
        self._export_convex_parts(raw_valid_parts, self.convex_parts_dir(raw_root))
        self._write_visual_assets(source_dir, raw_root)

        norm_scene = trimesh.Scene()
        for i, mesh in enumerate(norm_valid_parts):
            part_name = self.scene_part_name(i)
            norm_scene.add_geometry(
                mesh.copy(), node_name=part_name, geom_name=part_name
            )
        norm_scene.export(str(self.coacd_path(norm_root)))
        norm["norm_manifold"].copy().export(self.manifold_path(norm_root))
        self._export_convex_parts(norm_valid_parts, self.convex_parts_dir(norm_root))
        self._write_visual_assets(
            source_dir=source_dir,
            mesh_root=norm_root,
            center=np.asarray(norm["norm_center"], dtype=np.float64),
            extent=float(norm["norm_extent"]),
        )

        return self._shared_mesh_record(raw_root, norm_root)

    def build_native_assets(
        self,
        config_stem: str,
        object_info: Dict,
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict:
        self.build_shared_mesh_assets(
            config_stem=config_stem, object_info=object_info, overwrite=overwrite
        )
        raw = self._load_raw_meshes(object_info)

        object_name = raw["object_name"]
        scale_tag = self.native_tag()
        scale_dir = self.asset_dir(config_stem, object_name, scale_tag)
        xml_path = self.object_xml_path(scale_dir)
        urdf_path = self.object_urdf_path(scale_dir)
        raw_root = self.shared_mesh_root(config_stem, object_name, normalized=False)

        if not overwrite and xml_path.exists() and urdf_path.exists():
            return self._build_record(
                object_name=object_name,
                scale=None,
                scale_tag=scale_tag,
                is_native=True,
                coacd_path=self.coacd_path(raw_root),
                xml_path=xml_path,
                urdf_path=urdf_path,
            )

        scale_dir.mkdir(parents=True, exist_ok=True)

        mass_native = float(mass_kg)
        inertia_native = np.asarray(principal_moments, dtype=np.float64).reshape(3)
        self._validate_object_inertial(
            object_name=object_name,
            context="Native",
            mass_kg=mass_native,
            inertia_diag=inertia_native,
        )
        xml_path, urdf_path = self._write_variant_asset_files(
            object_name=object_name,
            scale_tag=scale_tag,
            asset_dir=scale_dir,
            mesh_root=raw_root,
            mesh_kind="native",
            mass_kg=mass_native,
            inertia_diag=inertia_native,
            mesh_scale=None,
        )

        return self._build_record(
            object_name=object_name,
            scale=None,
            scale_tag=scale_tag,
            is_native=True,
            coacd_path=self.coacd_path(raw_root),
            xml_path=xml_path,
            urdf_path=urdf_path,
        )

    def build_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scale: float,
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict:
        self.build_shared_mesh_assets(
            config_stem=config_stem, object_info=object_info, overwrite=overwrite
        )
        norm = self._load_normalized_meshes(object_info)

        object_name = norm["object_name"]
        scale_value = float(scale)
        scale_tag = self.scale_tag(scale_value)
        scale_dir = self.asset_dir(config_stem, object_name, scale_tag)
        xml_path = self.object_xml_path(scale_dir)
        urdf_path = self.object_urdf_path(scale_dir)
        norm_root = self.shared_mesh_root(config_stem, object_name, normalized=True)

        if not overwrite and xml_path.exists() and urdf_path.exists():
            return self._build_record(
                object_name=object_name,
                scale=scale_value,
                scale_tag=scale_tag,
                is_native=False,
                coacd_path=self.coacd_path(norm_root),
                xml_path=xml_path,
                urdf_path=urdf_path,
            )

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
        mass_scaled = m0 * volume_ratio * (scale_value**3)
        inertia_scaled = p0 * (volume_ratio ** (5.0 / 3.0)) * (scale_value**5)
        inertia_scaled = np.asarray(inertia_scaled, dtype=np.float64)
        self._validate_object_inertial(
            object_name=object_name,
            context=f"Scaled scale={scale_value:.6f}",
            mass_kg=mass_scaled,
            inertia_diag=inertia_scaled,
        )
        xml_path, urdf_path = self._write_variant_asset_files(
            object_name=object_name,
            scale_tag=scale_tag,
            asset_dir=scale_dir,
            mesh_root=norm_root,
            mesh_kind="normalized",
            mass_kg=mass_scaled,
            inertia_diag=inertia_scaled,
            mesh_scale=scale_value,
        )

        return self._build_record(
            object_name=object_name,
            scale=scale_value,
            scale_tag=scale_tag,
            is_native=False,
            coacd_path=self.coacd_path(norm_root),
            xml_path=xml_path,
            urdf_path=urdf_path,
        )

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
