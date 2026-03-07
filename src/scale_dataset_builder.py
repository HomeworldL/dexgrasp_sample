"""Build per-scale object assets under datasets/<config>/<object>/scaleXXX/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

DEFAULT_FIXED_SCALES = [0.06, 0.08, 0.10, 0.12, 0.14]


class ScaleDatasetBuilder:
    """Generate scaled convex parts + coacd + MJCF for one object/scale."""

    def __init__(self, base_output_dir: str, object_mass_kg: float = 0.1):
        self.base_output_dir = Path(base_output_dir)
        self.object_mass_kg = float(object_mass_kg)
        self._norm_cache: Dict[str, Dict] = {}

    @staticmethod
    def scale_tag(scale: float) -> str:
        return f"scale{int(round(float(scale) * 1000)):03d}"

    def _load_normalized_meshes(self, object_info: Dict) -> Dict:
        object_name = object_info["object_name"]
        cached = self._norm_cache.get(object_name)
        if cached is not None:
            return cached

        convex_meshes: List[trimesh.Trimesh] = []
        for p in object_info["convex_parts_abs"]:
            m = trimesh.load(p, process=False)
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(list(m.geometry.values()))
            convex_meshes.append(m)

        coacd_mesh = trimesh.load(object_info["coacd_abs"], process=False)
        if isinstance(coacd_mesh, trimesh.Scene):
            coacd_mesh = trimesh.util.concatenate(list(coacd_mesh.geometry.values()))

        merged_convex = trimesh.util.concatenate(convex_meshes)
        center = np.asarray(merged_convex.bounding_box.centroid, dtype=np.float64)
        extent = float(np.max(merged_convex.extents))
        if extent <= 1e-12:
            raise ValueError(f"Invalid object extent for normalization: {object_name}")

        norm_convex = []
        for mesh in convex_meshes:
            mc = mesh.copy()
            mc.vertices = (mc.vertices - center) / extent
            norm_convex.append(mc)

        norm_coacd = coacd_mesh.copy()
        norm_coacd.vertices = (norm_coacd.vertices - center) / extent

        rec = {
            "object_name": object_name,
            "norm_center": center,
            "norm_extent": extent,
            "norm_convex": norm_convex,
            "norm_coacd": norm_coacd,
        }
        self._norm_cache[object_name] = rec
        return rec

    def _compute_inertial(self, scaled_coacd: trimesh.Trimesh) -> Tuple[float, np.ndarray]:
        volume = float(scaled_coacd.volume)
        if not np.isfinite(volume) or volume <= 1e-12:
            ext = np.asarray(scaled_coacd.bounding_box.extents, dtype=np.float64)
            volume = float(np.prod(np.clip(ext, 1e-8, None)))

        density = float(self.object_mass_kg / max(volume, 1e-12))

        ext = np.asarray(scaled_coacd.bounding_box.extents, dtype=np.float64)
        x, y, z = np.clip(ext, 1e-8, None)
        m = self.object_mass_kg
        ixx = (m / 12.0) * (y * y + z * z)
        iyy = (m / 12.0) * (x * x + z * z)
        izz = (m / 12.0) * (x * x + y * y)
        return density, np.array([ixx, iyy, izz], dtype=np.float64)

    def _render_object_xml(
        self,
        object_name: str,
        density: float,
        inertia_diag: np.ndarray,
        convex_rel_paths: List[str],
    ) -> str:
        mesh_assets = [f'    <mesh name="convex_{i}" file="{rel}"/>' for i, rel in enumerate(convex_rel_paths)]
        geom_lines = [
            (
                f'      <geom name="{object_name}_collision_{i}" type="mesh" mesh="convex_{i}" '
                'contype="1" conaffinity="1" group="1" rgba="0.70 0.70 0.70 1"/>'
            )
            for i in range(len(convex_rel_paths))
        ]

        xml_lines = [
            '<mujoco model="object">',
            '  <compiler angle="radian" coordinate="local"/>',
            '  <option gravity="0 0 0"/>',
            '  <asset>',
            *mesh_assets,
            '  </asset>',
            '  <worldbody>',
            f'    <body name="{object_name}" pos="0 0 0">',
            f'      <freejoint name="{object_name}_joint"/>',
            (
                f'      <inertial pos="0 0 0" mass="{self.object_mass_kg:.8f}" '
                f'diaginertia="{inertia_diag[0]:.10f} {inertia_diag[1]:.10f} {inertia_diag[2]:.10f}"/>'
            ),
            *geom_lines,
            '    </body>',
            '  </worldbody>',
            f'  <!-- density={density:.12f} mass={self.object_mass_kg:.6f}kg -->',
            '</mujoco>',
        ]
        return "\n".join(xml_lines) + "\n"

    def build_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scale: float,
        overwrite: bool = False,
    ) -> Dict:
        norm = self._load_normalized_meshes(object_info)
        object_name = norm["object_name"]
        scale_value = float(scale)
        scale_dir = self.base_output_dir / config_stem / object_name / self.scale_tag(scale_value)
        convex_dir = scale_dir / "convex_parts"
        coacd_path = scale_dir / "coacd.obj"
        xml_path = scale_dir / "object.xml"

        if not overwrite and xml_path.exists() and coacd_path.exists() and convex_dir.is_dir():
            convex_parts_abs = [str(p.resolve()) for p in sorted(convex_dir.glob("*.obj"))]
            if convex_parts_abs:
                return {
                    "object_name": object_name,
                    "scale": scale_value,
                    "scale_tag": self.scale_tag(scale_value),
                    "coacd_abs": str(coacd_path.resolve()),
                    "convex_parts_abs": convex_parts_abs,
                    "xml_abs": str(xml_path.resolve()),
                }

        convex_dir.mkdir(parents=True, exist_ok=True)

        scaled_convex_paths: List[Path] = []
        for i, mesh in enumerate(norm["norm_convex"]):
            sm = mesh.copy()
            sm.vertices = sm.vertices * scale_value
            part_path = convex_dir / f"part_{i:03d}.obj"
            sm.export(part_path)
            scaled_convex_paths.append(part_path)

        scaled_coacd = norm["norm_coacd"].copy()
        scaled_coacd.vertices = scaled_coacd.vertices * scale_value
        scaled_coacd.export(coacd_path)

        density, inertia_diag = self._compute_inertial(scaled_coacd)
        convex_rel_paths = [os.path.relpath(p, scale_dir).replace("\\", "/") for p in scaled_convex_paths]
        xml_text = self._render_object_xml(object_name, density, inertia_diag, convex_rel_paths)
        xml_path.write_text(xml_text, encoding="utf-8")

        return {
            "object_name": object_name,
            "scale": scale_value,
            "scale_tag": self.scale_tag(scale_value),
            "coacd_abs": str(coacd_path.resolve()),
            "convex_parts_abs": [str(p.resolve()) for p in scaled_convex_paths],
            "xml_abs": str(xml_path.resolve()),
        }

    def build_multi_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scales: List[float],
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        object_name = object_info["object_name"]
        iterator = scales
        if show_progress:
            iterator = tqdm(scales, desc=f"prebuild:{object_name}", leave=False)

        out: Dict[str, Dict] = {}
        for scale in iterator:
            rec = self.build_scale_assets(config_stem, object_info, float(scale), overwrite=overwrite)
            out[self.scale_tag(float(scale))] = rec
        return out
