"""Build per-scale object assets under datasets/<config>/<object>/scaleXXX/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import trimesh

DEFAULT_FIXED_SCALES = [0.06, 0.08, 0.10, 0.12, 0.14]


class ScaleDatasetBuilder:
    """Generate scaled convex parts + coacd + MJCF for one object/scale."""

    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self._norm_cache: Dict[str, Dict] = {}

    @staticmethod
    def scale_tag(scale: float) -> str:
        return f"scale{int(round(float(scale) * 1000)):03d}"

    def _load_normalized_meshes(self, object_info: Dict) -> Dict:
        object_name = object_info["object_name"]
        cached = self._norm_cache.get(object_name)
        if cached is not None:
            return cached

        # Load coacd once; prefer scene mode to preserve OBJ groups (o convex_0/1/...).
        loaded = trimesh.load(object_info["coacd_abs"], process=False, force="scene")
        if isinstance(loaded, trimesh.Scene):
            convex_meshes = [g.copy() for g in loaded.geometry.values()]
        else:
            mesh = loaded if isinstance(loaded, trimesh.Trimesh) else trimesh.load(object_info["coacd_abs"], process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            # Fallback: split disconnected components as convex parts.
            convex_meshes = [m.copy() for m in mesh.split(only_watertight=False)]
            if not convex_meshes:
                convex_meshes = [mesh.copy()]

        merged_convex = trimesh.util.concatenate(convex_meshes)
        center = np.asarray(merged_convex.bounding_box.centroid, dtype=np.float64)
        extent = float(np.max(merged_convex.extents))
        if extent <= 1e-12:
            raise ValueError(f"Invalid object extent for normalization: {object_name}")

        norm_convex: List[trimesh.Trimesh] = []
        for mesh in convex_meshes:
            mc = mesh.copy()
            mc.vertices = (mc.vertices - center) / extent
            norm_convex.append(mc)

        norm_coacd = merged_convex.copy()
        norm_coacd.vertices = (norm_coacd.vertices - center) / extent

        rec = {
            "object_name": object_name,
            "norm_convex": norm_convex,
            "norm_coacd": norm_coacd,
            "orig_coacd_volume": self._mesh_volume_safe(merged_convex),
            "norm_volume": self._mesh_volume_safe(norm_coacd),
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

    def _render_object_xml(
        self,
        object_name: str,
        mass_kg_scaled: float,
        inertia_diag_scaled: np.ndarray,
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
                f'      <inertial pos="0 0 0" mass="{mass_kg_scaled:.10f}" '
                f'diaginertia="{inertia_diag_scaled[0]:.12e} {inertia_diag_scaled[1]:.12e} {inertia_diag_scaled[2]:.12e}"/>'
            ),
            *geom_lines,
            '    </body>',
            '  </worldbody>',
            '</mujoco>',
        ]
        return "\n".join(xml_lines) + "\n"

    def build_scale_assets(
        self,
        config_stem: str,
        object_info: Dict,
        scale: float,
        mass_kg: float,
        principal_moments: Sequence[float],
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

        convex_rel_paths = [os.path.relpath(p, scale_dir).replace("\\", "/") for p in scaled_convex_paths]
        xml_text = self._render_object_xml(
            object_name=object_name,
            mass_kg_scaled=mass_scaled,
            inertia_diag_scaled=inertia_scaled,
            convex_rel_paths=convex_rel_paths,
        )
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
        mass_kg: float,
        principal_moments: Sequence[float],
        overwrite: bool = False,
    ) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for scale in scales:
            rec = self.build_scale_assets(
                config_stem=config_stem,
                object_info=object_info,
                scale=float(scale),
                mass_kg=float(mass_kg),
                principal_moments=principal_moments,
                overwrite=overwrite,
            )
            out[self.scale_tag(float(scale))] = rec
        return out
