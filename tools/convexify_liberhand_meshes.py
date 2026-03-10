#!/usr/bin/env python3
"""
Convert liberhand STL meshes to convex OBJ meshes and patch MJCF mesh paths.

Default behavior:
1) Read all STL files under assets/hands/liberhand/meshes
2) Export convex hull meshes as *_convex.obj in the same folder
3) Replace .stl/.STL mesh file paths in liberhand_right.xml with *_convex.obj
4) Create a backup xml file once: liberhand_right.xml.bak
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import trimesh


def iter_stl_files(mesh_dir: Path) -> Iterable[Path]:
    return sorted(
        p for p in mesh_dir.iterdir() if p.is_file() and p.suffix.lower() == ".stl"
    )


def load_mesh_as_single_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        return trimesh.util.concatenate(parts)
    raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)}")


def convexify_stl_to_obj(stl_path: Path, out_obj_path: Path, overwrite: bool) -> bool:
    if out_obj_path.exists() and not overwrite:
        return False
    mesh = load_mesh_as_single_trimesh(stl_path)
    hull = mesh.convex_hull
    out_obj_path.parent.mkdir(parents=True, exist_ok=True)
    hull.export(out_obj_path.as_posix())
    return True


def patch_xml_mesh_paths(xml_path: Path, backup: bool = True) -> int:
    if backup:
        backup_path = xml_path.with_suffix(xml_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(xml_path, backup_path)

    text = xml_path.read_text(encoding="utf-8")
    pattern = re.compile(r'file="([^"]+?)(?i:\.stl)"')
    new_text, num_replaced = pattern.subn(r'file="\1_convex.obj"', text)
    if num_replaced > 0 and new_text != text:
        xml_path.write_text(new_text, encoding="utf-8")
    return num_replaced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert liberhand STL meshes to convex OBJ and patch MJCF paths."
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("assets/hands/liberhand/meshes"),
        help="Directory containing source STL files.",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=Path("assets/hands/liberhand/liberhand_right.xml"),
        help="MJCF file to patch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_convex.obj files.",
    )
    parser.add_argument(
        "--no-patch-xml",
        action="store_true",
        help="Only convert meshes, do not modify xml.",
    )
    args = parser.parse_args()

    mesh_dir = args.mesh_dir.resolve()
    xml_path = args.xml_path.resolve()

    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    if not args.no_patch_xml and not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    stl_files = list(iter_stl_files(mesh_dir))
    if not stl_files:
        raise RuntimeError(f"No STL files found in: {mesh_dir}")

    converted = 0
    skipped = 0
    for stl_path in stl_files:
        out_obj_path = stl_path.with_name(stl_path.stem + "_convex.obj")
        did_convert = convexify_stl_to_obj(stl_path, out_obj_path, args.overwrite)
        if did_convert:
            converted += 1
        else:
            skipped += 1

    replaced = 0
    if not args.no_patch_xml:
        replaced = patch_xml_mesh_paths(xml_path, backup=True)

    print(f"STL files found: {len(stl_files)}")
    print(f"Converted convex OBJ: {converted}")
    print(f"Skipped existing OBJ: {skipped}")
    if not args.no_patch_xml:
        print(f"XML mesh path replacements: {replaced}")
        print(f"Patched xml: {xml_path}")


if __name__ == "__main__":
    main()
