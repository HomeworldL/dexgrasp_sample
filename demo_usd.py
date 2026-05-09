#!/usr/bin/env python3
"""Convert one prepared MJCF object asset into USD and print a collision summary."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_ASSET_DIR = Path("datasets/objdata_YCB/YCB_001_chips_can/native")
DEFAULT_OUTPUT_DIR = Path("tmp/demo_usd")
COLLISION_APPROX_CHOICES = [
    "boundingCube",
    "boundingSphere",
    "convexDecomposition",
    "convexHull",
    "meshSimplification",
    "none",
    "sdf",
]


parser = argparse.ArgumentParser(
    description="Convert one prepared MJCF object asset into USD and print collision summary."
)
parser.add_argument(
    "--asset-dir",
    type=Path,
    default=DEFAULT_ASSET_DIR,
    help="Prepared object-scale asset directory containing object.xml.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory used to store temporary USD demo artifacts.",
)
parser.add_argument("--fix-base", action="store_true", help="Fix the imported base in the generated USD.")
parser.add_argument("--import-sites", action="store_true", help="Import MJCF <site> tags.")
parser.add_argument(
    "--collision-approximation",
    choices=COLLISION_APPROX_CHOICES,
    default=None,
    help="Optional post-conversion override for MeshCollisionAPI approximation.",
)
parser.add_argument("--force", action="store_true", help="Delete old demo outputs and regenerate them.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaacsim.core.utils.extensions import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pxr import Sdf, Usd, UsdPhysics


def _asset_key(asset_dir: Path) -> str:
    return f"{asset_dir.parent.name}__{asset_dir.name}"


def _validate_asset_dir(asset_dir: Path) -> tuple[Path, Path]:
    asset_dir = asset_dir.resolve()
    xml_path = asset_dir / "object.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing MJCF asset: {xml_path}")
    return asset_dir, xml_path


def _convert_mjcf(xml_path: Path, usd_path: Path) -> Path:
    sim_utils.create_new_stage()
    enable_extension("isaacsim.asset.importer.mjcf")
    converter_cfg = MjcfConverterCfg(
        asset_path=str(xml_path),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        fix_base=bool(args_cli.fix_base),
        import_sites=bool(args_cli.import_sites),
        force_usd_conversion=True,
        make_instanceable=True,
    )
    converter = MjcfConverter(converter_cfg)
    return Path(converter.usd_path).resolve()


def _collect_mesh_collision_prims(usd_path: Path) -> list[dict]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    items = []
    for prim in stage.Traverse():
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            continue
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        items.append(
            {
                "path": str(prim.GetPath()),
                "approximation": str(approx) if approx is not None else None,
            }
        )
    return items


def _override_collision_approximation(usd_path: Path, approximation: str) -> list[dict]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    updated = []
    for prim in stage.Traverse():
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            continue
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        api = UsdPhysics.MeshCollisionAPI(prim)
        api.GetApproximationAttr().Set(approximation)
        updated.append({"path": str(prim.GetPath()), "approximation": approximation})

    stage.Save()
    return updated


def _print_summary(asset_dir: Path, output_dir: Path, usd_path: Path, collision_items: list[dict], source_xml: Path) -> None:
    physics_layer = output_dir / "configuration/object_physics.usd"
    print(f"asset_dir={asset_dir}")
    print(f"source_mjcf={source_xml}")
    print(f"output_dir={output_dir}")
    print(f"usd_path={usd_path}")
    print(f"physics_layer={physics_layer}")
    print(f"mesh_collision_count={len(collision_items)}")
    for item in collision_items:
        print(f"{item['path']} approximation={item['approximation']}")


def main() -> None:
    asset_dir, xml_path = _validate_asset_dir(args_cli.asset_dir)
    output_dir = args_cli.output_dir.resolve() / _asset_key(asset_dir)
    if args_cli.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usd_path = _convert_mjcf(xml_path, output_dir / "object.usd")
    physics_usd_path = output_dir / "configuration/object_physics.usd"
    if args_cli.collision_approximation is not None:
        _override_collision_approximation(physics_usd_path, args_cli.collision_approximation)
    collision_items = _collect_mesh_collision_prims(physics_usd_path)
    _print_summary(asset_dir, output_dir, usd_path, collision_items, xml_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
