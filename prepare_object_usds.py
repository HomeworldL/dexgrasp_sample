#!/usr/bin/env python3
"""Convert prepared object assets into USD beside each object asset directory."""

from __future__ import annotations

import argparse
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher
from tqdm import tqdm

from prepare_object_assets import _load_existing_entries
from utils.utils_file import (
    DEFAULT_ASSET_CONFIG_PATH,
    load_asset_config,
    objdata_tag_cfg,
    usd_convert_cfg,
)

parser = argparse.ArgumentParser(
    description="Convert prepared object-scale assets into USD beside each object asset directory."
)
parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
parser.add_argument(
    "--force", action="store_true", help="Override usd_convert.force from config."
)

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.sim.converters import (
    MjcfConverter,
    MjcfConverterCfg,
    UrdfConverter,
    UrdfConverterCfg,
)
from pxr import Sdf, Usd, UsdPhysics


def _resolve_entries(cfg: dict, config_path: str) -> list[dict]:
    """Reuse the objdata entry scan so USD conversion follows the same asset set."""
    entries = _load_existing_entries(cfg, config_path)
    if not entries:
        raise ValueError(
            "No matching prepared objdata assets found for USD conversion."
        )
    return entries


def _remove_existing_usd_outputs(asset_dir: Path) -> None:
    """Delete previously generated USD-side outputs before a forced reconvert."""
    for rel_path in ["object.usd", "config.yaml", ".asset_hash"]:
        path = asset_dir / rel_path
        if path.exists():
            path.unlink()
    config_dir = asset_dir / "configuration"
    if config_dir.exists():
        shutil.rmtree(config_dir)


def _usd_outputs_ready(asset_dir: Path) -> bool:
    """Check whether the expected USD payload already exists in place."""
    return (
        (asset_dir / "object.usd").exists()
        and (asset_dir / "config.yaml").exists()
        and (asset_dir / ".asset_hash").exists()
        and (asset_dir / "configuration").is_dir()
    )


def _convert_asset_mjcf(
    *,
    xml_path: Path,
    asset_dir: Path,
    fix_base: bool,
    import_sites: bool,
    force_usd_conversion: bool,
) -> Path:
    """Run IsaacLab's MJCF converter for one prepared asset directory."""
    sim_utils.create_new_stage()
    converter_cfg = MjcfConverterCfg(
        asset_path=str(xml_path),
        usd_dir=str(asset_dir),
        usd_file_name="object.usd",
        fix_base=fix_base,
        import_sites=import_sites,
        force_usd_conversion=force_usd_conversion,
        make_instanceable=False,
    )
    converter = MjcfConverter(converter_cfg)
    return Path(converter.usd_path).resolve()


def _convert_asset_urdf(
    *,
    urdf_path: Path,
    asset_dir: Path,
    fix_base: bool,
    merge_joints: bool,
    make_instanceable: bool,
    convex_decompose_mesh: bool,
    force_usd_conversion: bool,
) -> Path:
    """Run IsaacLab's URDF converter for one prepared asset directory."""
    sim_utils.create_new_stage()
    collider_type = "convex_decomposition" if convex_decompose_mesh else "convex_hull"
    converter_cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(asset_dir),
        usd_file_name="object.usd",
        fix_base=fix_base,
        merge_fixed_joints=merge_joints,
        force_usd_conversion=force_usd_conversion,
        make_instanceable=make_instanceable,
        joint_drive=None,
        collision_from_visuals=False,
        collider_type=collider_type,
    )
    converter = UrdfConverter(converter_cfg)
    return Path(converter.usd_path).resolve()


def _read_mjcf_inertial(xml_path: Path) -> dict:
    """Read the expected mass and diagonal inertia directly from prepared MJCF."""
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    body = root.find("./worldbody/body")
    if body is None:
        raise ValueError(f"Missing worldbody/body in {xml_path}")
    inertial = body.find("./inertial")
    if inertial is None:
        raise ValueError(f"Missing inertial tag in {xml_path}")
    return {
        "mass": float(inertial.attrib["mass"]),
        "diag": tuple(float(v) for v in inertial.attrib["diaginertia"].split()),
    }


def _read_urdf_inertial(urdf_path: Path) -> dict:
    """Read the expected mass and diagonal inertia directly from prepared URDF."""
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    link = root.find("./link[@name='base_link']")
    if link is None:
        raise ValueError(f"Missing base_link in {urdf_path}")
    inertial = link.find("./inertial")
    if inertial is None:
        raise ValueError(f"Missing inertial tag in {urdf_path}")
    mass_node = inertial.find("./mass")
    inertia_node = inertial.find("./inertia")
    if mass_node is None or inertia_node is None:
        raise ValueError(f"Missing inertial mass/inertia tags in {urdf_path}")
    return {
        "mass": float(mass_node.attrib["value"]),
        "diag": (
            float(inertia_node.attrib["ixx"]),
            float(inertia_node.attrib["iyy"]),
            float(inertia_node.attrib["izz"]),
        ),
    }


def _read_usd_inertial(usd_path: Path) -> dict:
    """Extract the single physics mass payload authored into the converted USD."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    matches = []
    for prim in stage.Traverse():
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            continue
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass_val = api.GetMassAttr().Get()
            diag_val = api.GetDiagonalInertiaAttr().Get()
            if mass_val is None or diag_val is None:
                continue
            matches.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "mass": float(mass_val),
                    "diag": tuple(float(v) for v in diag_val),
                }
            )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one MassAPI prim in {usd_path}, got {len(matches)}"
        )
    return matches[0]


def _close(a: float, b: float, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """Small helper for inertial-value comparisons across source and USD."""
    return abs(a - b) <= (atol + rtol * abs(a))


def _verify_inertial(expected: dict, usd_path: Path) -> dict:
    """Compare converted USD inertials against the prepared source asset."""
    usd = _read_usd_inertial(usd_path)
    ok_mass = _close(expected["mass"], usd["mass"])
    ok_diag = all(_close(a, b) for a, b in zip(expected["diag"], usd["diag"]))
    return {
        "ok": bool(ok_mass and ok_diag),
        "expected": expected,
        "usd": usd,
        "ok_mass": ok_mass,
        "ok_diag": ok_diag,
    }


def main() -> None:
    """Convert every prepared objdata entry into in-place USD outputs.

    High-level flow:
    1. load the asset config and select the backend-specific converter policy
    2. scan prepared objdata entries from the existing manifest
    3. convert each entry in place, optionally reusing or replacing prior outputs
    4. optionally verify USD inertials against the prepared MJCF/URDF source
    """
    start = time.perf_counter()
    cfg = load_asset_config(args_cli.config)
    usd_cfg = usd_convert_cfg(cfg)
    objdata_tag = objdata_tag_cfg(cfg, args_cli.config)
    entries = _resolve_entries(cfg, args_cli.config)

    force = bool(args_cli.force or usd_cfg["force"])
    backend = str(usd_cfg["backend"])
    fix_base = bool(usd_cfg["fix_base"])
    import_sites = bool(usd_cfg["import_sites"])
    verify_inertial = bool(usd_cfg["verify_inertial"])
    merge_joints = bool(usd_cfg["merge_joints"])
    make_instanceable = bool(usd_cfg["make_instanceable"])
    convex_decompose_mesh = bool(usd_cfg["convex_decompose_mesh"])

    success_count = 0
    error_count = 0
    verify_fail_count = 0
    entry_iter = tqdm(
        entries,
        desc=f"usd:{objdata_tag}:{backend}",
        total=len(entries),
        dynamic_ncols=True,
        leave=True,
    )
    for entry in entry_iter:
        asset_dir = Path(entry["asset_dir_abs"]).resolve()
        xml_path = asset_dir / "object.xml"
        urdf_path = asset_dir / "object.urdf"
        entry_iter.set_postfix(
            current=str(entry["object_scale_key"]),
            success=success_count,
            error=error_count,
            verify_fail=verify_fail_count,
        )
        try:
            if backend == "urdf" and (not urdf_path.exists()):
                raise FileNotFoundError(f"Missing URDF asset: {urdf_path}")
            if backend == "mjcf" and (not xml_path.exists()):
                raise FileNotFoundError(f"Missing MJCF asset: {xml_path}")

            # Conversion can either start clean or reuse an existing output tree,
            # depending on the force/ready state computed above.
            if force:
                _remove_existing_usd_outputs(asset_dir)
            if force or (not _usd_outputs_ready(asset_dir)):
                if backend == "urdf":
                    usd_path = _convert_asset_urdf(
                        urdf_path=urdf_path,
                        asset_dir=asset_dir,
                        fix_base=fix_base,
                        merge_joints=merge_joints,
                        make_instanceable=make_instanceable,
                        convex_decompose_mesh=convex_decompose_mesh,
                        force_usd_conversion=force,
                    )
                else:
                    usd_path = _convert_asset_mjcf(
                        xml_path=xml_path,
                        asset_dir=asset_dir,
                        fix_base=fix_base,
                        import_sites=import_sites,
                        force_usd_conversion=force,
                    )
            else:
                usd_path = (asset_dir / "object.usd").resolve()

            # Inertial verification is a post-convert consistency check. It does
            # not affect converter selection, only whether the result is accepted.
            if verify_inertial:
                expected = (
                    _read_urdf_inertial(urdf_path)
                    if backend == "urdf"
                    else _read_mjcf_inertial(xml_path)
                )
                check = _verify_inertial(expected=expected, usd_path=usd_path)
                if not check["ok"]:
                    verify_fail_count += 1
                    raise RuntimeError(
                        "Inertial mismatch: "
                        f"expected_mass={check['expected']['mass']} usd_mass={check['usd']['mass']} "
                        f"expected_diag={check['expected']['diag']} usd_diag={check['usd']['diag']}"
                    )
            success_count += 1
        except Exception as exc:
            error_count += 1
            tqdm.write(
                f"error object_scale_key={entry['object_scale_key']} "
                f"asset_dir={asset_dir} error_type={type(exc).__name__} error={exc}"
            )
        entry_iter.set_postfix(
            current=str(entry["object_scale_key"]),
            success=success_count,
            error=error_count,
            verify_fail=verify_fail_count,
        )

    print(
        f"[prepare_object_usds] objdata_tag={objdata_tag} backend={backend} entries={len(entries)} "
        f"success_count={success_count} error_count={error_count} "
        f"verify_fail_count={verify_fail_count} total_sec={time.perf_counter() - start:.3f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
