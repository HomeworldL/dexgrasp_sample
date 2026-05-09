#!/usr/bin/env python3
"""Convert prepared MJCF object assets into USD beside each object.xml."""

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
    objdata_tag_from_config,
    usd_convert_cfg_from_config,
)


parser = argparse.ArgumentParser(
    description="Convert prepared object-scale MJCF assets into USD beside each object.xml."
)
parser.add_argument("-c", "--config", type=str, default=DEFAULT_ASSET_CONFIG_PATH)
parser.add_argument("--force", action="store_true", help="Override usd_convert.force from config.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaacsim.core.utils.extensions import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pxr import Sdf, Usd, UsdPhysics


def _resolve_entries(cfg: dict, config_path: str) -> list[dict]:
    entries = _load_existing_entries(cfg, config_path)
    if not entries:
        raise ValueError("No matching prepared objdata assets found for USD conversion.")
    return entries


def _remove_existing_usd_outputs(asset_dir: Path) -> None:
    for rel_path in ["object.usd", "config.yaml", ".asset_hash"]:
        path = asset_dir / rel_path
        if path.exists():
            path.unlink()
    config_dir = asset_dir / "configuration"
    if config_dir.exists():
        shutil.rmtree(config_dir)


def _usd_outputs_ready(asset_dir: Path) -> bool:
    return (
        (asset_dir / "object.usd").exists()
        and (asset_dir / "config.yaml").exists()
        and (asset_dir / ".asset_hash").exists()
        and (asset_dir / "configuration").is_dir()
    )


def _convert_asset(
    *,
    xml_path: Path,
    asset_dir: Path,
    fix_base: bool,
    import_sites: bool,
    force_usd_conversion: bool,
) -> Path:
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


def _read_mjcf_inertial(xml_path: Path) -> dict:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    body = root.find("./worldbody/body")
    if body is None:
        raise ValueError(f"Missing worldbody/body in {xml_path}")
    inertial = body.find("./inertial")
    if inertial is None:
        raise ValueError(f"Missing inertial tag in {xml_path}")
    return {
        "body_name": str(body.attrib["name"]),
        "mass": float(inertial.attrib["mass"]),
        "diag": tuple(float(v) for v in inertial.attrib["diaginertia"].split()),
        "pos": tuple(float(v) for v in inertial.attrib["pos"].split()),
    }


def _read_usd_inertial(usd_path: Path) -> dict:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    matches = []
    for prim in stage.Traverse():
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            continue
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            matches.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "mass": float(api.GetMassAttr().Get()),
                    "diag": tuple(float(v) for v in api.GetDiagonalInertiaAttr().Get()),
                    "pos": tuple(float(v) for v in api.GetCenterOfMassAttr().Get()),
                }
            )
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one MassAPI prim in {usd_path}, got {len(matches)}")
    return matches[0]


def _close(a: float, b: float, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    return abs(a - b) <= (atol + rtol * abs(a))


def _verify_inertial(xml_path: Path, usd_path: Path) -> dict:
    mjcf = _read_mjcf_inertial(xml_path)
    usd = _read_usd_inertial(usd_path)
    ok_mass = _close(mjcf["mass"], usd["mass"])
    ok_diag = all(_close(a, b) for a, b in zip(mjcf["diag"], usd["diag"]))
    ok_pos = all(_close(a, b) for a, b in zip(mjcf["pos"], usd["pos"]))
    return {
        "ok": bool(ok_mass and ok_diag and ok_pos),
        "mjcf": mjcf,
        "usd": usd,
        "ok_mass": ok_mass,
        "ok_diag": ok_diag,
        "ok_pos": ok_pos,
    }


def main() -> None:
    start = time.perf_counter()
    cfg = load_asset_config(args_cli.config)
    usd_cfg = usd_convert_cfg_from_config(cfg)
    objdata_tag = objdata_tag_from_config(cfg, args_cli.config)
    entries = _resolve_entries(cfg, args_cli.config)

    if not bool(usd_cfg["enabled"]):
        raise ValueError("usd_convert.enabled is false in config; refusing to run conversion.")

    force = bool(args_cli.force or usd_cfg["force"])
    fix_base = bool(usd_cfg["fix_base"])
    import_sites = bool(usd_cfg["import_sites"])
    verify_inertial = bool(usd_cfg["verify_inertial"])

    enable_extension("isaacsim.asset.importer.mjcf")

    success_count = 0
    error_count = 0
    verify_fail_count = 0
    entry_iter = tqdm(
        entries,
        desc=f"usd:{objdata_tag}",
        total=len(entries),
        dynamic_ncols=True,
        leave=True,
    )
    for entry in entry_iter:
        asset_dir = Path(entry["asset_dir_abs"]).resolve()
        xml_path = asset_dir / "object.xml"
        entry_iter.set_postfix(
            current=str(entry["object_scale_key"]),
            success=success_count,
            error=error_count,
            verify_fail=verify_fail_count,
        )
        try:
            if force:
                _remove_existing_usd_outputs(asset_dir)
            if force or (not _usd_outputs_ready(asset_dir)):
                usd_path = _convert_asset(
                    xml_path=xml_path,
                    asset_dir=asset_dir,
                    fix_base=fix_base,
                    import_sites=import_sites,
                    force_usd_conversion=force,
                )
            else:
                usd_path = (asset_dir / "object.usd").resolve()
            verify_msg = "verify=skipped"
            if verify_inertial:
                check = _verify_inertial(usd_path=usd_path, xml_path=xml_path)
                if not check["ok"]:
                    verify_fail_count += 1
                    raise RuntimeError(
                        "Inertial mismatch: "
                        f"mjcf_mass={check['mjcf']['mass']} usd_mass={check['usd']['mass']} "
                        f"mjcf_diag={check['mjcf']['diag']} usd_diag={check['usd']['diag']} "
                        f"mjcf_pos={check['mjcf']['pos']} usd_pos={check['usd']['pos']}"
                    )
                verify_msg = (
                    "verify=ok "
                    f"mass={check['usd']['mass']:.9f} "
                    f"diag={check['usd']['diag']}"
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
        f"[prepare_object_usds] objdata_tag={objdata_tag} entries={len(entries)} "
        f"success_count={success_count} error_count={error_count} "
        f"verify_fail_count={verify_fail_count} total_sec={time.perf_counter() - start:.3f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
