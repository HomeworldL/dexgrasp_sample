import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RUN_CONFIG_PATH = "configs/run_YCB_liberhand_right.json"
DEFAULT_ASSET_CONFIG_PATH = "configs/assets_YCB.json"
_SCALE_TAG_FACTOR = 1000

__all__ = [
    "DEFAULT_ASSET_CONFIG_PATH",
    "DEFAULT_RUN_CONFIG_PATH",
    "hand_anchor_params_cfg",
    "data_asset_scales_cfg",
    "build_logs_dir",
    "data_build_native_asset_cfg",
    "data_verbose_cfg",
    "ensure_parent_dir",
    "data_generated_dataset_root_cfg",
    "graspdata_tag_cfg",
    "graspdata_tag_from_path",
    "hand_profile_cfg",
    "hand_root_stabilization_cfg",
    "list_existing_files",
    "load_asset_config",
    "load_run_config",
    "native_tag",
    "objdata_tag_cfg",
    "object_profile_cfg",
    "parse_scale_tag",
    "data_raw_dataset_name_cfg",
    "data_raw_dataset_root_cfg",
    "relpath_str",
    "resolve_object_asset_name",
    "resolve_split_manifest_path_cfg",
    "data_run_scales_cfg",
    "safe_filename",
    "scale_tag",
    "usd_convert_cfg",
    "data_use_native_asset_cfg",
]


def _require(cfg: Dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[key]
    return cur


def _require_str(cfg: Dict, path: str) -> str:
    value = _require(cfg, path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {path} must be a non-empty string.")
    return value.strip()


def _require_float_list(cfg: Dict, path: str) -> List[float]:
    values = _require(cfg, path)
    if not isinstance(values, list):
        raise ValueError(f"Config field {path} must be a list.")
    parsed = [float(value) for value in values]
    for value in parsed:
        if value <= 0.0:
            raise ValueError(f"All {path} values must be > 0.")
    return parsed


def _bool(cfg: Dict, path: str, default: bool) -> bool:
    cur = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return bool(default)
        cur = cur[key]
    if not isinstance(cur, dict):
        return bool(default)
    value = cur.get(keys[-1], default)
    if not isinstance(value, bool):
        raise ValueError(f"Config field {path} must be a boolean when provided.")
    return value


def _tag(
    cfg: Dict,
    path: str,
    config_path: Optional[str],
    infer_from_path,
) -> str:
    tag = cfg.get("data", {}).get(path)
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    if config_path:
        inferred = infer_from_path(config_path)
        if inferred is not None:
            return inferred
    raise KeyError(f"Missing required config field: data.{path}")


def _profile(cfg: Dict, path: str) -> Dict:
    profile = _require(cfg, path)
    if not isinstance(profile, dict) or not profile:
        raise ValueError(f"Config field {path} must be a non-empty object.")
    return deepcopy(profile)


def _load_json(config_path: str, validator) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    validator(cfg, config_path)
    return cfg


def graspdata_tag_from_path(config_path: str) -> str:
    stem = Path(config_path).stem
    if stem.startswith("run_"):
        return f"graspdata_{stem[len('run_'):]}"
    return stem


def _infer_objdata_tag_from_path(config_path: str) -> Optional[str]:
    stem = Path(config_path).stem
    if stem.startswith("assets_"):
        suffix = stem[len("assets_") :]
        if suffix:
            return f"objdata_{suffix}"
    if stem.startswith("run_"):
        parts = stem[len("run_") :].split("_")
        if parts and parts[0]:
            return f"objdata_{parts[0]}"
    return None


def objdata_tag_cfg(cfg: Dict, config_path: Optional[str] = None) -> str:
    return _tag(cfg, "objdata_tag", config_path, _infer_objdata_tag_from_path)


def graspdata_tag_cfg(cfg: Dict, config_path: Optional[str] = None) -> str:
    return _tag(cfg, "graspdata_tag", config_path, graspdata_tag_from_path)


def data_raw_dataset_name_cfg(cfg: Dict) -> str:
    return _require_str(cfg, "data.raw_dataset_name")


def data_raw_dataset_root_cfg(cfg: Dict) -> str:
    return _require_str(cfg, "data.raw_dataset_root")


def data_generated_dataset_root_cfg(cfg: Dict) -> str:
    return _require_str(cfg, "data.generated_dataset_root")


def data_verbose_cfg(cfg: Dict) -> bool:
    value = _require(cfg, "data.verbose")
    if not isinstance(value, bool):
        raise ValueError("Config field data.verbose must be a boolean.")
    return value


def data_asset_scales_cfg(cfg: Dict) -> List[float]:
    return _require_float_list(cfg, "data.asset_scales")


def data_run_scales_cfg(cfg: Dict) -> List[float]:
    return _require_float_list(cfg, "data.run_scales")


def data_build_native_asset_cfg(cfg: Dict) -> bool:
    return _bool(cfg, "data.build_native_asset", False)


def data_use_native_asset_cfg(cfg: Dict) -> bool:
    return _bool(cfg, "data.use_native_asset", False)


def usd_convert_cfg(cfg: Dict) -> Dict:
    usd_cfg = _require(cfg, "usd_convert")
    if not isinstance(usd_cfg, dict) or not usd_cfg:
        raise ValueError("Config field usd_convert must be a non-empty object.")

    normalized = {
        "backend": str(usd_cfg.get("backend", "urdf")).strip().lower(),
        "force": bool(usd_cfg.get("force", False)),
        "fix_base": bool(usd_cfg.get("fix_base", False)),
        "import_sites": bool(usd_cfg.get("import_sites", False)),
        "verify_inertial": bool(usd_cfg.get("verify_inertial", True)),
        "merge_joints": bool(usd_cfg.get("merge_joints", False)),
        "make_instanceable": bool(usd_cfg.get("make_instanceable", False)),
        "convex_decompose_mesh": bool(usd_cfg.get("convex_decompose_mesh", True)),
    }
    if normalized["backend"] not in {"urdf", "mjcf"}:
        raise ValueError("Config field usd_convert.backend must be one of: urdf, mjcf")

    for key in [
        "force",
        "fix_base",
        "import_sites",
        "verify_inertial",
        "merge_joints",
        "make_instanceable",
        "convex_decompose_mesh",
    ]:
        value = usd_cfg.get(key, normalized[key])
        if not isinstance(value, bool):
            raise ValueError(f"Config field usd_convert.{key} must be a boolean.")
    return normalized


def hand_profile_cfg(cfg: Dict) -> Dict:
    return _profile(cfg, "hand.profile")


def object_profile_cfg(cfg: Dict) -> Dict:
    return _profile(cfg, "profile_object")


def hand_anchor_params_cfg(cfg: Dict) -> Dict[str, Dict[str, float | str]]:
    params = _require(cfg, "hand.anchor_params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config field hand.anchor_params must be a non-empty object.")
    normalized: Dict[str, Dict[str, float | str]] = {}
    valid_axes = {"X", "-X", "Y", "-Y", "Z", "-Z"}
    for body_name, value in params.items():
        if not isinstance(body_name, str) or not body_name.strip():
            raise ValueError(
                "Each key in hand.anchor_params must be a non-empty body name string."
            )
        if isinstance(value, (int, float)):
            normalized[body_name] = {"weight": float(value), "axis": "Z"}
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"hand.anchor_params['{body_name}'] must be a number or object with weight/axis."
            )
        if "weight" not in value:
            raise KeyError(
                f"Missing required config field: hand.anchor_params['{body_name}'].weight"
            )
        if "axis" not in value:
            raise KeyError(
                f"Missing required config field: hand.anchor_params['{body_name}'].axis"
            )
        try:
            weight = float(value["weight"])
        except Exception as exc:
            raise ValueError(
                f"hand.anchor_params['{body_name}'].weight must be numeric."
            ) from exc
        axis = str(value["axis"]).strip().upper()
        if axis not in valid_axes:
            raise ValueError(
                f"hand.anchor_params['{body_name}'].axis must be one of {sorted(valid_axes)}."
            )
        normalized[body_name] = {"weight": weight, "axis": axis}
    return deepcopy(normalized)


def hand_root_stabilization_cfg(cfg: Dict) -> Optional[Dict[str, float | str]]:
    hand_cfg = _require(cfg, "hand")
    root_cfg = hand_cfg.get("root_stabilization")
    if root_cfg is None:
        return None
    if not isinstance(root_cfg, dict):
        raise ValueError("Config field hand.root_stabilization must be an object.")

    body_name = root_cfg.get("root_body_name")
    if not isinstance(body_name, str) or not body_name.strip():
        raise ValueError(
            "Config field hand.root_stabilization.root_body_name must be a non-empty string."
        )
    try:
        root_scale = float(root_cfg.get("root_scale"))
    except Exception as exc:
        raise ValueError(
            "Config field hand.root_stabilization.root_scale must be numeric."
        ) from exc
    if root_scale <= 0.0:
        raise ValueError("Config field hand.root_stabilization.root_scale must be > 0.")
    return {
        "root_body_name": body_name.strip(),
        "root_scale": float(root_scale),
    }


def ensure_parent_dir(filepath: str) -> None:
    Path(filepath).resolve().parent.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def resolve_object_asset_name(obj_info: Dict) -> str:
    """Derive a stable asset name from an object-info dict."""
    obj_name = str(obj_info.get("name") or "").strip()
    scale_tag_value = str(obj_info.get("scale_tag") or "").strip()
    if obj_name and scale_tag_value:
        return f"{obj_name}_{scale_tag_value}"

    xml_abs = str(obj_info.get("xml_abs") or "").strip()
    if xml_abs:
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(xml_abs)))
        if parent_name:
            if obj_name:
                return f"{obj_name}_{parent_name}"
            return parent_name
    return obj_name


def scale_tag(scale: float) -> str:
    """Encode a float scale value to a scale-tag string."""
    return f"scale{int(round(float(scale) * _SCALE_TAG_FACTOR)):03d}"


def native_tag() -> str:
    """Return the canonical scale-tag for native (unscaled) assets."""
    return "native"


def parse_scale_tag(tag: str) -> Optional[float]:
    """Decode a scale-tag string back to a float, or ``None`` for native."""
    tag = str(tag).strip()
    if tag == native_tag():
        return None
    if not tag.startswith("scale"):
        return None
    digits = tag[len("scale") :]
    if not digits.isdigit():
        return None
    return float(int(digits)) / _SCALE_TAG_FACTOR


def build_logs_dir(script: str, dataset_tag: str) -> Path:
    script_name = Path(script).stem or "run"
    return Path("logs") / safe_filename(script_name) / safe_filename(dataset_tag)


def relpath_str(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def list_existing_files(folder: Path, prefix: str) -> List[Path]:
    return sorted(path for path in folder.glob(f"{prefix}*.npy") if path.is_file())


def resolve_split_manifest_path_cfg(cfg: Dict, config_path: str, split: str) -> Path:
    dataset_root = Path(data_generated_dataset_root_cfg(cfg)).resolve()
    dataset_tag = graspdata_tag_cfg(cfg, config_path)
    return dataset_root / dataset_tag / f"{split}.json"


def _validate_anchor_params(cfg: Dict) -> None:
    _ = hand_anchor_params_cfg(cfg)


def _validate_contact_profile(profile: Dict, path: str, require_control: bool) -> None:
    if not isinstance(profile, dict) or not profile:
        raise ValueError(f"Config field {path} must be a non-empty object.")

    if require_control:
        ctrl_joint_indices = profile.get("ctrl_joint_indices")
        if not isinstance(ctrl_joint_indices, list) or not ctrl_joint_indices:
            raise ValueError(f"{path}.ctrl_joint_indices must be a non-empty list.")
        normalized_ctrl_joint_indices = []
        for idx, value in enumerate(ctrl_joint_indices):
            try:
                joint_idx = int(value)
            except Exception as exc:
                raise ValueError(
                    f"{path}.ctrl_joint_indices[{idx}] must be an integer."
                ) from exc
            if joint_idx < 0:
                raise ValueError(f"{path}.ctrl_joint_indices[{idx}] must be >= 0.")
            normalized_ctrl_joint_indices.append(joint_idx)
        if len(set(normalized_ctrl_joint_indices)) != len(
            normalized_ctrl_joint_indices
        ):
            raise ValueError(f"{path}.ctrl_joint_indices must not contain duplicates.")

        for field in ["side_swing_indices", "thumb_relax_indices"]:
            values = profile.get(field)
            if not isinstance(values, list):
                raise ValueError(f"{path}.{field} must be a list.")
            for idx, value in enumerate(values):
                try:
                    int(value)
                except Exception as exc:
                    raise ValueError(
                        f"{path}.{field}[{idx}] must be an integer."
                    ) from exc

        thumb_relax_divisor = float(profile.get("thumb_relax_divisor"))
        if thumb_relax_divisor <= 0.0:
            raise ValueError(f"{path}.thumb_relax_divisor must be > 0.")

    friction_coef = profile.get("friction_coef")
    if not isinstance(friction_coef, list) or len(friction_coef) not in {1, 2, 3}:
        raise ValueError(f"{path}.friction_coef must be a list with length 1, 2, or 3.")
    for idx, value in enumerate(friction_coef):
        try:
            float(value)
        except Exception as exc:
            raise ValueError(f"{path}.friction_coef[{idx}] must be numeric.") from exc

    for field, expected_len in [("solimp", 5), ("solref", 2)]:
        values = profile.get(field)
        if not isinstance(values, list) or len(values) != expected_len:
            raise ValueError(
                f"{path}.{field} must be a list with exactly {expected_len} values."
            )
        for idx, value in enumerate(values):
            try:
                float(value)
            except Exception as exc:
                raise ValueError(f"{path}.{field}[{idx}] must be numeric.") from exc

    for field in ["kp", "forcerange", "actuatorfrcrange"]:
        if field not in profile:
            continue
        try:
            value = float(profile[field])
        except Exception as exc:
            raise ValueError(f"{path}.{field} must be numeric when provided.") from exc
        if value <= 0.0:
            raise ValueError(f"{path}.{field} must be > 0 when provided.")


def _validate_hand_profile(cfg: Dict) -> None:
    _validate_contact_profile(
        _require(cfg, "hand.profile"), "hand.profile", require_control=True
    )


def _validate_object_profile(cfg: Dict) -> None:
    _validate_contact_profile(
        _require(cfg, "profile_object"), "profile_object", require_control=False
    )


def _validate_common_config(cfg: Dict, source_path: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a JSON object: {source_path}")
    _require(cfg, "seed")
    data_raw_dataset_name_cfg(cfg)
    data_raw_dataset_root_cfg(cfg)
    data_generated_dataset_root_cfg(cfg)
    data_verbose_cfg(cfg)


def _validate_asset_config(cfg: Dict, source_path: str) -> None:
    _validate_common_config(cfg, source_path)
    tag = _require_str(cfg, "data.objdata_tag")
    if not tag:
        raise ValueError("Config field data.objdata_tag must be a non-empty string.")
    scales = data_asset_scales_cfg(cfg)
    build_native = data_build_native_asset_cfg(cfg)
    if not scales and not build_native:
        raise ValueError(
            "Config must enable at least one asset scale or data.build_native_asset=true."
        )
    _require(cfg, "sampling.n_points")
    _require(cfg, "warp_render")


def _validate_run_config(cfg: Dict, source_path: str) -> None:
    _validate_common_config(cfg, source_path)
    for tag_field in ["objdata_tag", "graspdata_tag"]:
        _require_str(cfg, f"data.{tag_field}")

    scales = data_run_scales_cfg(cfg)
    use_native = data_use_native_asset_cfg(cfg)
    if not scales and not use_native:
        raise ValueError(
            "Config must enable at least one run scale or data.use_native_asset=true."
        )
    for key in [
        "n_points",
        "downsample_for_sim",
        "Nd",
        "rot_n",
        "d_min",
        "d_max",
        "pc_subdir",
    ]:
        _require(cfg, f"sampling.{key}")

    transform_cfg = _require(cfg, "hand.transform")
    if "pos" not in transform_cfg:
        raise KeyError("Missing required config field: hand.transform.pos")
    if "quat_wxyz" not in transform_cfg:
        raise KeyError("Missing required config field: hand.transform.quat_wxyz")
    pos = transform_cfg["pos"]
    quat_wxyz = transform_cfg["quat_wxyz"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError(
            "Config field hand.transform.pos must be a list with exactly 3 values."
        )
    if not isinstance(quat_wxyz, list) or len(quat_wxyz) != 4:
        raise ValueError(
            "Config field hand.transform.quat_wxyz must be a list with exactly 4 values."
        )
    for idx, value in enumerate(pos):
        try:
            float(value)
        except Exception as exc:
            raise ValueError(f"hand.transform.pos[{idx}] must be numeric.") from exc
    for idx, value in enumerate(quat_wxyz):
        try:
            float(value)
        except Exception as exc:
            raise ValueError(
                f"hand.transform.quat_wxyz[{idx}] must be numeric."
            ) from exc
    quat_sq_norm = sum(float(value) * float(value) for value in quat_wxyz)
    if quat_sq_norm <= 1e-24:
        raise ValueError("Config field hand.transform.quat_wxyz must be non-zero.")

    xml_path = _require_str(cfg, "hand.xml_path")
    abs_xml = os.path.abspath(xml_path)
    if not os.path.exists(abs_xml):
        raise FileNotFoundError(f"hand.xml_path not found: {xml_path} -> {abs_xml}")

    _require(cfg, "hand.prepared_joints")
    _require(cfg, "hand.approach_joints")
    _require(cfg, "hand.shift_local")
    hand_root_stabilization_cfg(cfg)
    _validate_hand_profile(cfg)
    _validate_object_profile(cfg)
    _validate_anchor_params(cfg)

    contact_min_count = int(_require(cfg, "sim_grasp.contact_min_count"))
    if contact_min_count <= 0:
        raise ValueError("sim_grasp.contact_min_count must be > 0.")
    target_point_method = int(_require(cfg, "sim_grasp.target_point_method"))
    if target_point_method not in {1, 2, 3}:
        raise ValueError("sim_grasp.target_point_method must be one of [1, 2, 3].")

    for key in [
        "max_cap",
        "max_time_sec",
        "h5_name",
        "npy_name",
        "fail_h5_name",
        "fail_npy_name",
        "flush_every",
        "fail_keep_ratio",
        "min_valid_count",
    ]:
        _require(cfg, f"data.{key}")

    max_time_sec = float(_require(cfg, "data.max_time_sec"))
    if max_time_sec <= 0.0:
        raise ValueError("data.max_time_sec must be > 0.")


def load_asset_config(config_path: str) -> Dict:
    return _load_json(config_path, _validate_asset_config)


def load_run_config(config_path: str) -> Dict:
    return _load_json(config_path, _validate_run_config)
