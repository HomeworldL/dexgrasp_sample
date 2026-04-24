import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RUN_CONFIG_PATH = "configs/run_YCB_liberhand_right.json"
DEFAULT_ASSET_CONFIG_PATH = "configs/assets_YCB.json"


def dataset_tag_from_config(config_path: str) -> str:
    stem = Path(config_path).stem
    if stem.startswith("run_"):
        return f"graspdata_{stem[len('run_'):]}"
    return stem


def objdata_tag_from_config(cfg: Dict, config_path: Optional[str] = None) -> str:
    tag = cfg.get("data", {}).get("objdata_tag")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    if config_path:
        stem = Path(config_path).stem
        if stem.startswith("assets_"):
            suffix = stem[len("assets_") :]
            if suffix:
                return f"objdata_{suffix}"
        if stem.startswith("run_"):
            parts = stem[len("run_") :].split("_")
            if parts and parts[0]:
                return f"objdata_{parts[0]}"
    raise KeyError("Missing required config field: data.objdata_tag")


def graspdata_tag_from_config(cfg: Dict, config_path: Optional[str] = None) -> str:
    tag = cfg.get("data", {}).get("graspdata_tag")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    if config_path:
        return dataset_tag_from_config(config_path)
    raise KeyError("Missing required config field: data.graspdata_tag")


def raw_dataset_name_from_config(cfg: Dict) -> str:
    name = _require(cfg, "data.raw_dataset_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Config field data.raw_dataset_name must be a non-empty string.")
    return name.strip()


def raw_dataset_root_from_config(cfg: Dict) -> str:
    root = _require(cfg, "data.raw_dataset_root")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("Config field data.raw_dataset_root must be a non-empty string.")
    return root.strip()


def generated_dataset_root_from_config(cfg: Dict) -> str:
    root = _require(cfg, "data.generated_dataset_root")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("Config field data.generated_dataset_root must be a non-empty string.")
    return root.strip()


def data_verbose_from_config(cfg: Dict) -> bool:
    verbose = _require(cfg, "data.verbose")
    if not isinstance(verbose, bool):
        raise ValueError("Config field data.verbose must be a boolean.")
    return verbose


def asset_scales_from_config(cfg: Dict) -> List[float]:
    scales = _require(cfg, "data.asset_scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Config field data.asset_scales must be a non-empty list.")
    parsed = [float(scale) for scale in scales]
    for scale in parsed:
        if scale <= 0.0:
            raise ValueError("All data.asset_scales values must be > 0.")
    return parsed


def run_scales_from_config(cfg: Dict) -> List[float]:
    scales = _require(cfg, "data.scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Config field data.scales must be a non-empty list.")
    parsed = [float(scale) for scale in scales]
    for scale in parsed:
        if scale <= 0.0:
            raise ValueError("All data.scales values must be > 0.")
    return parsed


def hand_profile_from_config(cfg: Dict) -> Dict:
    profile = _require(cfg, "hand.profile")
    if not isinstance(profile, dict) or not profile:
        raise ValueError("Config field hand.profile must be a non-empty object.")
    return deepcopy(profile)


def object_profile_from_config(cfg: Dict) -> Dict:
    profile = _require(cfg, "profile_object")
    if not isinstance(profile, dict) or not profile:
        raise ValueError("Config field profile_object must be a non-empty object.")
    return deepcopy(profile)


def anchor_params_from_config(cfg: Dict) -> Dict[str, Dict[str, float | str]]:
    params = _require(cfg, "hand.anchor_params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config field hand.anchor_params must be a non-empty object.")
    normalized: Dict[str, Dict[str, float | str]] = {}
    valid_axes = {"X", "-X", "Y", "-Y", "Z", "-Z"}
    for body_name, value in params.items():
        if not isinstance(body_name, str) or not body_name.strip():
            raise ValueError("Each key in hand.anchor_params must be a non-empty body name string.")
        # Backward compatibility: allow numeric value as shorthand for {"weight": value, "axis": "Z"}.
        if isinstance(value, (int, float)):
            normalized[body_name] = {"weight": float(value), "axis": "Z"}
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"hand.anchor_params['{body_name}'] must be a number or object with weight/axis."
            )
        if "weight" not in value:
            raise KeyError(f"Missing required config field: hand.anchor_params['{body_name}'].weight")
        if "axis" not in value:
            raise KeyError(f"Missing required config field: hand.anchor_params['{body_name}'].axis")
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


def hand_root_stabilization_from_config(cfg: Dict) -> Optional[Dict[str, float | str]]:
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
        raise ValueError(
            "Config field hand.root_stabilization.root_scale must be > 0."
        )
    return {
        "root_body_name": body_name.strip(),
        "root_scale": float(root_scale),
    }


def ensure_dir_for_file(filepath: str):
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def build_logs_dir(script: str, dataset_tag: str) -> Path:
    script_name = Path(script).stem or "run"
    return Path("logs") / safe_filename(script_name) / safe_filename(dataset_tag)


def relpath_str(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def list_existing_files(folder: Path, prefix: str) -> List[Path]:
    return sorted(p for p in folder.glob(f"{prefix}*.npy") if p.is_file())


def resolve_split_manifest_path(cfg: Dict, config_path: str, split: str) -> Path:
    dataset_root = Path(generated_dataset_root_from_config(cfg)).resolve()
    dataset_tag = graspdata_tag_from_config(cfg, config_path)
    return dataset_root / dataset_tag / f"{split}.json"


def _require(cfg: Dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[key]
    return cur


def _validate_anchor_params(cfg: Dict) -> None:
    _ = anchor_params_from_config(cfg)


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
                raise ValueError(f"{path}.ctrl_joint_indices[{idx}] must be an integer.") from exc
            if joint_idx < 0:
                raise ValueError(f"{path}.ctrl_joint_indices[{idx}] must be >= 0.")
            normalized_ctrl_joint_indices.append(joint_idx)
        if len(set(normalized_ctrl_joint_indices)) != len(normalized_ctrl_joint_indices):
            raise ValueError(f"{path}.ctrl_joint_indices must not contain duplicates.")

        for field in ["side_swing_indices", "thumb_relax_indices"]:
            values = profile.get(field)
            if not isinstance(values, list):
                raise ValueError(f"{path}.{field} must be a list.")
            for idx, value in enumerate(values):
                try:
                    int(value)
                except Exception as exc:
                    raise ValueError(f"{path}.{field}[{idx}] must be an integer.") from exc

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
            raise ValueError(f"{path}.{field} must be a list with exactly {expected_len} values.")
        for idx, value in enumerate(values):
            try:
                float(value)
            except Exception as exc:
                raise ValueError(f"{path}.{field}[{idx}] must be numeric.") from exc

    # Optional runtime actuation overrides for MjHO hand setup.
    optional_positive_scalars = ["kp", "forcerange", "actuatorfrcrange"]
    for field in optional_positive_scalars:
        if field not in profile:
            continue
        try:
            value = float(profile[field])
        except Exception as exc:
            raise ValueError(f"{path}.{field} must be numeric when provided.") from exc
        if value <= 0.0:
            raise ValueError(f"{path}.{field} must be > 0 when provided.")


def _validate_hand_profile(cfg: Dict) -> None:
    _validate_contact_profile(_require(cfg, "hand.profile"), "hand.profile", require_control=True)


def _validate_object_profile(cfg: Dict) -> None:
    _validate_contact_profile(_require(cfg, "profile_object"), "profile_object", require_control=False)


def _validate_common_config(cfg: Dict, source_path: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a JSON object: {source_path}")
    _require(cfg, "seed")
    raw_dataset_name_from_config(cfg)
    raw_dataset_root_from_config(cfg)
    generated_dataset_root_from_config(cfg)
    data_verbose_from_config(cfg)


def _validate_asset_config(cfg: Dict, source_path: str) -> None:
    _validate_common_config(cfg, source_path)
    tag = _require(cfg, "data.objdata_tag")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("Config field data.objdata_tag must be a non-empty string.")
    asset_scales_from_config(cfg)
    _require(cfg, "sampling.n_points")
    _require(cfg, "warp_render")


def _validate_run_config(cfg: Dict, source_path: str) -> None:
    _validate_common_config(cfg, source_path)
    for tag_field in ["objdata_tag", "graspdata_tag"]:
        tag = _require(cfg, f"data.{tag_field}")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Config field data.{tag_field} must be a non-empty string.")
    run_scales_from_config(cfg)
    for k in ["n_points", "downsample_for_sim", "Nd", "rot_n", "d_min", "d_max", "pc_subdir"]:
        _require(cfg, f"sampling.{k}")

    transform_cfg = _require(cfg, "hand.transform")
    if "pos" not in transform_cfg:
        raise KeyError("Missing required config field: hand.transform.pos")
    if "quat_wxyz" not in transform_cfg:
        raise KeyError("Missing required config field: hand.transform.quat_wxyz")
    pos = transform_cfg["pos"]
    quat_wxyz = transform_cfg["quat_wxyz"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError("Config field hand.transform.pos must be a list with exactly 3 values.")
    if not isinstance(quat_wxyz, list) or len(quat_wxyz) != 4:
        raise ValueError("Config field hand.transform.quat_wxyz must be a list with exactly 4 values.")
    for idx, value in enumerate(pos):
        try:
            float(value)
        except Exception as exc:
            raise ValueError(f"hand.transform.pos[{idx}] must be numeric.") from exc
    for idx, value in enumerate(quat_wxyz):
        try:
            float(value)
        except Exception as exc:
            raise ValueError(f"hand.transform.quat_wxyz[{idx}] must be numeric.") from exc
    quat_sq_norm = sum(float(v) * float(v) for v in quat_wxyz)
    if quat_sq_norm <= 1e-24:
        raise ValueError("Config field hand.transform.quat_wxyz must be non-zero.")

    xml_path = _require(cfg, "hand.xml_path")
    if not isinstance(xml_path, str) or not xml_path:
        raise ValueError("Config field hand.xml_path must be a non-empty string.")
    abs_xml = os.path.abspath(xml_path)
    if not os.path.exists(abs_xml):
        raise FileNotFoundError(f"hand.xml_path not found: {xml_path} -> {abs_xml}")

    _require(cfg, "hand.prepared_joints")
    _require(cfg, "hand.approach_joints")
    _require(cfg, "hand.shift_local")
    hand_root_stabilization_from_config(cfg)
    _validate_hand_profile(cfg)
    _validate_object_profile(cfg)
    _validate_anchor_params(cfg)

    contact_min_count = int(_require(cfg, "sim_grasp.contact_min_count"))
    if contact_min_count <= 0:
        raise ValueError("sim_grasp.contact_min_count must be > 0.")
    target_point_method = int(_require(cfg, "sim_grasp.target_point_method"))
    if target_point_method not in {1, 2, 3}:
        raise ValueError("sim_grasp.target_point_method must be one of [1, 2, 3].")

    for k in [
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
        _require(cfg, f"data.{k}")

    max_time_sec = float(_require(cfg, "data.max_time_sec"))
    if max_time_sec <= 0.0:
        raise ValueError("data.max_time_sec must be > 0.")


def _load_json_config(path: Optional[str]) -> tuple[str, Dict]:
    if not path:
        raise ValueError("Config path is required. Example: -c configs/run_YCB_liberhand_right.json")
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {path} -> {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return abs_path, cfg


def load_asset_config(path: Optional[str]) -> Dict:
    abs_path, cfg = _load_json_config(path)
    _validate_asset_config(cfg, abs_path)
    return cfg


def load_run_config(path: Optional[str]) -> Dict:
    abs_path, cfg = _load_json_config(path)
    _validate_run_config(cfg, abs_path)
    return cfg


def load_config(path: Optional[str]) -> Dict:
    return load_run_config(path)
