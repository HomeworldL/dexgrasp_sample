import json
import os
from pathlib import Path
from typing import Dict, Optional


DEFAULT_RUN_CONFIG_PATH = "configs/run_YCB_liberhand.json"


def dataset_tag_from_config(config_path: str) -> str:
    """Map config stem to dataset output tag.

    Convention:
    - run_YCB_liberhand -> graspdata_YCB_liberhand
    - non-run_ stems are kept as-is
    """
    stem = Path(config_path).stem
    if stem.startswith("run_"):
        return f"graspdata_{stem[len('run_'):]}"
    return stem


def ensure_dir_for_file(filepath: str):
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _require(cfg: Dict, path: str):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[key]
    return cur


def _validate_target_body_params(cfg: Dict) -> None:
    tbp = _require(cfg, "hand.target_body_params")
    if not isinstance(tbp, dict) or not tbp:
        raise ValueError("Config field hand.target_body_params must be a non-empty object.")
    for body_name, weights in tbp.items():
        if not isinstance(body_name, str) or not body_name:
            raise ValueError("Each key in hand.target_body_params must be a non-empty body name string.")
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError(
                f"hand.target_body_params['{body_name}'] must be [contact_weight, distance_weight]."
            )
        try:
            float(weights[0])
            float(weights[1])
        except Exception as exc:
            raise ValueError(
                f"hand.target_body_params['{body_name}'] values must be numeric."
            ) from exc


def _validate_config(cfg: Dict, source_path: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a JSON object: {source_path}")

    _require(cfg, "seed")

    include = _require(cfg, "dataset.include")
    if not isinstance(include, list) or not include:
        raise ValueError("Config field dataset.include must be a non-empty list.")

    root = _require(cfg, "dataset.root")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("Config field dataset.root must be a non-empty string.")

    verbose = _require(cfg, "dataset.verbose")
    if not isinstance(verbose, bool):
        raise ValueError("Config field dataset.verbose must be a boolean.")

    scales = _require(cfg, "dataset.scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Config field dataset.scales must be a non-empty list.")
    for s in scales:
        if float(s) <= 0:
            raise ValueError("All dataset.scales values must be > 0.")

    _require(cfg, "object.id")

    for k in ["n_points", "downsample_for_sim", "Nd", "rot_n", "d_min", "d_max"]:
        _require(cfg, f"sampling.{k}")

    _require(cfg, "transform.base_rot_grasp_to_palm")
    _require(cfg, "transform.extra_euler.axis")
    _require(cfg, "transform.extra_euler.degrees")

    xml_path = _require(cfg, "hand.xml_path")
    if not isinstance(xml_path, str) or not xml_path:
        raise ValueError("Config field hand.xml_path must be a non-empty string.")
    abs_xml = os.path.abspath(xml_path)
    if not os.path.exists(abs_xml):
        raise FileNotFoundError(f"hand.xml_path not found: {xml_path} -> {abs_xml}")

    _require(cfg, "hand.prepared_joints")
    _require(cfg, "hand.approach_joints")
    _require(cfg, "hand.shift_local")
    _validate_target_body_params(cfg)

    _require(cfg, "validation.contact_min_count")

    for k in ["base_dir", "max_cap", "h5_name", "npy_name"]:
        _require(cfg, f"output.{k}")


def load_config(path: Optional[str]) -> Dict:
    if not path:
        raise ValueError("Config path is required. Example: -c configs/run_YCB_liberhand.json")

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {path} -> {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _validate_config(cfg, abs_path)
    return cfg
