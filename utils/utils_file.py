import json
import os
from typing import Dict, Optional


DEFAULT_RUN_CONFIG_PATH = "configs/run_YCB_liberhand.json"


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


def _validate_mjx_optional(cfg: Dict) -> None:
    if "mjx" not in cfg:
        return
    mjx_cfg = cfg["mjx"]
    if not isinstance(mjx_cfg, dict):
        raise ValueError("Config field mjx must be an object when provided.")

    if "impl" in mjx_cfg and not isinstance(mjx_cfg["impl"], str):
        raise ValueError("Config field mjx.impl must be a string.")
    if "device" in mjx_cfg and mjx_cfg["device"] is not None and not isinstance(mjx_cfg["device"], str):
        raise ValueError("Config field mjx.device must be null or a string.")
    if "naconmax" in mjx_cfg and mjx_cfg["naconmax"] is not None:
        if int(mjx_cfg["naconmax"]) <= 0:
            raise ValueError("Config field mjx.naconmax must be > 0 when provided.")

    batch_cfg = mjx_cfg.get("batch", {})
    if batch_cfg:
        if not isinstance(batch_cfg, dict):
            raise ValueError("Config field mjx.batch must be an object.")
        for key in [
            "precheck_batch_size",
            "precheck_inner_chunk_size",
            "sim_grasp_batch_size",
            "extforce_batch_size",
        ]:
            if key in batch_cfg and int(batch_cfg[key]) <= 0:
                raise ValueError(f"Config field mjx.batch.{key} must be > 0.")

    tail_cfg = mjx_cfg.get("tail", {})
    if tail_cfg:
        if not isinstance(tail_cfg, dict):
            raise ValueError("Config field mjx.tail must be an object.")
        if "drop_tail" in tail_cfg and not isinstance(tail_cfg["drop_tail"], bool):
            raise ValueError("Config field mjx.tail.drop_tail must be a boolean.")


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

    for k in ["n_points", "downsample_for_sim", "Nd", "d_min", "d_max"]:
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

    _validate_mjx_optional(cfg)


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
