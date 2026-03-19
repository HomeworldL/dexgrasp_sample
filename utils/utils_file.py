import json
import os
from pathlib import Path
from typing import Dict, List, Optional


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


def safe_filename(name: str) -> str:
    """Sanitize a name for log/output filenames.

    English: Used by multi-run entrypoints to convert object-scale keys or script
    names into filesystem-safe path fragments.
    中文：供并行运行脚本使用，把脚本名或 object-scale key 转成适合日志路径的安全文件名。
    """
    return "".join(c if c.isalnum() or c in "-_.()" else "_" for c in name)


def build_logs_dir(script: str, dataset_tag: str) -> Path:
    """Build logs/<script>/<dataset_tag> directory path.

    English: Shared by CPU/GPU multi-run scripts so their child-process logs are
    grouped by entry script and dataset tag.
    中文：供 CPU/GPU 两个 multi 脚本共用，把子进程日志统一归档到
    logs/<script>/<dataset_tag>/ 目录下。
    """
    script_name = Path(script).stem or "run"
    return Path("logs") / safe_filename(script_name) / safe_filename(dataset_tag)


def relpath_str(path: Path, start: Path) -> str:
    """Return POSIX-style relative path string for dataset manifests.

    English: Used when exporting dataset split json files so saved paths are
    relative to the dataset root and platform-independent.
    中文：用于导出数据集 split 清单，把路径写成相对 dataset 根目录的
    POSIX 字符串，避免平台分隔符差异。
    """
    return path.resolve().relative_to(start.resolve()).as_posix()


def list_existing_files(folder: Path, prefix: str) -> List[Path]:
    """List existing npy files with a given prefix in sorted order.

    English: Used by run_multi.py when validating rendered warp assets such as
    partial point clouds and camera extrinsics.
    中文：用于 run_multi.py 检查 warp 渲染产物，按前缀收集并排序已有的
    npy 文件，例如 partial_pc 和 cam_ex。
    """
    return sorted(p for p in folder.glob(f"{prefix}*.npy") if p.is_file())


def resolve_split_manifest_path(cfg: Dict, config_path: str, split: str) -> Path:
    dataset_root = Path(cfg.get("output", {}).get("dataset_root", "datasets")).resolve()
    dataset_tag = dataset_tag_from_config(config_path)
    return dataset_root / dataset_tag / f"{split}.json"


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

    for k in ["n_points", "downsample_for_sim", "Nd", "rot_n", "d_min", "d_max"]:
        _require(cfg, f"sampling.{k}")

    transform_cfg = _require(cfg, "hand.transform")
    if "base_rot_grasp_to_palm" not in transform_cfg:
        raise KeyError("Missing required config field: hand.transform.base_rot_grasp_to_palm")
    extra_euler = transform_cfg.get("extra_euler")
    if not isinstance(extra_euler, dict):
        raise KeyError("Missing required config field: hand.transform.extra_euler")
    if "axis" not in extra_euler:
        raise KeyError("Missing required config field: hand.transform.extra_euler.axis")
    if "degrees" not in extra_euler:
        raise KeyError("Missing required config field: hand.transform.extra_euler.degrees")

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

    contact_min_count = int(_require(cfg, "sim_grasp.contact_min_count"))
    if contact_min_count <= 0:
        raise ValueError("sim_grasp.contact_min_count must be > 0.")

    for k in ["max_cap", "max_time_sec", "h5_name", "npy_name"]:
        _require(cfg, f"output.{k}")

    max_time_sec = float(_require(cfg, "output.max_time_sec"))
    if max_time_sec <= 0.0:
        raise ValueError("output.max_time_sec must be > 0.")


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
