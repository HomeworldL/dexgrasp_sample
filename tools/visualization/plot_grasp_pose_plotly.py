import argparse
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from src.dataset_objects import DatasetObjects
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_vis import visualize_with_plotly

QPOS_KEYS = ("qpos_init", "qpos_approach", "qpos_prepared", "qpos_grasp")


def _load_grasp_from_h5(h5_path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as hf:
        for key in QPOS_KEYS:
            out[key] = np.asarray(hf[key][:], dtype=np.float32)
    return out


def _load_grasp_from_npy(npy_path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(npy_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()
    out: Dict[str, np.ndarray] = {}
    for key in QPOS_KEYS:
        out[key] = np.asarray(payload[key], dtype=np.float32)
    return out


def _resolve_grasp_path(output_dir: Path, cfg: Dict, grasp_path_arg: Optional[str]) -> Path:
    if grasp_path_arg:
        p = Path(grasp_path_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"grasp path not found: {p}")
        return p

    output_cfg = cfg.get("output", {})
    candidates = [
        output_dir / "grasp.h5",
        output_dir / str(output_cfg.get("h5_name", "")),
        output_dir / "grasp_data.h5",
        output_dir / str(output_cfg.get("npy_name", "")),
        output_dir / "grasp.npy",
        output_dir / "grasp_data.npy",
    ]
    for p in candidates:
        if p.name and p.exists():
            return p
    raise FileNotFoundError(f"No grasp file found under {output_dir}")


def _load_grasp_data(grasp_path: Path) -> Dict[str, np.ndarray]:
    if grasp_path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_grasp_from_h5(grasp_path)
    if grasp_path.suffix.lower() == ".npy":
        return _load_grasp_from_npy(grasp_path)
    raise ValueError(f"Unsupported grasp file: {grasp_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plotly visualization for all grasp frames.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument("--grasp-path", type=str, default=None)
    parser.add_argument(
        "--frame-stage",
        type=str,
        default="qpos_grasp",
        choices=list(QPOS_KEYS),
        help="Which qpos stage to visualize as coordinate frames.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_stem = Path(args.config).stem
    ds = DatasetObjects(
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=config_stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )

    if args.obj_key:
        info = ds.get_info(args.obj_key)
    else:
        obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
        info = ds.get_info(obj_id)

    grasp_path = _resolve_grasp_path(Path(info["output_dir_abs"]), cfg, args.grasp_path)
    grasp_data = _load_grasp_data(grasp_path)

    frames = grasp_data[args.frame_stage][:, :7]
    obj_mesh = ds.load_mesh(info["coacd_abs"])
    visualize_with_plotly(meshes={"obj": obj_mesh}, frames=frames, show=True)


if __name__ == "__main__":
    main()
