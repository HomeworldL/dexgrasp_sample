import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import trimesh

from src.dataset_objects import DatasetObjects
from src.mj_ho import RobotKinematics
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config
from utils.utils_vis import visualize_with_plotly, visualize_with_viser

QPOS_KEYS = ("qpos_init", "qpos_approach", "qpos_prepared", "qpos_grasp")


def _load_grasp_from_h5(h5_path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as hf:
        for key in QPOS_KEYS:
            if key not in hf:
                raise KeyError(f"Missing dataset '{key}' in {h5_path}")
            out[key] = np.asarray(hf[key][:], dtype=np.float32)
    return out


def _load_grasp_from_npy(npy_path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(npy_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported npy grasp format: {npy_path}")

    out: Dict[str, np.ndarray] = {}
    for key in QPOS_KEYS:
        if key not in payload:
            raise KeyError(f"Missing key '{key}' in {npy_path}")
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
    seen = set()
    dedup_candidates = []
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        dedup_candidates.append(p)

    for p in dedup_candidates:
        if p.name and p.exists():
            return p
    joined = "\n".join(f"- {p}" for p in dedup_candidates)
    raise FileNotFoundError(f"No grasp file found under {output_dir}. Checked:\n{joined}")


def _load_grasp_data(grasp_path: Path) -> Dict[str, np.ndarray]:
    suffix = grasp_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _load_grasp_from_h5(grasp_path)
    if suffix == ".npy":
        return _load_grasp_from_npy(grasp_path)
    raise ValueError(f"Unsupported grasp file: {grasp_path}")


def _parse_vis_ids(raw: Optional[str], total: int) -> List[int]:
    if total <= 0:
        return []
    if raw is None or raw.strip().lower() == "auto":
        picks = [0, total // 2, total - 1]
    else:
        picks = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            picks.append(int(token))

    normalized: List[int] = []
    for idx in picks:
        real_idx = total + idx if idx < 0 else idx
        if 0 <= real_idx < total and real_idx not in normalized:
            normalized.append(real_idx)
    if not normalized:
        raise ValueError(f"No valid vis ids in '{raw}', total={total}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize object mesh + hand grasps (viser) and all grasp frames (plotly).")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument("--grasp-path", type=str, default=None, help="Optional grasp file path (.h5/.npy).")
    parser.add_argument(
        "--vis-ids",
        type=str,
        default="auto",
        help="Comma separated indices for hand mesh visualization, e.g. '0,10,-1'. Default: auto (first/mid/last).",
    )
    parser.add_argument(
        "--frame-stage",
        type=str,
        default="qpos_grasp",
        choices=list(QPOS_KEYS),
        help="Which qpos stage to visualize as plotly coordinate frames.",
    )
    parser.add_argument(
        "--skip-plotly",
        action="store_true",
        help="Skip plotly frame visualization.",
    )
    parser.add_argument(
        "--plotly-html",
        type=str,
        default=None,
        help="If set, save plotly figure to this html path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = DatasetObjects(
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=dataset_tag_from_config(args.config),
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )

    if args.obj_key:
        info = ds.get_obj_info_by_scale_key(args.obj_key)
    else:
        obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
        info = ds.get_obj_info_by_index(obj_id)

    obj_name = info["object_name"]
    print(f"[vis_grasp] id={info['global_id']} name={obj_name} scale={info['scale']}")

    output_dir = Path(info["output_dir_abs"]).resolve()
    grasp_path = _resolve_grasp_path(output_dir, cfg, args.grasp_path)
    grasp_data = _load_grasp_data(grasp_path)
    total_grasps = int(grasp_data["qpos_grasp"].shape[0])
    if total_grasps == 0:
        raise RuntimeError(f"No grasps found in {grasp_path}")
    vis_ids = _parse_vis_ids(args.vis_ids, total_grasps)
    print(f"[vis_grasp] grasp={grasp_path} total={total_grasps} vis_ids={vis_ids}")

    rk = RobotKinematics(str(Path(cfg["hand"]["xml_path"]).resolve()))

    parts = [ds.load_mesh(path) for path in info["convex_parts_abs"]]
    meshes_for_vis: Dict[str, trimesh.Trimesh] = {
        "obj_coacd": ds.load_mesh(info["coacd_abs"]),
        "obj_convex_parts": trimesh.util.concatenate(parts),
    }

    for idx in vis_ids:
        for qkey in QPOS_KEYS:
            hand_mesh = rk.get_posed_meshes(grasp_data[qkey][idx], kind="visual")
            if hand_mesh is None:
                continue
            meshes_for_vis[f"hand_{idx}_{qkey}"] = hand_mesh

    pts, _ = ds.sample_surface_o3d(
        info["coacd_abs"],
        n_points=min(2048, int(cfg.get("sampling", {}).get("n_points", 2048))),
        method="poisson",
    )
    cols = np.tile(np.array([[90, 160, 255]], dtype=np.uint8), (pts.shape[0], 1))

    _server = visualize_with_viser(meshes=meshes_for_vis, pointclouds={"obj_pointcloud": (pts, cols)})
    print("[vis_grasp] Viser started. Press Ctrl+C to quit.")

    if not args.skip_plotly:
        frames = grasp_data[args.frame_stage][:, :7]
        fig = visualize_with_plotly(
            meshes={"obj_coacd": meshes_for_vis["obj_coacd"]},
            frames=frames,
            show=True,
        )
        if args.plotly_html:
            html_path = Path(args.plotly_html).expanduser().resolve()
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path))
            print(f"[vis_grasp] plotly html saved: {html_path}")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
