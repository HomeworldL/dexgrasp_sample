import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.dataset_objects import DatasetObjects
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, load_config
from utils.utils_vis import generate_ncolors, visualize_with_viser


def _parse_indices(raw: Optional[str], total: int) -> List[int]:
    if total <= 0:
        return []
    if raw is None or raw.strip().lower() == "all":
        return list(range(total))

    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            idx = total + idx
        if 0 <= idx < total and idx not in out:
            out.append(idx)
    return out


def _cam_ex_to_wxyz_pose(cam_ex: np.ndarray) -> np.ndarray:
    mat = np.asarray(cam_ex, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"cam_ex must be 4x4, got {mat.shape}")
    pos = mat[:3, 3]
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)
    return np.concatenate([pos, quat_wxyz], axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved Warp partial point clouds for one object-scale entry.")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument("-k", "--obj-key", type=str, default=None)
    parser.add_argument(
        "--pc-subdir",
        type=str,
        default=None,
        help="Override pointcloud subdir; default from config warp_render.output_subdir.",
    )
    parser.add_argument(
        "--view-ids",
        type=str,
        default="all",
        help="Comma-separated view ids to display, e.g. '0,1,3' or '-1'. Default: all.",
    )
    parser.add_argument(
        "--hide-mesh",
        action="store_true",
        help="Do not display object mesh.",
    )
    parser.add_argument(
        "--show-cam-frames",
        action="store_true",
        help="Display saved camera frames from cam_ex_XX.npy.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = DatasetObjects(
        dataset_root=cfg["dataset"]["root"],
        dataset_names=list(cfg["dataset"].get("include", [])),
        scales=list(cfg["dataset"].get("scales", [])),
        dataset_tag=Path(args.config).stem,
        dataset_output_root=cfg.get("output", {}).get("dataset_root", "datasets"),
        verbose=bool(cfg["dataset"].get("verbose", False)),
    )

    if args.obj_key:
        info = ds.get_info(args.obj_key)
    else:
        obj_id = int(args.obj_id) if args.obj_id is not None else int(cfg.get("object", {}).get("id", 0))
        info = ds.get_info(obj_id)

    subdir = args.pc_subdir or cfg.get("warp_render", {}).get("output_subdir", "partial_pc_warp")
    pc_dir = Path(info["output_dir_abs"]).resolve() / subdir
    if not pc_dir.exists():
        raise FileNotFoundError(f"Pointcloud directory not found: {pc_dir}")

    pc_files = sorted(pc_dir.glob("partial_pc_*.npy"))
    if not pc_files:
        raise FileNotFoundError(f"No partial_pc_*.npy found under {pc_dir}")

    selected = _parse_indices(args.view_ids, len(pc_files))
    if not selected:
        raise ValueError(f"No valid view ids selected from '{args.view_ids}'")

    colors = generate_ncolors(len(selected))
    pointclouds: Dict[str, tuple] = {}
    for i, vid in enumerate(selected):
        pts = np.asarray(np.load(pc_files[vid], allow_pickle=True), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape in {pc_files[vid]}: {pts.shape}")
        cols = np.tile(colors[i : i + 1], (pts.shape[0], 1))
        pointclouds[f"partial_pc_{vid:02d}"] = (pts, cols)

    meshes = {}
    if not args.hide_mesh:
        meshes["obj_coacd"] = ds.load_mesh(info["coacd_abs"])

    frames = None
    if args.show_cam_frames:
        frame_list = []
        for vid in selected:
            cam_path = pc_dir / f"cam_ex_{vid:02d}.npy"
            if cam_path.exists():
                frame_list.append(_cam_ex_to_wxyz_pose(np.load(cam_path, allow_pickle=True)))
        if frame_list:
            frames = np.stack(frame_list, axis=0)

    print(
        f"[vis_partial_pc] id={info['global_id']} key={info['object_scale_key']} "
        f"pc_dir={pc_dir} views={selected}"
    )
    visualize_with_viser(meshes=meshes, pointclouds=pointclouds, frames=frames, blocking=True)


if __name__ == "__main__":
    main()
