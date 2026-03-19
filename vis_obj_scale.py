import argparse
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from src.dataset_objects import DatasetObjects
from utils.utils_file import DEFAULT_RUN_CONFIG_PATH, dataset_tag_from_config, load_config
from utils.utils_vis import visualize_with_viser


def _resolve_anchor_info(ds: DatasetObjects, obj_id: int | None, obj_key: str | None) -> Dict:
    if obj_key:
        return ds.get_obj_info_by_scale_key(obj_key)
    if obj_id is not None:
        return ds.get_obj_info_by_index(int(obj_id))
    raise ValueError("vis_obj_scale requires either --obj-id or --obj-key.")


def _entries_for_same_object(ds: DatasetObjects, object_name: str) -> List[Dict]:
    entries = [item for item in ds.get_entries() if item["object_name"] == object_name]
    if not entries:
        raise RuntimeError(f"No object-scale entries found for object '{object_name}'.")
    return sorted(entries, key=lambda item: float(item["scale"]))


def _layout_offsets(meshes: List[trimesh.Trimesh], gap_ratio: float = 0.25) -> List[np.ndarray]:
    offsets: List[np.ndarray] = []
    cursor_x = 0.0
    previous_half_width = 0.0

    for idx, mesh in enumerate(meshes):
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        extents = np.maximum(bounds[1] - bounds[0], 1e-6)
        center = 0.5 * (bounds[0] + bounds[1])
        half_width = 0.5 * float(extents[0])
        gap = max(float(extents.max()) * gap_ratio, 0.02)

        if idx == 0:
            offset_x = -center[0]
        else:
            cursor_x += previous_half_width + half_width + gap
            offset_x = cursor_x - center[0]

        offsets.append(np.array([offset_x, -center[1], -bounds[0, 2]], dtype=np.float64))
        previous_half_width = half_width

    return offsets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all available scales for one object with non-overlapping viser meshes."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH)
    parser.add_argument("-i", "--obj-id", type=int, default=None)
    parser.add_argument(
        "-k",
        "--obj-key",
        type=str,
        default=None,
        help="Object-scale key used only to locate the target object, e.g. '002_master_chef_can__scale080'.",
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

    anchor_info = _resolve_anchor_info(ds, args.obj_id, args.obj_key)
    object_name = anchor_info["object_name"]
    entries = _entries_for_same_object(ds, object_name)

    base_meshes = [ds.load_mesh(entry["coacd_abs"]) for entry in entries]
    offsets = _layout_offsets(base_meshes)

    meshes_for_vis: Dict[str, trimesh.Trimesh] = {}
    for entry, mesh, offset in zip(entries, base_meshes, offsets):
        placed_mesh = mesh.copy()
        placed_mesh.apply_translation(offset)
        scale_tag = f"scale{int(round(float(entry['scale']) * 1000)):03d}"
        mesh_name = f"{object_name}/{scale_tag}"
        meshes_for_vis[mesh_name] = placed_mesh
        print(
            f"[vis_obj_scale] name={object_name} scale={entry['scale']:.3f} "
            f"offset=({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}) mesh={entry['coacd_abs']}"
        )

    _server = visualize_with_viser(
        meshes=meshes_for_vis,
        look_at=tuple(offsets[len(offsets) // 2].tolist()) if offsets else (0.0, 0.0, 0.0),
    )
    print("[vis_obj_scale] Viser started. Keep this process alive and open the printed local URL in your browser.")
    try:
        while True:
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[vis_obj_scale] Exiting.")


if __name__ == "__main__":
    main()
