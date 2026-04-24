import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh
import viser

from demo_grasp import DEFAULT_RUN_CONFIG_PATH
from src.mj_ho import MjHO, RobotKinematics
from src.sample import downsample_fps
from utils.utils_file import (
    anchor_params_from_config,
    hand_profile_from_config,
    hand_root_stabilization_from_config,
    load_config,
    object_profile_from_config,
)
from utils.utils_pointcloud import sample_surface_o3d
from utils.utils_sample import (
    build_pose_candidates,
    encode_h5_str,
    load_global_pc_and_normals,
    make_qpos_triplets,
    parse_object_scale_key,
    sample_frames_from_points,
)
from utils.utils_seed import set_seed, stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one sim_grasp(record_history=True) case and inspect it in viser."
    )
    parser.add_argument("--object-scale-key", type=str, required=True, help="Unique object-scale key.")
    parser.add_argument("--coacd-path", type=str, required=True, help="Path to scaled COACD mesh OBJ.")
    parser.add_argument("--mjcf-path", type=str, required=True, help="Path to scaled object MJCF.")
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=None,
        help="Prepared object-scale asset directory. If set and pc_warp data exists, reuse it.",
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_RUN_CONFIG_PATH, help="JSON config path.")
    parser.add_argument(
        "--skip-candidates",
        type=int,
        default=0,
        help="Skip this many candidates after prepared/approach/init collision filtering.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="If > 0, stop searching after this many collision-free candidates were checked.",
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        default=None,
        help="Optional output path for the captured debug dump (.npz).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def _default_dump_path(object_scale_key: str, config_path: str) -> Path:
    cfg_stem = Path(config_path).stem
    safe_key = object_scale_key.replace("/", "_")
    return Path("tmp") / f"debug_viser_grasp_{cfg_stem}_{safe_key}.npz"


def _load_points_and_normals(cfg: Dict[str, Any], asset_dir: Optional[str], coacd_path: str) -> tuple[np.ndarray, np.ndarray]:
    pc_subdir = str(cfg["sampling"]["pc_subdir"])
    if asset_dir is not None:
        try:
            return load_global_pc_and_normals(asset_dir, pc_subdir)
        except Exception:
            pass
    n_points = int(cfg["sampling"]["n_points"])
    return sample_surface_o3d(coacd_path, n_points=n_points, method="poisson")


def _build_dump(
    cfg: Dict[str, Any],
    object_scale_key: str,
    hand_xml_path: str,
    object_mjcf_path: str,
    coacd_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    skip_candidates: int,
    max_candidates: int,
    verbose: bool,
) -> Dict[str, Any]:
    object_name, parsed_scale = parse_object_scale_key(object_scale_key)
    obj_info = {"name": object_name, "xml_abs": object_mjcf_path}
    anchor_params = anchor_params_from_config(cfg)
    hand_profile = hand_profile_from_config(cfg)
    object_profile = object_profile_from_config(cfg)
    root_stabilization = hand_root_stabilization_from_config(cfg)

    mjho = MjHO(
        obj_info,
        hand_xml_path,
        anchor_params=anchor_params,
        hand_profile=hand_profile,
        object_profile=object_profile,
        root_stabilization=root_stabilization,
    )
    pts_for_sim, norms_for_sim, _ = downsample_fps(
        points,
        normals,
        int(cfg["sampling"]["downsample_for_sim"]),
        seed=stable_seed(int(cfg["seed"]), object_scale_key, "downsample_for_sim"),
    )
    mjho._set_obj_pts_norms(pts_for_sim, norms_for_sim)

    set_seed(stable_seed(int(cfg["seed"]), object_scale_key, "sample_frames"))
    transforms_np = sample_frames_from_points(cfg, points, normals)
    pose = build_pose_candidates(cfg, transforms_np)
    qpos_init, qpos_approach, qpos_prepared = make_qpos_triplets(cfg, pose)

    sim_grasp_cfg = dict(cfg.get("sim_grasp", {}))
    sim_grasp_cfg.pop("visualize", None)
    sim_grasp_cfg.pop("contact_min_count", None)

    eligible_seen = 0
    checked_eligible = 0
    for i in range(qpos_prepared.shape[0]):
        mjho.set_hand_qpos(qpos_prepared[i])
        if mjho.is_contact():
            continue
        mjho.set_hand_qpos(qpos_approach[i])
        if mjho.is_contact():
            continue
        mjho.set_hand_qpos(qpos_init[i])
        if mjho.is_contact():
            continue

        if eligible_seen < int(skip_candidates):
            eligible_seen += 1
            continue

        checked_eligible += 1
        if int(max_candidates) > 0 and checked_eligible > int(max_candidates):
            break

        mjho.set_hand_qpos(qpos_prepared[i])
        qpos_grasp, history = mjho.sim_grasp(visualize=False, record_history=True, **sim_grasp_cfg)
        qpos_grasp = np.asarray(qpos_grasp, dtype=np.float32)

        anchor_positions = np.asarray(history.get("anchor_positions", []), dtype=np.float32)
        pts_target = np.asarray(history.get("pts_target", []), dtype=np.float32)
        pts_top_Mp = np.asarray(history.get("pts_top_Mp", []), dtype=np.float32)
        qpos_history = np.asarray(history.get("qpos", []), dtype=np.float32)
        v_anchors = np.asarray(history.get("v_anchors", []), dtype=np.float32)
        dq_per_anchor = np.asarray(history.get("dq_per_anchor", []), dtype=np.float32)
        jacobian_per_anchor = np.asarray(history.get("jacobian_per_anchor", []), dtype=np.float32)
        total_dq_hand = np.asarray(history.get("total_dq_hand", []), dtype=np.float32)
        actuated_dq = np.asarray(history.get("actuated_dq", []), dtype=np.float32)
        ctrl_history = np.asarray(history.get("ctrl", []), dtype=np.float32)

        contact_counts = []
        if qpos_history.size > 0:
            for qpos_step in qpos_history:
                mjho.set_hand_qpos(qpos_step)
                contact_counts.append(int(mjho.get_contact_num(obj_margin=0.0)))
        mjho.set_hand_qpos(qpos_grasp)
        final_contact_count = int(mjho.get_contact_num(obj_margin=0.0))

        dump = {
            "meta": {
                "config_path": cfg.get("__config_path__", ""),
                "object_scale_key": object_scale_key,
                "object_name": object_name,
                "scale": float(parsed_scale) if parsed_scale is not None else None,
                "candidate_index": int(i),
                "eligible_index": int(eligible_seen),
                "hand_xml_path": hand_xml_path,
                "object_mjcf_path": object_mjcf_path,
                "coacd_path": coacd_path,
                "anchor_body_names": list(anchor_params.keys()),
                "actuated_joint_indices": list(np.asarray(hand_profile["ctrl_joint_indices"], dtype=int).tolist()),
                "sim_grasp_cfg": dict(cfg["sim_grasp"]),
                "prepared_contact_count": 0,
                "final_contact_count": final_contact_count,
            },
            "points": np.asarray(points, dtype=np.float32),
            "normals": np.asarray(normals, dtype=np.float32),
            "qpos_init": np.asarray(qpos_init[i], dtype=np.float32),
            "qpos_approach": np.asarray(qpos_approach[i], dtype=np.float32),
            "qpos_prepared": np.asarray(qpos_prepared[i], dtype=np.float32),
            "qpos_grasp": np.asarray(qpos_grasp, dtype=np.float32),
            "qpos_history": qpos_history,
            "anchor_positions": anchor_positions,
            "pts_target": pts_target,
            "pts_top_Mp": pts_top_Mp,
            "v_anchors": v_anchors,
            "dq_per_anchor": dq_per_anchor,
            "jacobian_per_anchor": jacobian_per_anchor,
            "total_dq_hand": total_dq_hand,
            "actuated_dq": actuated_dq,
            "ctrl_history": ctrl_history,
            "contact_counts": np.asarray(contact_counts, dtype=np.int32),
        }
        if verbose:
            print(
                f"[{object_scale_key}] selected candidate={i} eligible_index={eligible_seen} "
                f"history_steps={len(qpos_history)} final_contact_count={final_contact_count}"
            )
        return dump

    raise RuntimeError(
        f"No collision-free candidate found after skip={skip_candidates}, max_candidates={max_candidates}."
    )


def _save_dump(path: Path, dump: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps(dump["meta"], ensure_ascii=False), dtype=object),
        points=dump["points"],
        normals=dump["normals"],
        qpos_init=dump["qpos_init"],
        qpos_approach=dump["qpos_approach"],
        qpos_prepared=dump["qpos_prepared"],
        qpos_grasp=dump["qpos_grasp"],
        qpos_history=dump["qpos_history"],
        anchor_positions=dump["anchor_positions"],
        pts_target=dump["pts_target"],
        pts_top_Mp=dump["pts_top_Mp"],
        v_anchors=dump["v_anchors"],
        dq_per_anchor=dump["dq_per_anchor"],
        jacobian_per_anchor=dump["jacobian_per_anchor"],
        total_dq_hand=dump["total_dq_hand"],
        actuated_dq=dump["actuated_dq"],
        ctrl_history=dump["ctrl_history"],
        contact_counts=dump["contact_counts"],
    )


def _mesh_with_tint(mesh: Optional[trimesh.Trimesh], rgba: tuple[int, int, int, int]) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return None
    out = mesh.copy()
    out.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (out.vertices.shape[0], 1))
    return out


def _build_history_point_cloud(points_by_step: np.ndarray, start_rgb: np.ndarray, end_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_by_step.ndim != 3 or points_by_step.shape[0] <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    n_steps, n_points, _ = points_by_step.shape
    flat_points = points_by_step.reshape(-1, 3).astype(np.float32, copy=False)
    if n_steps == 1:
        colors = np.tile(start_rgb.reshape(1, 3), (flat_points.shape[0], 1))
        return flat_points, colors.astype(np.float32, copy=False)
    step_alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32).reshape(n_steps, 1)
    step_colors = (1.0 - step_alphas) * start_rgb.reshape(1, 3) + step_alphas * end_rgb.reshape(1, 3)
    colors = np.repeat(step_colors, n_points, axis=0)
    return flat_points, colors.astype(np.float32, copy=False)


def _update_dynamic_scene(
    server: viser.ViserServer,
    rk: RobotKinematics,
    qpos_history: np.ndarray,
    contact_counts: np.ndarray,
    total_dq_hand: np.ndarray,
    actuated_dq: np.ndarray,
    ctrl_history: np.ndarray,
    v_anchors: np.ndarray,
    step_idx: int,
    handles: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    max_step = int(qpos_history.shape[0]) - 1
    step_idx = int(np.clip(step_idx, 0, max_step))

    for key in ("hand_step",):
        handle = handles.get(key)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    hand_mesh = rk.get_posed_meshes(qpos_history[step_idx], kind="visual")
    if hand_mesh is not None:
        handles["hand_step"] = server.scene.add_mesh_trimesh(
            "/hand_step",
            mesh=_mesh_with_tint(hand_mesh, (240, 120, 60, 255)),
        )

    handles["status_markdown"].content = (
        f"Step `{step_idx + 1}/{max_step + 1}`  \n"
        f"Contact count: `{int(contact_counts[step_idx]) if step_idx < len(contact_counts) else 'n/a'}`  \n"
        f"Actuated joint indices: `{meta['actuated_joint_indices']}`  \n"
        f"total_dq_hand: `{np.array2string(total_dq_hand[step_idx], precision=4, suppress_small=True)}`  \n"
        f"actuated_dq: `{np.array2string(actuated_dq[step_idx], precision=4, suppress_small=True)}`  \n"
        f"ctrl: `{np.array2string(ctrl_history[step_idx], precision=4, suppress_small=True)}`  \n"
        f"v_anchors: `{np.array2string(v_anchors[step_idx], precision=4, suppress_small=True)}`"
    )


def _launch_viser(dump: Dict[str, Any]) -> None:
    meta = dump["meta"]
    hand_xml_path = str(meta["hand_xml_path"])
    coacd_path = str(meta["coacd_path"])
    qpos_prepared = np.asarray(dump["qpos_prepared"], dtype=np.float32)
    qpos_history = np.asarray(dump["qpos_history"], dtype=np.float32)
    anchor_positions = np.asarray(dump["anchor_positions"], dtype=np.float32)
    pts_target = np.asarray(dump["pts_target"], dtype=np.float32)
    contact_counts = np.asarray(dump["contact_counts"], dtype=np.int32)
    total_dq_hand = np.asarray(dump["total_dq_hand"], dtype=np.float32)
    actuated_dq = np.asarray(dump["actuated_dq"], dtype=np.float32)
    ctrl_history = np.asarray(dump["ctrl_history"], dtype=np.float32)
    v_anchors = np.asarray(dump["v_anchors"], dtype=np.float32)

    if qpos_history.ndim != 2 or qpos_history.shape[0] <= 0:
        raise RuntimeError("qpos_history is empty; sim_grasp did not record any steps.")

    rk = RobotKinematics(hand_xml_path)
    object_mesh = trimesh.load_mesh(coacd_path, process=False)
    prepared_mesh = rk.get_posed_meshes(qpos_prepared, kind="visual")

    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 0.0, 1.0])
    server.scene.add_mesh_trimesh("/object", mesh=_mesh_with_tint(object_mesh, (170, 170, 190, 255)))
    if prepared_mesh is not None:
        server.scene.add_mesh_trimesh(
            "/hand_prepared",
            mesh=_mesh_with_tint(prepared_mesh, (80, 140, 240, 255)),
        )

    anchor_hist_points, anchor_hist_colors = _build_history_point_cloud(
        anchor_positions,
        start_rgb=np.array([0.2, 0.95, 0.35], dtype=np.float32),
        end_rgb=np.array([0.0, 0.35, 0.0], dtype=np.float32),
    )
    target_hist_points, target_hist_colors = _build_history_point_cloud(
        pts_target,
        start_rgb=np.array([1.0, 0.85, 0.2], dtype=np.float32),
        end_rgb=np.array([0.85, 0.0, 0.0], dtype=np.float32),
    )
    if anchor_hist_points.shape[0] > 0:
        server.scene.add_point_cloud(
            name="/anchor_history",
            points=anchor_hist_points,
            colors=anchor_hist_colors,
            point_size=0.001,
            point_shape="circle",
        )
    if target_hist_points.shape[0] > 0:
        server.scene.add_point_cloud(
            name="/target_history",
            points=target_hist_points,
            colors=target_hist_colors,
            point_size=0.001,
            point_shape="circle",
        )

    info = server.gui.add_markdown(
        (
            f"Object: `{meta['object_scale_key']}`  \n"
            f"Candidate: `{meta['candidate_index']}`  \n"
            f"Anchors: `{', '.join(meta['anchor_body_names'])}`  \n"
            f"Final contact count: `{meta['final_contact_count']}`"
        )
    )
    _ = info
    step_slider = server.gui.add_slider(
        "step",
        min=0,
        max=int(qpos_history.shape[0] - 1),
        step=1,
        initial_value=0,
    )
    play_checkbox = server.gui.add_checkbox("play", initial_value=False)
    status_markdown = server.gui.add_markdown("")

    handles: Dict[str, Any] = {"status_markdown": status_markdown}

    def refresh() -> None:
        _update_dynamic_scene(
            server=server,
            rk=rk,
            qpos_history=qpos_history,
            contact_counts=contact_counts,
            total_dq_hand=total_dq_hand,
            actuated_dq=actuated_dq,
            ctrl_history=ctrl_history,
            v_anchors=v_anchors,
            step_idx=int(step_slider.value),
            handles=handles,
            meta=meta,
        )

    @step_slider.on_update
    def _(_: Any) -> None:
        refresh()

    step_button = server.gui.add_button("next")

    @step_button.on_click
    def _(_: Any) -> None:
        step_slider.value = int((int(step_slider.value) + 1) % int(qpos_history.shape[0]))

    refresh()
    print("viser debug viewer is running. Press Ctrl-C to stop.")
    try:
        while True:
            if bool(play_checkbox.value):
                step_slider.value = int((int(step_slider.value) + 1) % int(qpos_history.shape[0]))
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("viser debug viewer stopped.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["__config_path__"] = str(Path(args.config).resolve())
    set_seed(int(cfg["seed"]))

    verbose = bool(args.verbose)
    hand_xml_path = os.path.abspath(cfg["hand"]["xml_path"])
    asset_dir = args.asset_dir or str(Path(args.coacd_path).resolve().parent)
    points, normals = _load_points_and_normals(cfg, asset_dir, args.coacd_path)

    dump = _build_dump(
        cfg=cfg,
        object_scale_key=args.object_scale_key,
        hand_xml_path=hand_xml_path,
        object_mjcf_path=args.mjcf_path,
        coacd_path=args.coacd_path,
        points=points,
        normals=normals,
        skip_candidates=int(args.skip_candidates),
        max_candidates=int(args.max_candidates),
        verbose=verbose,
    )

    dump_path = Path(args.dump_path) if args.dump_path else _default_dump_path(args.object_scale_key, args.config)
    _save_dump(dump_path, dump)
    print(f"debug dump saved to {dump_path.resolve()}")
    _launch_viser(dump)


if __name__ == "__main__":
    main()
