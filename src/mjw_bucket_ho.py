from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import warp as wp
from scipy.spatial import cKDTree

try:
    import mujoco_warp as mjw
except Exception as exc:  # pragma: no cover - depends on runtime install.
    raise RuntimeError("mujoco_warp is required for src/mjw_bucket_ho.py.") from exc

from src.mjw_ho import MjWarpHO
from utils.utils_file import resolve_object_asset_name


@dataclass(frozen=True)
class BucketSlot:
    slot_id: int
    object_scale_key: str
    object_name: str
    scale: Optional[float]
    scale_tag: str
    prefix: str
    asset_name: str
    root_body_name: str
    root_body_id: int
    body_ids: frozenset[int]
    origin: np.ndarray
    freejoint_qposadr: Optional[int]


class MjWarpBucketHO(MjWarpHO):
    """MJWarp helper for a fixed bucket of object slots in one compiled model."""

    def __init__(
        self,
        object_infos: Sequence[Dict],
        hand_xml_path: str,
        anchor_params: Optional[Dict] = None,
        hand_profile: Optional[Dict] = None,
        object_profile: Optional[Dict] = None,
        root_stabilization: Optional[Dict] = None,
        object_fixed: bool = True,
        nworld: int = 1,
        device: str = "cuda:0",
        nconmax: int = 512,
        naconmax: int = 512,
        njmax: Optional[int] = None,
        njmax_nnz: Optional[int] = None,
        ccd_iterations: int = 200,
        slot_spacing: float = 5.0,
    ):
        if int(nworld) <= 0:
            raise ValueError(f"nworld must be positive, got {nworld}.")
        if not object_infos:
            raise ValueError("object_infos must contain at least one object.")

        wp.init()
        wp.set_device(device)

        self.object_infos = [dict(info) for info in object_infos]
        self.obj_info = {"name": "bucket", "xml_abs": ""}
        self.hand_xml_path = os.path.abspath(hand_xml_path)
        self.hand_name = os.path.basename(self.hand_xml_path).split(".")[0]
        self.object_fixed = bool(object_fixed)
        self.nworld = int(nworld)
        self.device = str(device)
        self.anchor_params = dict(anchor_params or {})
        self.ccd_iterations = int(ccd_iterations)
        if not hand_profile:
            raise ValueError(
                f"hand_profile must be provided explicitly for hand '{self.hand_name}'."
            )
        if not object_profile:
            raise ValueError("object_profile must be provided explicitly.")
        self.hand_profile = dict(hand_profile)
        self.object_profile = dict(object_profile)
        self.root_stabilization = (
            dict(root_stabilization) if root_stabilization is not None else None
        )
        self.hand_actuation_profile = self._read_hand_actuation_profile(
            self.hand_profile
        )
        self.side_swing_indices = np.asarray(
            self.hand_profile["side_swing_indices"],
            dtype=int,
        )
        self.side_swing_index_set = set(self.side_swing_indices.tolist())
        self.thumb_relax_indices = np.asarray(
            self.hand_profile["thumb_relax_indices"],
            dtype=int,
        )
        self.thumb_relax_divisor = float(self.hand_profile["thumb_relax_divisor"])
        self.slot_spacing = float(slot_spacing)

        self.spec = self._add_hand(self.hand_xml_path)
        self._slot_specs = self._add_bucket_objects(self.object_infos)
        self._set_ccd_iterations()

        self.mj_model = self.spec.compile()
        self.mj_model.opt.ccd_iterations = max(
            int(self.mj_model.opt.ccd_iterations), self.ccd_iterations
        )
        self.cpu_data = mujoco.MjData(self.mj_model)
        self._apply_root_stabilization()
        self.mjw_model = mjw.put_model(self.mj_model)

        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)
        self.nbody = int(self.mj_model.nbody)
        self.geom_bodyid = np.asarray(self.mj_model.geom_bodyid, dtype=np.int32)
        self.body_parentid = np.asarray(self.mj_model.body_parentid, dtype=np.int32)
        self.body_name_to_id: Dict[str, int] = {}
        for body_id in range(self.nbody):
            body_name = self.mj_model.body(body_id).name
            if body_name:
                self.body_name_to_id[str(body_name)] = int(body_id)

        self.slots = self._resolve_slots()
        self.slot_origins = np.asarray(
            [slot.origin for slot in self.slots], dtype=np.float32
        )
        self.all_object_body_ids = set()
        for slot in self.slots:
            self.all_object_body_ids.update(slot.body_ids)

        if self.object_fixed:
            self.nq_hand = self.nq
        else:
            free_qpos_adrs = [
                int(slot.freejoint_qposadr)
                for slot in self.slots
                if slot.freejoint_qposadr is not None
            ]
            if len(free_qpos_adrs) != len(self.slots):
                raise RuntimeError("All bucket slots must have a freejoint qposadr.")
            self.nq_hand = min(free_qpos_adrs)

        ctrl_joint_indices = np.asarray(
            self.hand_profile["ctrl_joint_indices"],
            dtype=int,
        ).reshape(-1)
        if ctrl_joint_indices.size <= 0:
            raise ValueError("hand.profile.ctrl_joint_indices must be non-empty.")
        if np.any(ctrl_joint_indices < 0):
            raise ValueError(
                "hand.profile.ctrl_joint_indices must be non-negative local joint ids."
            )
        max_local_joint_idx = self.nq_hand - 8
        if np.any(ctrl_joint_indices > max_local_joint_idx):
            raise ValueError(
                "hand.profile.ctrl_joint_indices contains local ids outside hand joint range. "
                f"max allowed is {max_local_joint_idx} for hand '{self.hand_name}'."
            )
        self.ctrl_qpos_indices = 7 + ctrl_joint_indices
        if int(self.ctrl_qpos_indices.shape[0]) != self.nu:
            raise ValueError(
                f"ctrl index length {self.ctrl_qpos_indices.shape[0]} != model.nu {self.nu} "
                f"for hand '{self.hand_name}'."
            )

        self.nconmax = int(nconmax)
        self.naconmax = int(naconmax)
        self.njmax = self._resolve_njmax(njmax=njmax, nconmax=self.nconmax)
        self.data = mjw.make_data(
            self.mj_model,
            nworld=self.nworld,
            nconmax=self.nconmax,
            naconmax=self.naconmax,
            njmax=self.njmax,
            njmax_nnz=njmax_nnz,
        )

        self._default_qpos = self._build_default_qpos()
        self._default_qvel = np.zeros((self.nworld, self.nv), dtype=np.float32)
        self._default_ctrl = np.zeros((self.nworld, self.nu), dtype=np.float32)
        self._default_xfrc = np.zeros((self.nworld, self.nbody, 6), dtype=np.float32)
        self._host_qpos = self._default_qpos.copy()
        self._host_qvel = self._default_qvel.copy()
        self._host_ctrl = self._default_ctrl.copy()
        self._host_xfrc = self._default_xfrc.copy()
        self._reset_mask_host = np.zeros((self.nworld,), dtype=bool)

        self.anchor_body_ids = []
        self.anchor_weights = np.zeros((0,), dtype=float)
        self.anchor_plane_axes = np.zeros((0, 3), dtype=float)
        self.n_anchors = 0
        if self.anchor_params:
            self._init_anchor_metadata()
        self.anchor_body_ids = np.asarray(self.anchor_body_ids, dtype=np.int32)
        self.slot_obj_pts: List[np.ndarray] = []
        self.slot_obj_norms: List[np.ndarray] = []
        self.slot_obj_trees: List[cKDTree] = []

        self.reset_batch()

    def _add_bucket_objects(self, object_infos: Sequence[Dict]) -> List[Dict]:
        slot_specs = []
        for slot_id, obj_info in enumerate(object_infos):
            obj_name = str(obj_info.get("name") or obj_info.get("object_name") or "")
            obj_xml = str(obj_info.get("xml_abs") or obj_info.get("mjcf_abs") or "")
            if not obj_name or not obj_xml:
                raise ValueError(
                    "Each object info must provide non-empty name/object_name and xml_abs/mjcf_abs."
                )
            obj_dir = os.path.dirname(os.path.abspath(obj_xml))
            obj_spec = mujoco.MjSpec.from_file(obj_xml)
            for mesh in getattr(obj_spec, "meshes", []):
                mesh_file = getattr(mesh, "file", None)
                if mesh_file and not os.path.isabs(mesh_file):
                    mesh.file = os.path.abspath(os.path.join(obj_dir, mesh_file))
            for texture in getattr(obj_spec, "textures", []):
                texture_file = getattr(texture, "file", None)
                if texture_file and not os.path.isabs(texture_file):
                    texture.file = os.path.abspath(os.path.join(obj_dir, texture_file))
            for geom in obj_spec.geoms:
                self._apply_geom_profile(geom, self.object_profile)

            origin = np.asarray(
                [float(slot_id) * self.slot_spacing, 0.0, 0.0], dtype=np.float32
            )
            prefix = f"bucket_slot{slot_id}_"
            asset_name = resolve_object_asset_name(obj_info)
            frame = self.spec.worldbody.add_frame()
            frame.pos = origin.astype(float).tolist()
            self.spec.attach(obj_spec, frame=frame, prefix=prefix)

            if self.object_fixed:
                joint_name = f"{prefix}{asset_name}_joint"
                for joint in list(self.spec.joints):
                    if joint.name == joint_name:
                        self.spec.delete(joint)
                        break

            slot_specs.append(
                {
                    "slot_id": slot_id,
                    "object_scale_key": str(
                        obj_info.get("object_scale_key")
                        or f"{obj_name}__{obj_info.get('scale_tag', slot_id)}"
                    ),
                    "object_name": obj_name,
                    "scale": obj_info.get("scale"),
                    "scale_tag": str(obj_info.get("scale_tag") or ""),
                    "prefix": prefix,
                    "asset_name": asset_name,
                    "root_body_name": f"{prefix}{asset_name}",
                    "origin": origin,
                }
            )
        return slot_specs

    def _resolve_slots(self) -> List[BucketSlot]:
        slots = []
        for spec in self._slot_specs:
            root_body_name = str(spec["root_body_name"])
            root_body_id = self.body_name_to_id.get(root_body_name)
            if root_body_id is None:
                raise RuntimeError(f"Bucket slot body not found: {root_body_name}")
            body_ids = frozenset(self._collect_descendant_body_ids(root_body_id))
            freejoint_qposadr = None
            if not self.object_fixed:
                joint_name = f"{spec['prefix']}{spec['asset_name']}_joint"
                joint_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if int(joint_id) < 0:
                    raise RuntimeError(f"Bucket slot freejoint not found: {joint_name}")
                freejoint_qposadr = int(self.mj_model.jnt_qposadr[int(joint_id)])
            slots.append(
                BucketSlot(
                    slot_id=int(spec["slot_id"]),
                    object_scale_key=str(spec["object_scale_key"]),
                    object_name=str(spec["object_name"]),
                    scale=(None if spec["scale"] is None else float(spec["scale"])),
                    scale_tag=str(spec["scale_tag"]),
                    prefix=str(spec["prefix"]),
                    asset_name=str(spec["asset_name"]),
                    root_body_name=root_body_name,
                    root_body_id=int(root_body_id),
                    body_ids=body_ids,
                    origin=np.asarray(spec["origin"], dtype=np.float32),
                    freejoint_qposadr=freejoint_qposadr,
                )
            )
        return slots

    def _init_anchor_metadata(self) -> None:
        anchor_body_names = list(self.anchor_params.keys())
        weights: List[float] = []
        axes: List[np.ndarray] = []
        axis_table = {
            "X": np.array([1.0, 0.0, 0.0], dtype=float),
            "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
            "Y": np.array([0.0, 1.0, 0.0], dtype=float),
            "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
            "Z": np.array([0.0, 0.0, 1.0], dtype=float),
            "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
        }
        for body_name in anchor_body_names:
            value = self.anchor_params[body_name]
            if isinstance(value, dict):
                weight = float(value["weight"])
                axis_key = str(value["axis"]).strip().upper()
            else:
                weight = float(value)
                axis_key = "Z"
            if axis_key not in axis_table:
                raise ValueError(
                    f"hand.anchor_params['{body_name}'].axis must be one of "
                    f"{sorted(axis_table.keys())}, got '{axis_key}'."
                )
            weights.append(weight)
            axes.append(axis_table[axis_key].copy())
        self.anchor_weights = np.asarray(weights, dtype=float)
        self.anchor_plane_axes = np.asarray(axes, dtype=float)
        for body_name in anchor_body_names:
            if body_name not in self.body_name_to_id:
                raise ValueError(f"Body name '{body_name}' not found in bucket model.")
            self.anchor_body_ids.append(int(self.body_name_to_id[body_name]))
        self.n_anchors = len(self.anchor_body_ids)

    def _build_default_qpos(self) -> np.ndarray:
        qpos0 = np.asarray(self.mj_model.qpos0, dtype=np.float32)
        qpos = np.repeat(qpos0[None, :], self.nworld, axis=0)
        if self.nq_hand >= 7:
            qpos[:, 0] = -1.0
            qpos[:, 3] = 1.0
        return qpos

    def apply_slot_offsets(
        self,
        hand_qpos_batch: np.ndarray,
        slot_ids: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32).copy()
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if hand_qpos_batch.ndim == 1:
            hand_qpos_batch = hand_qpos_batch[None, :]
        if hand_qpos_batch.shape[0] != slot_ids.size:
            raise ValueError("hand_qpos_batch rows must match slot_ids length.")
        hand_qpos_batch[:, :3] += self.slot_origins[slot_ids]
        return hand_qpos_batch

    def remove_slot_offsets(
        self,
        hand_qpos_batch: np.ndarray,
        slot_ids: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32).copy()
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if hand_qpos_batch.ndim == 1:
            hand_qpos_batch = hand_qpos_batch[None, :]
        if hand_qpos_batch.shape[0] != slot_ids.size:
            raise ValueError("hand_qpos_batch rows must match slot_ids length.")
        hand_qpos_batch[:, :3] -= self.slot_origins[slot_ids]
        return hand_qpos_batch

    def load_hand_qpos_to_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        hand_qpos_batch: np.ndarray,
        slot_ids: np.ndarray | Sequence[int],
        ctrl_batch: Optional[np.ndarray] = None,
        do_forward: bool = False,
    ) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError(
                f"slot_ids must have shape ({world_ids.size},), got {slot_ids.shape}."
            )
        if np.any(slot_ids < 0) or np.any(slot_ids >= len(self.slots)):
            raise ValueError(f"slot_ids out of range: {slot_ids}.")
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32)
        if hand_qpos_batch.ndim == 1:
            hand_qpos_batch = hand_qpos_batch[None, :]
        if hand_qpos_batch.shape != (world_ids.size, self.nq_hand):
            raise ValueError(
                f"hand_qpos_batch must have shape ({world_ids.size}, {self.nq_hand}), "
                f"got {hand_qpos_batch.shape}."
            )
        sim_qpos_batch = self.apply_slot_offsets(hand_qpos_batch, slot_ids)
        if ctrl_batch is None:
            ctrl_batch = (
                self.qpos_to_ctrl_batch(sim_qpos_batch)
                if self.nu > 0
                else np.zeros((world_ids.size, 0), dtype=np.float32)
            )
        ctrl_batch = np.asarray(ctrl_batch, dtype=np.float32)
        if ctrl_batch.shape != (world_ids.size, self.nu):
            raise ValueError(
                f"ctrl_batch must have shape ({world_ids.size}, {self.nu}), got {ctrl_batch.shape}."
            )

        for local_idx, world_id in enumerate(world_ids):
            qpos_row = self._default_qpos[int(world_id)].copy()
            qvel_row = self._default_qvel[int(world_id)].copy()
            ctrl_row = self._default_ctrl[int(world_id)].copy()
            xfrc_row = self._default_xfrc[int(world_id)].copy()

            qpos_row[: self.nq_hand] = sim_qpos_batch[local_idx]
            ctrl_row[:] = ctrl_batch[local_idx]

            self._copy_world_row(
                self.data.qpos, qpos_row, int(world_id), self.nq, wp.float32
            )
            self._copy_world_row(
                self.data.qvel, qvel_row, int(world_id), self.nv, wp.float32
            )
            self._copy_world_row(
                self.data.ctrl, ctrl_row, int(world_id), self.nu, wp.float32
            )
            self._copy_world_row(
                self.data.xfrc_applied,
                xfrc_row,
                int(world_id),
                self.nbody,
                wp.spatial_vectorf,
            )

        if do_forward:
            self.forward_batch()

    def get_hand_qpos_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
        remove_slot_offsets: bool = True,
    ) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0, self.nq_hand), dtype=np.float32)
        qpos = np.asarray(
            self.data.qpos.numpy()[world_ids, : self.nq_hand], dtype=np.float32
        )
        if remove_slot_offsets:
            qpos = self.remove_slot_offsets(qpos, slot_ids)
        return qpos

    def set_slot_object_points(
        self,
        slot_points: Sequence[np.ndarray],
        slot_normals: Sequence[np.ndarray],
    ) -> None:
        if len(slot_points) != len(self.slots) or len(slot_normals) != len(self.slots):
            raise ValueError(
                "slot_points and slot_normals must each contain one array per bucket slot."
            )
        pts_out: List[np.ndarray] = []
        norms_out: List[np.ndarray] = []
        trees: List[cKDTree] = []
        for slot_id, (points, normals) in enumerate(zip(slot_points, slot_normals)):
            pts = np.asarray(points, dtype=np.float32)
            norms = np.asarray(normals, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"slot {slot_id} points must have shape (N, 3), got {pts.shape}."
                )
            if norms.shape != pts.shape:
                raise ValueError(
                    f"slot {slot_id} normals shape {norms.shape} does not match points {pts.shape}."
                )
            pts_out.append(pts.copy())
            norms_out.append(norms.copy())
            trees.append(cKDTree(pts))
        self.slot_obj_pts = pts_out
        self.slot_obj_norms = norms_out
        self.slot_obj_trees = trees

    def _target_points_for_anchor(
        self,
        anchor_pos: np.ndarray,
        axis_world: np.ndarray,
        slot_ids: np.ndarray,
        Mp: int,
        method_id: int,
    ) -> np.ndarray:
        target = np.zeros_like(anchor_pos, dtype=np.float32)
        for row_idx, slot_id_raw in enumerate(slot_ids):
            slot_id = int(slot_id_raw)
            origin = self.slot_origins[slot_id]
            pts_local = self.slot_obj_pts[slot_id]
            pts_world = pts_local + origin[None, :]
            query_local = anchor_pos[row_idx] - origin
            if method_id == 1:
                _, idx_near = self.slot_obj_trees[slot_id].query(query_local, k=int(Mp))
                idx_near = np.asarray(idx_near, dtype=np.int64).reshape(-1)
                target[row_idx] = pts_world[idx_near].mean(axis=0)
            elif method_id == 2:
                signed = np.abs((pts_world - anchor_pos[row_idx]) @ axis_world[row_idx])
                if int(Mp) == 1:
                    top_idx = np.asarray([int(np.argmin(signed))], dtype=np.int64)
                else:
                    top_idx = np.argpartition(signed, int(Mp) - 1)[: int(Mp)]
                pts_top = pts_world[top_idx]
                chosen = int(
                    np.argmin(np.linalg.norm(pts_top - anchor_pos[row_idx], axis=1))
                )
                target[row_idx] = pts_top[chosen]
            else:
                _, idx_near = self.slot_obj_trees[slot_id].query(query_local, k=int(Mp))
                idx_near = np.asarray(idx_near, dtype=np.int64).reshape(-1)
                pts_top = pts_world[idx_near]
                plane_dist = np.abs(
                    (pts_top - anchor_pos[row_idx]) @ axis_world[row_idx]
                )
                target[row_idx] = pts_top[int(np.argmin(plane_dist))]
        return target

    def sim_grasp_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
        Mp: int = 20,
        steps: int = 40,
        speed_gain: float = 1.5,
        max_tip_speed: float = 0.05,
        target_point_method: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError("slot_ids must match world_ids length.")
        if world_ids.size == 0:
            return (
                np.zeros((0, self.nq_hand), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        method_id = int(target_point_method)
        if method_id not in {1, 2, 3}:
            raise ValueError(
                f"sim_grasp.target_point_method must be one of [1, 2, 3], got {target_point_method}."
            )
        if self.n_anchors <= 0:
            raise ValueError("hand.anchor_params must be configured for sim_grasp.")
        if len(self.slot_obj_trees) != len(self.slots):
            raise ValueError(
                "set_slot_object_points must be called before sim_grasp_for_worlds."
            )

        active_count = int(world_ids.size)
        body_ids_host = np.zeros((self.nworld,), dtype=np.int32)
        jac_points_host = np.zeros((self.nworld, 3), dtype=np.float32)
        jacp_wp = wp.zeros(
            (self.nworld, 3, self.nv), dtype=wp.float32, device=self.device
        )
        dof_dim = self.nq_hand - 7
        side_swing_mask = np.isin(np.arange(dof_dim), self.side_swing_indices)

        for _ in range(int(steps)):
            hand_qpos = self.get_hand_qpos_for_worlds(
                world_ids, slot_ids, remove_slot_offsets=False
            ).copy()
            xpos = np.asarray(
                self.data.xpos.numpy()[world_ids][:, self.anchor_body_ids],
                dtype=np.float32,
            )
            xmat = np.asarray(
                self.data.xmat.numpy()[world_ids][:, self.anchor_body_ids],
                dtype=np.float32,
            ).reshape(active_count, self.n_anchors, 3, 3)

            target_points = np.zeros(
                (active_count, self.n_anchors, 3), dtype=np.float32
            )
            for anchor_idx in range(self.n_anchors):
                axis_local = self.anchor_plane_axes[anchor_idx].astype(np.float32)
                axis_world = np.einsum(
                    "wij,j->wi",
                    xmat[:, anchor_idx, :, :],
                    axis_local,
                )
                target_points[:, anchor_idx, :] = self._target_points_for_anchor(
                    xpos[:, anchor_idx, :],
                    axis_world,
                    slot_ids,
                    int(Mp),
                    method_id,
                )

            v_anchors = float(speed_gain) * (target_points - xpos)
            norms = np.linalg.norm(v_anchors, axis=2, keepdims=True)
            too_fast = norms[..., 0] > float(max_tip_speed)
            if np.any(too_fast):
                v_anchors[too_fast] = (
                    v_anchors[too_fast] / (norms[too_fast] + 1e-12)
                ) * float(max_tip_speed)

            total_dq = np.zeros((active_count, dof_dim), dtype=np.float32)
            for anchor_idx, body_id in enumerate(self.anchor_body_ids):
                body_ids_host.fill(0)
                jac_points_host.fill(0.0)
                body_ids_host[world_ids] = int(body_id)
                jac_points_host[world_ids] = xpos[:, anchor_idx, :]
                body_wp = wp.array(body_ids_host, dtype=wp.int32, device=self.device)
                point_wp = wp.array(jac_points_host, dtype=wp.vec3, device=self.device)
                jacp_wp.zero_()
                mjw.jac(self.mjw_model, self.data, jacp_wp, None, point_wp, body_wp)
                jacp = np.asarray(jacp_wp.numpy()[world_ids], dtype=np.float32)
                j_hand = jacp[:, :, 6 : 6 + dof_dim]
                dq = np.einsum(
                    "wdc,wc->wd",
                    np.linalg.pinv(j_hand),
                    v_anchors[:, anchor_idx, :],
                )
                total_dq += float(self.anchor_weights[anchor_idx]) * dq.astype(
                    np.float32,
                    copy=False,
                )

            total_dq[:, ~side_swing_mask] = np.maximum(
                total_dq[:, ~side_swing_mask],
                0.0,
            )
            thumb_div = max(float(self.thumb_relax_divisor), 1e-6)
            for idx in self.thumb_relax_indices:
                if 0 <= int(idx) < dof_dim:
                    total_dq[:, int(idx)] /= thumb_div

            hand_qpos[:, 7:] += total_dq
            ctrl_batch = np.zeros((self.nworld, self.nu), dtype=np.float32)
            if self.nu > 0:
                ctrl_batch[world_ids] = self.qpos_to_ctrl_batch(hand_qpos)
            self.step_batch(1, ctrl_batch=ctrl_batch)

        qpos_grasp = self.get_hand_qpos_for_worlds(
            world_ids, slot_ids, remove_slot_offsets=True
        )
        ho_counts = self.count_ho_contacts_for_worlds(
            world_ids,
            slot_ids,
            obj_margin=0.0,
        )
        return qpos_grasp, ho_counts

    def count_ho_contacts_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
        obj_margin: float = 0.0,
    ) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError("slot_ids must match world_ids length.")
        counts = np.zeros((world_ids.size,), dtype=np.int32)
        if world_ids.size == 0:
            return counts
        world_to_local = {int(world_id): idx for idx, world_id in enumerate(world_ids)}
        active_contact_count = int(np.asarray(self.data.nacon.numpy()).reshape(-1)[0])
        if active_contact_count <= 0:
            return counts

        contact_world_ids = np.asarray(
            self.data.contact.worldid.numpy()[:active_contact_count], dtype=np.int32
        )
        dists = np.asarray(
            self.data.contact.dist.numpy()[:active_contact_count], dtype=np.float32
        )
        geom_pairs = np.asarray(
            self.data.contact.geom.numpy()[:active_contact_count], dtype=np.int32
        )
        for world_id, dist, geom_pair in zip(contact_world_ids, dists, geom_pairs):
            local_idx = world_to_local.get(int(world_id))
            if local_idx is None or float(dist) > float(obj_margin):
                continue
            slot = self.slots[int(slot_ids[local_idx])]
            body1 = int(self.geom_bodyid[int(geom_pair[0])])
            body2 = int(self.geom_bodyid[int(geom_pair[1])])
            is_obj1 = body1 in slot.body_ids
            is_obj2 = body2 in slot.body_ids
            if is_obj1 == is_obj2:
                continue
            if body1 == 0 or body2 == 0:
                continue
            other_body = body2 if is_obj1 else body1
            if other_body in self.all_object_body_ids:
                continue
            counts[local_idx] += 1
        return counts

    def read_contact_mask_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
        dist_threshold: float = 0.0,
    ) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError("slot_ids must match world_ids length.")
        mask = np.zeros((world_ids.size,), dtype=bool)
        if world_ids.size == 0:
            return mask

        world_to_local = {int(world_id): idx for idx, world_id in enumerate(world_ids)}
        active_contact_count = int(np.asarray(self.data.nacon.numpy()).reshape(-1)[0])
        if active_contact_count <= 0:
            return mask

        contact_world_ids = np.asarray(
            self.data.contact.worldid.numpy()[:active_contact_count], dtype=np.int32
        )
        dists = np.asarray(
            self.data.contact.dist.numpy()[:active_contact_count], dtype=np.float32
        )
        geom_pairs = np.asarray(
            self.data.contact.geom.numpy()[:active_contact_count], dtype=np.int32
        )
        for world_id, dist, geom_pair in zip(contact_world_ids, dists, geom_pairs):
            local_idx = world_to_local.get(int(world_id))
            if local_idx is None or float(dist) > float(dist_threshold):
                continue
            active_slot = self.slots[int(slot_ids[local_idx])]
            body1 = int(self.geom_bodyid[int(geom_pair[0])])
            body2 = int(self.geom_bodyid[int(geom_pair[1])])
            is_active_obj1 = body1 in active_slot.body_ids
            is_active_obj2 = body2 in active_slot.body_ids
            is_any_obj1 = body1 in self.all_object_body_ids
            is_any_obj2 = body2 in self.all_object_body_ids

            if is_active_obj1 or is_active_obj2:
                mask[local_idx] = True
                continue
            if not is_any_obj1 and not is_any_obj2:
                mask[local_idx] = True

        return mask

    def get_obj_pose_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError("slot_ids must match world_ids length.")
        pose = np.zeros((world_ids.size, 7), dtype=np.float32)
        if world_ids.size == 0:
            return pose
        if self.object_fixed:
            pose[:, :3] = self.slot_origins[slot_ids]
            pose[:, 3] = 1.0
            return pose
        qpos_all = np.asarray(self.data.qpos.numpy()[world_ids], dtype=np.float32)
        for local_idx, slot_id in enumerate(slot_ids):
            qposadr = self.slots[int(slot_id)].freejoint_qposadr
            if qposadr is None:
                raise RuntimeError("freejoint_qposadr is unavailable for free backend.")
            pose[local_idx] = qpos_all[local_idx, qposadr : qposadr + 7]
        return pose

    def set_object_force_to_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        slot_ids: np.ndarray | Sequence[int],
        body_force_batch: np.ndarray,
    ) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        body_force_batch = np.asarray(body_force_batch, dtype=np.float32)
        if slot_ids.shape != (world_ids.size,):
            raise ValueError("slot_ids must match world_ids length.")
        if body_force_batch.shape != (world_ids.size, 6):
            raise ValueError(
                f"body_force_batch must have shape ({world_ids.size}, 6), got {body_force_batch.shape}."
            )
        for local_idx, world_id in enumerate(world_ids):
            xfrc_row = self._default_xfrc[int(world_id)].copy()
            body_id = self.slots[int(slot_ids[local_idx])].root_body_id
            xfrc_row[int(body_id), :] = body_force_batch[local_idx]
            self._copy_world_row(
                self.data.xfrc_applied,
                xfrc_row,
                int(world_id),
                self.nbody,
                wp.spatial_vectorf,
            )
