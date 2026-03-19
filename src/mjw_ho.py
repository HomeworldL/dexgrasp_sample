"""MJWarp batched hand-object helper.

This module currently targets validation-oriented GPU batching:
- batched collision filtering
- heuristic batched grasp closing
- batched external-force stability checks

The first implementation favors end-to-end executability over peak performance.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from scipy.spatial import cKDTree

import mujoco
import numpy as np
import warp as wp

try:
    import mujoco_warp as mjw
except Exception as exc:  # pragma: no cover - depends on runtime install.
    raise RuntimeError(
        "mujoco_warp is required for src/mjw_ho.py. Install it before using run_collision.py."
    ) from exc


def _default_hand_profiles() -> Dict[str, Dict]:
    return {
        "liberhand_right": {
            "ctrl_qpos_slices": [
                (7, 10),
                (11, 14),
                (15, 17),
                (19, 21),
                (23, 26),
            ],
            "solimp": [0.4, 0.99, 0.0001],
            "solref": [0.003, 1.0],
        },
        "liberhand2_right": {
            "ctrl_qpos_slices": [
                (7, 10),
                (11, 14),
                (15, 18),
                (19, 22),
                (23, 26),
            ],
            "solimp": [0.4, 0.99, 0.0001],
            "solref": [0.003, 1.0],
        },
    }


@dataclass
class ContactBatchResult:
    has_contact: np.ndarray
    world_ids: np.ndarray
    distances: np.ndarray
    active_contact_count: int


@dataclass
class GraspBatchResult:
    qpos_grasp: np.ndarray
    ho_contact_counts: np.ndarray


@dataclass
class ExtForceBatchResult:
    is_valid: np.ndarray
    pos_delta: np.ndarray
    angle_delta_deg: np.ndarray


class MjWarpHO:
    """Validation-oriented MJWarp wrapper for batched hand-object simulation."""

    def __init__(
        self,
        obj_info: Dict,
        hand_xml_path: str,
        target_body_params: Optional[Dict] = None,
        hand_profile: Optional[Dict] = None,
        friction_coef: Sequence[float] = (0.2, 0.2),
        object_fixed: bool = True,
        nworld: int = 1,
        device: str = "cuda:0",
        nconmax: int = 512,
        naconmax: int = 512,
        njmax: Optional[int] = None,
        njmax_nnz: Optional[int] = None,
        ccd_iterations: int = 200,
    ):
        if int(nworld) <= 0:
            raise ValueError(f"nworld must be positive, got {nworld}.")

        wp.init()
        wp.set_device(device)

        self.obj_info = dict(obj_info)
        self.hand_xml_path = os.path.abspath(hand_xml_path)
        self.hand_name = os.path.basename(self.hand_xml_path).split(".")[0]
        self.object_fixed = bool(object_fixed)
        self.nworld = int(nworld)
        self.device = str(device)
        self.target_body_params = dict(target_body_params or {})
        self.ccd_iterations = int(ccd_iterations)

        profiles = _default_hand_profiles()
        self.hand_profile = dict(profiles.get(self.hand_name, {}))
        if hand_profile:
            self.hand_profile.update(hand_profile)
        self.friction_coef = tuple(float(v) for v in friction_coef)

        self.spec = self._add_hand(self.hand_xml_path)
        self._add_object(self.obj_info, fixed=self.object_fixed)
        self._set_friction(self.friction_coef)
        self._set_sol()
        self._set_ccd_iterations()

        self.mj_model = self.spec.compile()
        self.mj_model.opt.ccd_iterations = max(
            int(self.mj_model.opt.ccd_iterations), self.ccd_iterations
        )
        self.mjw_model = mjw.put_model(self.mj_model)
        self.cpu_data = mujoco.MjData(self.mj_model)

        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)
        self.nq_hand = self.nq - 7 if not self.object_fixed else self.nq
        self.nbody = int(self.mj_model.nbody)
        self.geom_bodyid = np.asarray(self.mj_model.geom_bodyid, dtype=np.int32)

        self.body_name_to_id: Dict[str, int] = {}
        for bi in range(self.nbody):
            body_name = self.mj_model.body(bi).name
            if body_name:
                self.body_name_to_id[str(body_name)] = bi
        self.body_parentid = np.asarray(self.mj_model.body_parentid, dtype=np.int32)
        self.obj_name = str(self.obj_info.get("name") or "")
        if self.obj_name and self.obj_name in self.body_name_to_id:
            self.object_root_body_id = int(self.body_name_to_id[self.obj_name])
        else:
            self.object_root_body_id = self.nbody - 1
        self.object_body_ids = self._collect_descendant_body_ids(self.object_root_body_id)
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

        self.tip_body_ids = []
        self.target_body_weights = np.zeros((0,), dtype=np.float32)
        if self.target_body_params:
            fingertip_bodies = list(self.target_body_params.keys())
            for body_name in fingertip_bodies:
                if body_name not in self.body_name_to_id:
                    raise ValueError(f"Body name '{body_name}' not found in MJWarp model.")
                self.tip_body_ids.append(int(self.body_name_to_id[body_name]))
            self.target_body_weights = np.asarray(
                [self.target_body_params[name][0] for name in fingertip_bodies], dtype=np.float32
            )
        self.tip_body_ids = np.asarray(self.tip_body_ids, dtype=np.int32)
        self.n_tips = int(self.tip_body_ids.shape[0])

        self.obj_pts = None
        self.obj_norms = None
        self.obj_tree = None

        self.reset_batch()

    def _collect_descendant_body_ids(self, root_body_id: int) -> set[int]:
        descendants: set[int] = set()
        for body_id in range(self.nbody):
            cur = body_id
            while cur > 0 and cur != root_body_id:
                cur = int(self.body_parentid[cur])
            if body_id == root_body_id or cur == root_body_id:
                descendants.add(int(body_id))
        return descendants

    def _add_hand(self, hand_xml_path: str) -> mujoco.MjSpec:
        hand_spec = mujoco.MjSpec.from_file(hand_xml_path)
        hand_spec.meshdir = os.path.dirname(hand_xml_path)
        return hand_spec

    def _add_object(self, obj_info: Dict, fixed: bool = True) -> None:
        obj_name = str(obj_info.get("name") or "")
        obj_xml = str(obj_info.get("xml_abs") or "")
        if not obj_name or not obj_xml:
            raise ValueError("obj_info must provide non-empty 'name' and 'xml_abs'.")

        obj_dir = os.path.dirname(obj_xml)
        obj_spec = mujoco.MjSpec.from_file(obj_xml)

        for mesh in getattr(obj_spec, "meshes", []):
            mesh_file = getattr(mesh, "file", None)
            if mesh_file and not os.path.isabs(mesh_file):
                mesh.file = os.path.abspath(os.path.join(obj_dir, mesh_file))

        for texture in getattr(obj_spec, "textures", []):
            texture_file = getattr(texture, "file", None)
            if texture_file and not os.path.isabs(texture_file):
                texture.file = os.path.abspath(os.path.join(obj_dir, texture_file))

        attach_frame = self.spec.worldbody.add_frame()
        self.spec.attach(obj_spec, frame=attach_frame, prefix="")

        if fixed:
            obj_joint_name = f"{obj_name}_joint"
            for joint in list(self.spec.joints):
                if joint.name == obj_joint_name:
                    self.spec.delete(joint)
                    break

    def _set_friction(self, friction_coef: Sequence[float]) -> None:
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.impratio = 10
        for geom in self.spec.geoms:
            geom.friction[:2] = friction_coef[:2]
            geom.condim = 4

    def _set_sol(self) -> None:
        solimp = self.hand_profile.get("solimp", [0.4, 0.99, 0.0001])
        solref = self.hand_profile.get("solref", [0.003, 1.0])
        for geom in self.spec.geoms:
            geom.solimp[:3] = solimp[:3]
            geom.solref[:2] = solref[:2]

    def _set_ccd_iterations(self) -> None:
        if self.ccd_iterations <= 0:
            raise ValueError(f"ccd_iterations must be positive, got {self.ccd_iterations}.")
        self.spec.option.ccd_iterations = max(
            int(self.spec.option.ccd_iterations), self.ccd_iterations
        )

    @staticmethod
    def _resolve_njmax(njmax: Optional[int], nconmax: int) -> int:
        if njmax is not None:
            resolved = int(njmax)
            if resolved <= 0:
                raise ValueError(f"njmax must be positive, got {resolved}.")
            return resolved
        if int(nconmax) <= 0:
            raise ValueError(f"nconmax must be positive when njmax is unset, got {nconmax}.")
        return max(128, int(nconmax) * 4)

    def _build_default_qpos(self) -> np.ndarray:
        qpos = np.zeros((self.nworld, self.nq), dtype=np.float32)
        if self.nq_hand >= 7:
            qpos[:, 0] = -1.0
            qpos[:, 3] = 1.0
        if not self.object_fixed:
            qpos[:, -4] = 1.0
        return qpos

    def _upload_state(self) -> None:
        self.data.qpos = wp.array(self._host_qpos, dtype=wp.float32, device=self.device)
        self.data.qvel = wp.array(self._host_qvel, dtype=wp.float32, device=self.device)
        self.data.ctrl = wp.array(self._host_ctrl, dtype=wp.float32, device=self.device)
        self.data.xfrc_applied = wp.array(
            self._host_xfrc, dtype=wp.spatial_vectorf, device=self.device
        )

    def _sync_host_from_device(self) -> None:
        self._host_qpos = np.asarray(self.data.qpos.numpy(), dtype=np.float32)
        self._host_qvel = np.asarray(self.data.qvel.numpy(), dtype=np.float32)
        self._host_ctrl = np.asarray(self.data.ctrl.numpy(), dtype=np.float32)
        self._host_xfrc = np.asarray(self.data.xfrc_applied.numpy(), dtype=np.float32)

    def _copy_world_row(
        self,
        dest: wp.array,
        row_value: np.ndarray,
        row_idx: int,
        row_width: int,
        dtype,
    ) -> None:
        src = wp.array(np.ascontiguousarray(row_value), dtype=dtype, device=self.device)
        wp.copy(dest, src, dest_offset=int(row_idx) * int(row_width), count=int(row_width))

    @staticmethod
    def _normalize_world_ids(world_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        world_ids = np.asarray(world_ids, dtype=np.int32).reshape(-1)
        if world_ids.size == 0:
            return world_ids
        if np.any(world_ids < 0):
            raise ValueError(f"world_ids must be non-negative, got {world_ids}.")
        return world_ids

    def reset_batch(self) -> None:
        self._host_qpos = self._default_qpos.copy()
        self._host_qvel = self._default_qvel.copy()
        self._host_ctrl = self._default_ctrl.copy()
        self._host_xfrc = self._default_xfrc.copy()
        self._upload_state()
        self.forward_batch()

    def qpos_to_ctrl_batch(self, hand_qpos_batch: np.ndarray) -> np.ndarray:
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32)
        if hand_qpos_batch.ndim != 2 or hand_qpos_batch.shape[1] != self.nq_hand:
            raise ValueError(
                f"hand_qpos_batch must have shape (N, {self.nq_hand}), got {hand_qpos_batch.shape}."
            )

        if self.nu == 0:
            return np.zeros((hand_qpos_batch.shape[0], 0), dtype=np.float32)

        slices = self.hand_profile.get("ctrl_qpos_slices")
        if not slices:
            raise NotImplementedError(
                f"No ctrl mapping defined for hand '{self.hand_name}' in MjWarpHO."
            )

        ctrl_parts = [hand_qpos_batch[:, int(q0) : int(q1)] for q0, q1 in slices]
        ctrl = np.concatenate(ctrl_parts, axis=1).astype(np.float32, copy=False)
        if ctrl.shape[1] != self.nu:
            raise ValueError(f"ctrl width {ctrl.shape[1]} does not match model.nu {self.nu}.")
        return ctrl

    def set_hand_qpos_batch(
        self,
        hand_qpos_batch: np.ndarray,
        valid_count: Optional[int] = None,
        do_forward: bool = False,
    ) -> None:
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32)
        if hand_qpos_batch.ndim == 1:
            hand_qpos_batch = hand_qpos_batch[None, :]
        if hand_qpos_batch.ndim != 2 or hand_qpos_batch.shape[1] != self.nq_hand:
            raise ValueError(
                f"hand_qpos_batch must have shape (N, {self.nq_hand}), got {hand_qpos_batch.shape}."
            )

        if valid_count is None:
            valid_count = hand_qpos_batch.shape[0]
        valid_count = int(valid_count)
        if valid_count <= 0 or valid_count > self.nworld:
            raise ValueError(
                f"valid_count must be in [1, {self.nworld}], got {valid_count}."
            )
        if hand_qpos_batch.shape[0] < valid_count:
            raise ValueError("hand_qpos_batch rows must be >= valid_count.")

        self._host_qpos = self._default_qpos.copy()
        self._host_qvel = self._default_qvel.copy()
        self._host_ctrl = self._default_ctrl.copy()
        self._host_xfrc = self._default_xfrc.copy()

        self._host_qpos[:valid_count, : self.nq_hand] = hand_qpos_batch[:valid_count]
        if self.nu > 0:
            self._host_ctrl[:valid_count] = self.qpos_to_ctrl_batch(hand_qpos_batch[:valid_count])

        self._upload_state()
        if do_forward:
            self.forward_batch()

    def forward_batch(self) -> None:
        mjw.forward(self.mjw_model, self.data)

    def step_batch(self, n_steps: int = 1, ctrl_batch: Optional[np.ndarray] = None) -> None:
        if ctrl_batch is not None:
            ctrl_batch = np.asarray(ctrl_batch, dtype=np.float32)
            if ctrl_batch.shape != (self.nworld, self.nu):
                raise ValueError(
                    f"ctrl_batch must have shape ({self.nworld}, {self.nu}), got {ctrl_batch.shape}."
                )
            self._host_ctrl = ctrl_batch.copy()
            self.data.ctrl = wp.array(self._host_ctrl, dtype=wp.float32, device=self.device)

        for _ in range(int(n_steps)):
            mjw.step(self.mjw_model, self.data)

    def _set_obj_pts_norms(self, obj_pts: np.ndarray, obj_norms: np.ndarray) -> None:
        obj_pts = np.asarray(obj_pts, dtype=np.float32)
        obj_norms = np.asarray(obj_norms, dtype=np.float32)
        if obj_pts.ndim != 2 or obj_pts.shape[1] != 3:
            raise ValueError("obj_pts must have shape (N, 3).")
        if obj_norms.shape != obj_pts.shape:
            raise ValueError("obj_norms must have the same shape as obj_pts.")
        self.obj_pts = obj_pts
        self.obj_norms = obj_norms
        self.obj_tree = cKDTree(obj_pts)

    def get_hand_qpos_batch(self, valid_count: Optional[int] = None) -> np.ndarray:
        if valid_count is None:
            valid_count = self.nworld
        return np.asarray(self.data.qpos.numpy()[:valid_count, : self.nq_hand], dtype=np.float32)

    def get_obj_pose_batch(self, valid_count: Optional[int] = None) -> np.ndarray:
        if valid_count is None:
            valid_count = self.nworld
        if self.object_fixed:
            pose = np.zeros((valid_count, 7), dtype=np.float32)
            pose[:, 3] = 1.0
            return pose
        return np.asarray(self.data.qpos.numpy()[:valid_count, -7:], dtype=np.float32)

    def get_tip_positions_batch(self, valid_count: Optional[int] = None) -> np.ndarray:
        if valid_count is None:
            valid_count = self.nworld
        if self.n_tips <= 0:
            raise RuntimeError("tip_body_ids are unavailable; target_body_params must be configured.")
        return np.asarray(self.data.xpos.numpy()[:valid_count][:, self.tip_body_ids], dtype=np.float32)

    def set_object_force_batch(
        self,
        body_force_batch: np.ndarray,
        body_id: Optional[int] = None,
        valid_count: Optional[int] = None,
    ) -> None:
        if valid_count is None:
            valid_count = self.nworld
        if body_id is None:
            body_id = self.object_root_body_id
        body_force_batch = np.asarray(body_force_batch, dtype=np.float32)
        if body_force_batch.shape != (valid_count, 6):
            raise ValueError(
                f"body_force_batch must have shape ({valid_count}, 6), got {body_force_batch.shape}."
            )
        self._host_xfrc[:] = 0.0
        self._host_xfrc[:valid_count, int(body_id), :] = body_force_batch
        self.data.xfrc_applied = wp.array(
            self._host_xfrc, dtype=wp.spatial_vectorf, device=self.device
        )

    def set_ctrl_batch(self, ctrl_batch: np.ndarray) -> None:
        ctrl_batch = np.asarray(ctrl_batch, dtype=np.float32)
        if ctrl_batch.shape != (self.nworld, self.nu):
            raise ValueError(
                f"ctrl_batch must have shape ({self.nworld}, {self.nu}), got {ctrl_batch.shape}."
            )
        self._host_ctrl = ctrl_batch.copy()
        self.data.ctrl = wp.array(self._host_ctrl, dtype=wp.float32, device=self.device)

    def reset_worlds(self, world_ids: np.ndarray | Sequence[int], do_forward: bool = False) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return
        if np.any(world_ids >= self.nworld):
            raise ValueError(f"world_ids must be < {self.nworld}, got {world_ids}.")
        self._reset_mask_host.fill(False)
        self._reset_mask_host[world_ids] = True
        reset_mask = wp.array(self._reset_mask_host, dtype=bool, device=self.device)
        mjw.reset_data(self.mjw_model, self.data, reset=reset_mask)
        if do_forward:
            self.forward_batch()

    def load_hand_qpos_to_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        hand_qpos_batch: np.ndarray,
        ctrl_batch: Optional[np.ndarray] = None,
        do_forward: bool = False,
    ) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        hand_qpos_batch = np.asarray(hand_qpos_batch, dtype=np.float32)
        if hand_qpos_batch.ndim == 1:
            hand_qpos_batch = hand_qpos_batch[None, :]
        if hand_qpos_batch.shape != (world_ids.size, self.nq_hand):
            raise ValueError(
                f"hand_qpos_batch must have shape ({world_ids.size}, {self.nq_hand}), "
                f"got {hand_qpos_batch.shape}."
            )
        if ctrl_batch is None:
            if self.nu > 0:
                ctrl_batch = self.qpos_to_ctrl_batch(hand_qpos_batch)
            else:
                ctrl_batch = np.zeros((world_ids.size, 0), dtype=np.float32)
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

            qpos_row[: self.nq_hand] = hand_qpos_batch[local_idx]
            ctrl_row[:] = ctrl_batch[local_idx]

            self._copy_world_row(self.data.qpos, qpos_row, int(world_id), self.nq, wp.float32)
            self._copy_world_row(self.data.qvel, qvel_row, int(world_id), self.nv, wp.float32)
            self._copy_world_row(self.data.ctrl, ctrl_row, int(world_id), self.nu, wp.float32)
            self._copy_world_row(
                self.data.xfrc_applied, xfrc_row, int(world_id), self.nbody, wp.spatial_vectorf
            )

        if do_forward:
            self.forward_batch()

    def set_object_force_to_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        body_force_batch: np.ndarray,
        body_id: Optional[int] = None,
    ) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        if body_id is None:
            body_id = self.object_root_body_id
        body_force_batch = np.asarray(body_force_batch, dtype=np.float32)
        if body_force_batch.shape != (world_ids.size, 6):
            raise ValueError(
                f"body_force_batch must have shape ({world_ids.size}, 6), got {body_force_batch.shape}."
            )
        for local_idx, world_id in enumerate(world_ids):
            xfrc_row = self._default_xfrc[int(world_id)].copy()
            xfrc_row[int(body_id), :] = body_force_batch[local_idx]
            self._copy_world_row(
                self.data.xfrc_applied, xfrc_row, int(world_id), self.nbody, wp.spatial_vectorf
            )

    def get_obj_pose_for_worlds(self, world_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        if self.object_fixed:
            pose = np.zeros((world_ids.size, 7), dtype=np.float32)
            pose[:, 3] = 1.0
            return pose
        return np.asarray(self.data.qpos.numpy()[world_ids, -7:], dtype=np.float32)

    def read_contact_mask_for_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        dist_threshold: float = 0.0,
    ) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0,), dtype=bool)
        active_contact_count = int(np.asarray(self.data.nacon.numpy()).reshape(-1)[0])
        if active_contact_count <= 0:
            return np.zeros((world_ids.size,), dtype=bool)

        contact_world_ids = np.asarray(
            self.data.contact.worldid.numpy()[:active_contact_count], dtype=np.int32
        )
        distances = np.asarray(self.data.contact.dist.numpy()[:active_contact_count], dtype=np.float32)
        valid = distances <= float(dist_threshold)
        contact_world_ids = contact_world_ids[valid]
        if contact_world_ids.size == 0:
            return np.zeros((world_ids.size,), dtype=bool)
        contact_set = set(np.unique(contact_world_ids).tolist())
        return np.asarray([int(world_id) in contact_set for world_id in world_ids], dtype=bool)

    def get_qpos_for_worlds(self, world_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0, self.nq), dtype=np.float32)
        return np.asarray(self.data.qpos.numpy()[world_ids], dtype=np.float32)

    def get_qvel_for_worlds(self, world_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0, self.nv), dtype=np.float32)
        return np.asarray(self.data.qvel.numpy()[world_ids], dtype=np.float32)

    def get_ctrl_for_worlds(self, world_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        world_ids = self._normalize_world_ids(world_ids)
        if world_ids.size == 0:
            return np.zeros((0, self.nu), dtype=np.float32)
        return np.asarray(self.data.ctrl.numpy()[world_ids], dtype=np.float32)

    def load_state_to_worlds(
        self,
        world_ids: np.ndarray | Sequence[int],
        qpos_batch: np.ndarray,
        qvel_batch: Optional[np.ndarray] = None,
        ctrl_batch: Optional[np.ndarray] = None,
        do_forward: bool = False,
    ) -> None:
        world_ids = self._normalize_world_ids(world_ids)
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.shape != (world_ids.size, self.nq):
            raise ValueError(
                f"qpos_batch must have shape ({world_ids.size}, {self.nq}), got {qpos_batch.shape}."
            )
        if qvel_batch is None:
            qvel_batch = np.zeros((world_ids.size, self.nv), dtype=np.float32)
        qvel_batch = np.asarray(qvel_batch, dtype=np.float32)
        if qvel_batch.shape != (world_ids.size, self.nv):
            raise ValueError(
                f"qvel_batch must have shape ({world_ids.size}, {self.nv}), got {qvel_batch.shape}."
            )
        if ctrl_batch is None:
            ctrl_batch = np.zeros((world_ids.size, self.nu), dtype=np.float32)
        ctrl_batch = np.asarray(ctrl_batch, dtype=np.float32)
        if ctrl_batch.shape != (world_ids.size, self.nu):
            raise ValueError(
                f"ctrl_batch must have shape ({world_ids.size}, {self.nu}), got {ctrl_batch.shape}."
            )

        for local_idx, world_id in enumerate(world_ids):
            xfrc_row = self._default_xfrc[int(world_id)].copy()
            self._copy_world_row(self.data.qpos, qpos_batch[local_idx], int(world_id), self.nq, wp.float32)
            self._copy_world_row(self.data.qvel, qvel_batch[local_idx], int(world_id), self.nv, wp.float32)
            self._copy_world_row(self.data.ctrl, ctrl_batch[local_idx], int(world_id), self.nu, wp.float32)
            self._copy_world_row(
                self.data.xfrc_applied, xfrc_row, int(world_id), self.nbody, wp.spatial_vectorf
            )

        if do_forward:
            self.forward_batch()

    def read_contact_batch(
        self,
        valid_count: Optional[int] = None,
        dist_threshold: float = 0.0,
    ) -> ContactBatchResult:
        if valid_count is None:
            valid_count = self.nworld
        valid_count = int(valid_count)

        active_contact_count = int(np.asarray(self.data.nacon.numpy()).reshape(-1)[0])
        if active_contact_count <= 0:
            return ContactBatchResult(
                has_contact=np.zeros((valid_count,), dtype=bool),
                world_ids=np.zeros((0,), dtype=np.int32),
                distances=np.zeros((0,), dtype=np.float32),
                active_contact_count=0,
            )

        world_ids = np.asarray(self.data.contact.worldid.numpy()[:active_contact_count], dtype=np.int32)
        distances = np.asarray(self.data.contact.dist.numpy()[:active_contact_count], dtype=np.float32)

        valid = (world_ids >= 0) & (world_ids < valid_count) & (distances <= float(dist_threshold))
        has_contact = np.zeros((valid_count,), dtype=bool)
        if np.any(valid):
            has_contact[np.unique(world_ids[valid])] = True

        return ContactBatchResult(
            has_contact=has_contact,
            world_ids=world_ids,
            distances=distances,
            active_contact_count=active_contact_count,
        )

    def check_contact_batch(
        self,
        hand_qpos_batch: np.ndarray,
        valid_count: Optional[int] = None,
        dist_threshold: float = 0.0,
    ) -> ContactBatchResult:
        self.set_hand_qpos_batch(hand_qpos_batch, valid_count=valid_count, do_forward=False)
        self.forward_batch()
        return self.read_contact_batch(valid_count=valid_count, dist_threshold=dist_threshold)

    def warmup(self, hand_qpos_batch: np.ndarray, valid_count: Optional[int] = None) -> float:
        ts = time.perf_counter()
        self.check_contact_batch(hand_qpos_batch, valid_count=valid_count)
        return time.perf_counter() - ts

    def count_ho_contacts_batch(
        self,
        valid_count: Optional[int] = None,
        obj_margin: float = 0.0,
    ) -> np.ndarray:
        if valid_count is None:
            valid_count = self.nworld
        active_contact_count = int(np.asarray(self.data.nacon.numpy()).reshape(-1)[0])
        counts = np.zeros((valid_count,), dtype=np.int32)
        if active_contact_count <= 0:
            return counts

        world_ids = np.asarray(self.data.contact.worldid.numpy()[:active_contact_count], dtype=np.int32)
        dists = np.asarray(self.data.contact.dist.numpy()[:active_contact_count], dtype=np.float32)
        geom_pairs = np.asarray(self.data.contact.geom.numpy()[:active_contact_count], dtype=np.int32)
        for world_id, dist, geom_pair in zip(world_ids, dists, geom_pairs):
            if world_id < 0 or world_id >= valid_count or dist > obj_margin:
                continue
            body1 = int(self.geom_bodyid[int(geom_pair[0])])
            body2 = int(self.geom_bodyid[int(geom_pair[1])])
            is_obj1 = body1 in self.object_body_ids
            is_obj2 = body2 in self.object_body_ids
            if is_obj1 == is_obj2:
                continue
            if body1 == 0 or body2 == 0:
                continue
            counts[world_id] += 1
        return counts

    def _build_close_delta(self, joint_step: float) -> np.ndarray:
        delta = np.zeros((self.nq_hand,), dtype=np.float32)
        if self.nq_hand <= 7:
            return delta
        delta[7:] = float(joint_step)
        side_swing = self.hand_profile.get("side_swing_indices", [0, 4, 8, 12, 16])
        for idx in side_swing:
            joint_idx = 7 + int(idx)
            if 0 <= joint_idx < self.nq_hand:
                delta[joint_idx] = 0.0
        thumb_indices = self.hand_profile.get("thumb_relax_indices", [17, 18, 19])
        thumb_div = max(float(self.hand_profile.get("thumb_relax_divisor", 1.2)), 1e-6)
        for idx in thumb_indices:
            joint_idx = 7 + int(idx)
            if 0 <= joint_idx < self.nq_hand:
                delta[joint_idx] /= thumb_div
        return delta

    def sim_grasp_batch(
        self,
        qpos_prepared_batch: np.ndarray,
        valid_count: Optional[int] = None,
        Mp: int = 20,
        steps: int = 40,
        speed_gain: float = 1.5,
        max_tip_speed: float = 0.05,
        visualize: bool = False,
    ) -> GraspBatchResult:
        del Mp, visualize
        if valid_count is None:
            valid_count = int(np.asarray(qpos_prepared_batch).shape[0])
        if valid_count <= 0:
            raise ValueError("valid_count must be positive for sim_grasp_batch.")

        self.set_hand_qpos_batch(qpos_prepared_batch, valid_count=valid_count, do_forward=True)
        target_qpos = self.get_hand_qpos_batch(valid_count=valid_count).copy()

        joint_step = min(0.05, max(0.01, float(speed_gain) * float(max_tip_speed)))
        close_delta = self._build_close_delta(joint_step)

        for _ in range(int(steps)):
            target_qpos += close_delta[None, :]
            ctrl_batch = np.zeros((self.nworld, self.nu), dtype=np.float32)
            if self.nu > 0:
                ctrl_batch[:valid_count] = self.qpos_to_ctrl_batch(target_qpos[:valid_count])
            self.step_batch(1, ctrl_batch=ctrl_batch)

        qpos_grasp = self.get_hand_qpos_batch(valid_count=valid_count)
        ho_contact_counts = self.count_ho_contacts_batch(valid_count=valid_count)
        return GraspBatchResult(qpos_grasp=qpos_grasp, ho_contact_counts=ho_contact_counts)

    @staticmethod
    def _pose_delta_batch(initial_pose: np.ndarray, current_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delta_pos = np.linalg.norm(current_pose[:, :3] - initial_pose[:, :3], axis=1)
        q1 = initial_pose[:, 3:7]
        q2 = current_pose[:, 3:7]
        dots = np.abs(np.sum(q1 * q2, axis=1))
        dots = np.clip(dots, 0.0, 1.0)
        delta_angle_deg = np.degrees(2.0 * np.arccos(dots))
        return delta_pos.astype(np.float32), delta_angle_deg.astype(np.float32)

    def sim_under_extforce_batch(
        self,
        qpos_grasp_batch: np.ndarray,
        valid_count: Optional[int] = None,
        duration: float = 1.0,
        trans_thresh: float = 0.05,
        angle_thresh: float = 10.0,
        grip_delta: float = 0.05,
        force_mag: float = 1.0,
        check_step: int = 50,
        visualize: bool = False,
    ) -> ExtForceBatchResult:
        del visualize
        if self.object_fixed:
            raise RuntimeError("sim_under_extforce_batch requires object_fixed=False.")
        if valid_count is None:
            valid_count = int(np.asarray(qpos_grasp_batch).shape[0])
        if valid_count <= 0:
            raise ValueError("valid_count must be positive for sim_under_extforce_batch.")

        qpos_grasp_batch = np.asarray(qpos_grasp_batch, dtype=np.float32)
        self.set_hand_qpos_batch(qpos_grasp_batch, valid_count=valid_count, do_forward=True)
        initial_obj_pose = self.get_obj_pose_batch(valid_count=valid_count).copy()

        grip_qpos = qpos_grasp_batch[:valid_count].copy()
        side_swing = set(self.hand_profile.get("side_swing_indices", [0, 4, 8, 12, 16]))
        for idx in range(7, self.nq_hand):
            joint_local_idx = idx - 7
            if joint_local_idx not in side_swing:
                grip_qpos[:, idx] += float(grip_delta)
        hand_ctrl_batch = np.zeros((self.nworld, self.nu), dtype=np.float32)
        if self.nu > 0:
            hand_ctrl_batch[:valid_count] = self.qpos_to_ctrl_batch(grip_qpos)

        dt = float(self.mj_model.opt.timestep)
        n_steps = max(1, int(duration / max(dt, 1e-8)))
        chunk_size = max(int(check_step), 1)
        full_chunks, remainder = divmod(n_steps, chunk_size)
        chunk_steps = [chunk_size] * full_chunks
        if remainder > 0:
            chunk_steps.append(remainder)
        if not chunk_steps:
            chunk_steps = [chunk_size]

        external_force_dirs = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        valid_mask = np.ones((valid_count,), dtype=bool)
        pos_delta = np.zeros((valid_count,), dtype=np.float32)
        angle_delta = np.zeros((valid_count,), dtype=np.float32)

        for dir_vec in external_force_dirs:
            self.set_hand_qpos_batch(qpos_grasp_batch, valid_count=valid_count, do_forward=True)
            body_force_batch = np.zeros((valid_count, 6), dtype=np.float32)
            body_force_batch[:, :3] = dir_vec[:3] * float(force_mag)
            self.set_object_force_batch(body_force_batch, valid_count=valid_count)
            if self.nu > 0:
                self.set_ctrl_batch(hand_ctrl_batch)

            dir_alive = valid_mask.copy()
            for steps_this_chunk in chunk_steps:
                self.step_batch(int(steps_this_chunk))

                has_contact = self.read_contact_batch(valid_count=valid_count).has_contact
                current_obj_pose = self.get_obj_pose_batch(valid_count=valid_count)
                dir_pos_delta, dir_angle_delta = self._pose_delta_batch(initial_obj_pose, current_obj_pose)
                still_ok = has_contact & (dir_pos_delta < float(trans_thresh)) & (
                    dir_angle_delta < float(angle_thresh)
                )
                dir_alive &= still_ok
                pos_delta = np.maximum(pos_delta, dir_pos_delta)
                angle_delta = np.maximum(angle_delta, dir_angle_delta)
                if not np.any(dir_alive):
                    break

            valid_mask &= dir_alive
            if not np.any(valid_mask):
                break

        body_force_batch = np.zeros((valid_count, 6), dtype=np.float32)
        self.set_object_force_batch(body_force_batch, valid_count=valid_count)
        return ExtForceBatchResult(
            is_valid=valid_mask,
            pos_delta=pos_delta,
            angle_delta_deg=angle_delta,
        )
