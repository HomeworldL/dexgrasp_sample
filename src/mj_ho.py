# src/mj_ho.py
"""
MjHO - MuJoCo Hand+Object helper

Chinese conversation, English comments.

Provides:
 - MjHO class: build a MuJoCo model that merges an existing hand MJCF and a decomposed object (convex pieces)
 - RobotKinematics: extract meshes and forward-kinematics utilities for a hand MJCF
"""

import os
import numpy as np
from scipy.spatial import cKDTree
import time
import trimesh
import mujoco
import transforms3d.quaternions as tq
from typing import Dict, List, Tuple, Optional, Sequence

try:
    from mujoco import viewer as mj_viewer
except Exception:
    mj_viewer = None


def _normalize_friction_coef(
    friction_coef: float | Sequence[float],
) -> Tuple[np.ndarray, int]:
    values = np.asarray(friction_coef, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("friction_coef must not be empty.")
    if values.size == 1:
        coeffs = np.array([float(values[0]), float(values[0])], dtype=float)
    elif values.size in {2, 3}:
        coeffs = values.astype(float)
    else:
        raise ValueError(
            "friction_coef must have length 1, 2, or 3. "
            f"Got shape {tuple(values.shape)}."
        )
    if np.any(coeffs < 0.0):
        raise ValueError(f"friction coefficients must be non-negative, got {coeffs.tolist()}.")
    condim = 6 if coeffs.shape[0] == 3 else 4
    return coeffs, condim


class MjHO:
    def __init__(
        self,
        obj_info: Dict,
        hand_xml_path: str,
        anchor_params: Optional[Dict] = None,
        hand_profile: Optional[Dict] = None,
        object_profile: Optional[Dict] = None,
        object_fixed: bool = True,
        visualize: bool = False,
    ):
        """
        Args:
            obj_info: dict returned by DatasetObjects entry query methods
            hand_xml_path: path to hand MJCF (used as base)
            object_fixed: if True, weld object to world (no DOFs)
        """
        self.obj_info = obj_info
        self.hand_xml_path = os.path.abspath(hand_xml_path)
        self.hand_name = os.path.basename(self.hand_xml_path).split(".")[0]
        if not hand_profile:
            raise ValueError(
                f"hand_profile must be provided explicitly for hand '{self.hand_name}'."
            )
        if not object_profile:
            raise ValueError("object_profile must be provided explicitly.")
        self.hand_profile = dict(hand_profile)
        self.object_profile = dict(object_profile)
        self.object_fixed = object_fixed

        # load hand spec as base
        self.spec = self._add_hand(self.hand_xml_path)

        # add object assets/geoms
        self._add_object(self.obj_info, fixed=self.object_fixed)

        # compile into model/data
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        # Set margin and gap to detect contact
        # self._set_margin(0.001)
        # self._set_gap(0.00)

        # indices and sizes
        self.nq = self.model.nq
        self.nu = self.model.nu

        # hand qpos length = total nq minus 7 if object exists
        self.nq_hand = self.nq - 7 if not self.object_fixed else self.nq
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
        self.object_body_id = int(self.model.nbody - 1)
        self.hand_body_upper_id = int(self.model.nbody - 2)
        self.world_body_id = 0
        self.obj_collision_geom_indices = [
            gi
            for gi in range(self.model.ngeom)
            if f"{self.obj_name}_collision" in self.model.geom(gi).name
        ]
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
        if self.ctrl_qpos_indices.shape[0] != self.nu:
            raise ValueError(
                f"ctrl index length {self.ctrl_qpos_indices.shape[0]} != model.nu {self.nu} "
                f"for hand '{self.hand_name}'."
            )

        # build mapping: body names -> body ids
        self.body_name_to_id = {}
        for bi in range(self.model.nbody):
            bname = self.model.body(bi).name
            if bname is not None:
                self.body_name_to_id[str(bname)] = bi

        self.anchor_body_names: Optional[List[str]] = None
        self.anchor_body_ids: List[int] = []
        self.anchor_weights = np.zeros((0,), dtype=float)
        self.n_anchors = 0
        if anchor_params is not None:
            self.anchor_body_names = list(anchor_params.keys())
            self.anchor_weights = np.asarray(
                [float(value) for value in anchor_params.values()],
                dtype=float,
            )
            for bname in self.anchor_body_names:
                if bname not in self.body_name_to_id:
                    raise ValueError(f"Body name '{bname}' not found in model bodies")
                self.anchor_body_ids.append(self.body_name_to_id[bname])
            self.n_anchors = len(self.anchor_body_ids)

        self.reset()

        self.viewer = None
        if visualize:
            self.open_viewer()

    # -----------------------
    # spec construction
    # -----------------------
    def _apply_geom_profile(self, geom, profile: Dict) -> None:
        """Apply config-defined contact parameters to one geom."""
        friction, condim = _normalize_friction_coef(profile["friction_coef"])
        geom.friction[: friction.shape[0]] = friction
        geom.condim = condim
        geom.solimp[:5] = np.asarray(profile["solimp"], dtype=float)
        geom.solref[:2] = np.asarray(profile["solref"], dtype=float)

    def _add_hand(self, hand_xml_path: str) -> mujoco.MjSpec:
        """Load hand MJCF and apply the hand contact profile."""
        hand_spec = mujoco.MjSpec.from_file(hand_xml_path)
        # adjust meshdir so relative mesh paths resolve relative to hand xml
        hand_spec.meshdir = os.path.dirname(hand_xml_path)
        hand_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        hand_spec.option.noslip_iterations = 2
        hand_spec.option.impratio = 10
        for geom in hand_spec.geoms:
            self._apply_geom_profile(geom, self.hand_profile)
        return hand_spec

    def _add_object(self, obj_info: Dict, fixed: bool = True):
        """
        Merge object MJCF assets/worldbody and apply the object contact profile.

        - obj_info['xml_abs'] should point to the object's pre-scaled MJCF/XML.
        - Only mesh assets and worldbody are merged (no texture/material copying).
        - If fixed==True, add a weld equality between 'world' and the object's root body.
        """
        obj_name = obj_info.get("name")
        obj_xml = obj_info.get("xml_abs")
        obj_dir = os.path.dirname(obj_xml)
        obj_spec = mujoco.MjSpec.from_file(obj_xml)
        self.obj_info = obj_info
        self.obj_name = obj_name
        # print(f"loading object xml: {obj_xml}")
        # attr in obj_spec
        # for a in dir(obj_spec):
        #     print(f"obj_spec.attr: {a}")

        # add meshes from object spec -> base spec
        for m in getattr(obj_spec, "meshes", []):
            mesh_file = getattr(m, "file", None)
            if mesh_file is not None:
                # resolve relative mesh file paths relative to the object's xml directory
                if not os.path.isabs(mesh_file):
                    mesh_file = os.path.join(obj_dir, mesh_file)
                    mesh_file = os.path.abspath(mesh_file)

                m.file = mesh_file

        for text in getattr(obj_spec, "textures", []):
            # pass
            texture_file = getattr(text, "file", None)
            if texture_file is not None:
                if not os.path.isabs(texture_file):
                    texture_file = os.path.join(obj_dir, texture_file)
                    texture_file = os.path.abspath(texture_file)
                text.file = texture_file

        for geom in obj_spec.geoms:
            self._apply_geom_profile(geom, self.object_profile)

        attach_frame = self.spec.worldbody.add_frame()
        self.spec.attach(obj_spec, frame=attach_frame, prefix="")

        if fixed:
            # delete free joints from object
            for j in self.spec.joints:
                # print(f"joint {j.name} ")
                if j.name == obj_name + "_joint":
                    # print(f"joint {j.name} is fixed")
                    # self.spec.joints.remove(j)
                    # j.type = mujoco.mjtJoint.mjJNT_WELD
                    self.spec.delete(j)
                    break

    def _set_margin(self, obj_margin):
        for i in range(self.model.ngeom):
            if f"{self.obj_name}_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = obj_margin

    def _set_gap(self, obj_gap):
        for i in range(self.model.ngeom):
            if f"{self.obj_name}_collision" in self.model.geom(i).name:
                self.model.geom_gap[i] = obj_gap

    def _set_obj_pts_norms(self, obj_pts, obj_norms):
        obj_pts = np.asarray(obj_pts, dtype=float)
        obj_norms = np.asarray(obj_norms, dtype=float)
        if obj_pts.ndim != 2 or obj_pts.shape[1] != 3:
            raise ValueError("obj_pts must be shape (N,3)")
        if obj_norms.ndim != 2 or obj_norms.shape != obj_pts.shape:
            raise ValueError("obj_norms must have same shape as obj_pts (N,3)")

        self.obj_pts = obj_pts
        self.obj_norms = obj_norms
        self.obj_tree = cKDTree(obj_pts)
        self.obj_pts_num = obj_pts.shape[0]

    # -----------------------
    # utility / IO
    # -----------------------
    def open_viewer(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        if mj_viewer is None:
            raise RuntimeError("Official MuJoCo viewer is unavailable in current mujoco build.")
        self.viewer = mj_viewer.launch_passive(self.model, self.data)

    def _render_viewer(self):
        if self.viewer is None:
            return
        try:
            if hasattr(self.viewer, "sync"):
                self.viewer.sync()
            else:
                self.viewer.render()
        except Exception:
            pass

    def _viewer_alive(self) -> bool:
        if self.viewer is None:
            return False
        if hasattr(self.viewer, "is_running"):
            return bool(self.viewer.is_running())
        return True


    def export_xml(self, out_path: str):
        """Export the merged spec to an XML file for inspection."""
        try:
            xml_str = self.spec.to_xml()
        except AttributeError:
            raise RuntimeError(
                "MjSpec.to_xml() not available in this mujoco-python version. "
                "Upgrade mujoco or implement an alternative writer."
            )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

    # -----------------------
    # runtime control
    # -----------------------
    def reset(self):
        """Reset data (keyframe if available) and set sane default qpos for hand/object."""
        try:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        except Exception:
            # fallback zero
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0

        # set hand root default: first 7 entries -> [0,0,0, 1,0,0,0] (pos + quat with qw=1)
        # only if model has at least 7 entries
        self.data.qpos[:7] = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # set object last 7 if present
        if not self.object_fixed:
            self.data.qpos[-7:] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self.model, self.data)

    def set_hand_qpos(self, hand_qpos: np.ndarray):
        """
        Set hand qpos (first self.hand_qpos_len entries) and run forward kinematics.
        hand_qpos length must equal self.hand_qpos_len.
        """
        if len(hand_qpos) != self.nq_hand:
            raise ValueError(f"hand_qpos must have length {self.nq_hand}")
        self.data.qpos[: self.nq_hand] = hand_qpos
        self.data.ctrl[:] = self.qpos2ctrl(self.data.qpos)
        # print(f"hand_qpos: {hand_qpos}")
        # print(f"ctrl: {self.data.ctrl}")
        mujoco.mj_forward(self.model, self.data)

    def set_obj_pose(self, obj_pose: np.ndarray):
        """
        Set object's 7-dof pose in last 7 qpos entries.
        pose7: [x,y,z,qw,qx,qy,qz]
        """
        if self.object_fixed:
            print(
                "This MjHO instance has no object to set pose for, because object is fixed."
            )
            return
        if obj_pose.shape[0] != 7:
            raise ValueError("pose7 must be length 7")
        self.data.qpos[-7:] = obj_pose
        mujoco.mj_forward(self.model, self.data)

    def get_hand_qpos(self, copy: bool = True) -> np.ndarray:
        """Return hand's qpos (first self.hand_qpos_len entries)."""
        hand_qpos = self.data.qpos[: self.nq_hand]
        return np.array(hand_qpos, copy=True) if copy else hand_qpos

    def get_obj_pose(self) -> np.ndarray:
        """Return object's 7-dof qpos (last 7 entries)."""
        if self.object_fixed:
            return np.array([0, 0, 0, 1, 0, 0, 0])
        else:
            return np.array(self.data.qpos[-7:])

    def build_squeeze_qpos(
        self,
        qpos_grasp: np.ndarray,
        grip_delta: float = 0.05,
    ) -> np.ndarray:
        """Return a hand qpos with extra squeeze applied to non-side-swing joints."""
        hand_qpos = np.asarray(qpos_grasp, dtype=float).reshape(-1).copy()
        if hand_qpos.shape[0] == self.nq:
            hand_qpos = hand_qpos[: self.nq_hand].copy()
        elif hand_qpos.shape[0] != self.nq_hand:
            raise ValueError(
                f"Unsupported qpos length {hand_qpos.shape[0]} for build_squeeze_qpos."
            )
        for i in range(7, hand_qpos.shape[0]):
            if (i - 7) not in self.side_swing_index_set:
                hand_qpos[i] += float(grip_delta)
        return hand_qpos

    def build_pregrasp_qpos(
        self,
        qpos_target: np.ndarray,
        prepared_joints: np.ndarray,
    ) -> np.ndarray:
        """Return a hand qpos that uses target pose with prepared finger joints."""
        hand_qpos = np.asarray(qpos_target, dtype=float).reshape(-1).copy()
        if hand_qpos.shape[0] == self.nq:
            hand_qpos = hand_qpos[: self.nq_hand].copy()
        elif hand_qpos.shape[0] != self.nq_hand:
            raise ValueError(
                f"Unsupported qpos length {hand_qpos.shape[0]} for build_pregrasp_qpos."
            )

        prepared_joints = np.asarray(prepared_joints, dtype=float).reshape(-1).copy()
        expected_joint_dim = self.nq_hand - 7
        if prepared_joints.shape[0] != expected_joint_dim:
            raise ValueError(
                f"prepared_joints length {prepared_joints.shape[0]} != expected {expected_joint_dim}."
            )

        hand_qpos[7:] = prepared_joints
        return hand_qpos

    def get_obj_mesh(self):
        # TODO
        pass

    def qpos2ctrl(self, qpos):
        q = np.asarray(qpos, dtype=float).reshape(-1)
        if q.shape[0] == self.nq:
            hand_qpos = q[: self.nq_hand]
        elif q.shape[0] == self.nq_hand:
            hand_qpos = q
        else:
            raise ValueError(f"Unsupported qpos length {q.shape[0]} for qpos2ctrl.")
        ctrl = np.take(hand_qpos, self.ctrl_qpos_indices)
        if ctrl.shape[0] != self.nu:
            raise ValueError(
                f"ctrl length {ctrl.shape[0]} != model.nu {self.nu} for hand '{self.hand_name}'."
            )
        return ctrl
            

    def step(self, n_steps: int = 1, ctrl: Optional[np.ndarray] = None):
        """
        Advance simulation by n_steps. If ctrl provided, it will be assigned to data.ctrl first.
        ctrl length must match model.nu.
        """
        if ctrl is not None:
            ctrl = np.asarray(ctrl, dtype=float)
            if ctrl.shape[0] != self.nu:
                raise ValueError(f"ctrl length {ctrl.shape[0]} != model.nu {self.nu}")
            self.data.ctrl[:] = ctrl

        for _ in range(int(n_steps)):
            mujoco.mj_step(self.model, self.data)

    def sim_grasp(
        self,
        Mp: int = 20,
        steps: int = 40,
        speed_gain: float = 1.5,
        max_tip_speed: float = 0.05,
        target_point_method: int = 2,
        visualize: bool = False,
        record_history: bool = False,
    ):
        # --- sanity checks ---
        if self.anchor_body_names is None:
            raise ValueError("hand.anchor_params must be specified in constructor")
        method_id = int(target_point_method)
        if method_id not in {1, 2, 3}:
            raise ValueError(
                f"sim_grasp.target_point_method must be one of [1, 2, 3], got {target_point_method}."
            )

        history = (
            {
                "qpos": [],
                "anchor_positions": [],
                "tip_positions": [],
                "pts_top_Mp": [],
                "pts_target": [],
            }
            if record_history
            else {}
        )

        # Main loop
        for step_i in range(int(steps)):
            # read current qpos and tip positions (no modification)
            hand_qpos = self.get_hand_qpos(copy=True)

            anchor_positions = np.vstack(
                [np.array(self.data.xpos[bid]).copy() for bid in self.anchor_body_ids]
            )  # (n_anchors,3)

            # containers for flattening per step
            all_top_Mp_pts = []  # will collect shape (n_anchors*Mp,3)
            step_targets = []  # (n_anchors,3)

            # per-anchor-body loop
            for ti, bid in enumerate(self.anchor_body_ids):
                anchor_pos = anchor_positions[ti]
                xmat = np.array(self.data.xmat[bid]).reshape(3, 3)
                z_body = xmat[:, 2].copy()
                z_body = z_body / (np.linalg.norm(z_body) + 1e-12)

                if method_id == 1:
                    _, idx_near = self.obj_tree.query(anchor_pos, k=Mp)
                    idx_near = np.atleast_1d(np.asarray(idx_near, dtype=int))
                    pts_top_Mp = self.obj_pts[idx_near].reshape(-1, 3).copy()
                    pts_target = np.mean(pts_top_Mp, axis=0)
                elif method_id == 2:
                    # Original plane-first target selection: choose object points nearest to
                    # the anchor body's local z-plane, then pick the Euclidean-nearest one.
                    r_all = self.obj_pts - anchor_pos[None, :]
                    abs_signed = np.abs(r_all.dot(z_body))
                    if Mp == 1:
                        top_idx_by_plane = np.array([int(np.argmin(abs_signed))])
                    else:
                        top_idx_by_plane = np.argpartition(abs_signed, Mp - 1)[:Mp]
                        top_idx_by_plane = top_idx_by_plane[
                            np.argsort(abs_signed[top_idx_by_plane])
                        ]
                    pts_top_Mp = self.obj_pts[top_idx_by_plane]
                    dists = np.linalg.norm(pts_top_Mp - anchor_pos[None, :], axis=1)
                    chosen_local = int(np.argmin(dists))
                    pts_target = pts_top_Mp[chosen_local].copy()
                else:
                    # Euclidean-first variant: choose nearest object points, then pick
                    # the candidate closest to the anchor body's local z-plane.
                    _, idx_near = self.obj_tree.query(anchor_pos, k=Mp)
                    idx_near = np.atleast_1d(np.asarray(idx_near, dtype=int))
                    pts_top_Mp = self.obj_pts[idx_near].reshape(-1, 3).copy()
                    abs_plane_dist = np.abs((pts_top_Mp - anchor_pos[None, :]).dot(z_body))
                    idx_local = int(np.argmin(abs_plane_dist))
                    pts_target = pts_top_Mp[idx_local].copy()

                # # Collect for flatten history
                step_targets.append(pts_target)
                all_top_Mp_pts.append(pts_top_Mp)

            # flatten and combine per-step arrays
            all_top_Mp_pts = np.vstack(all_top_Mp_pts)  # (n_anchors*Mp, 3)
            step_targets = np.vstack(step_targets)  # (n_anchors, 3)

            # store history (flattened as requested)
            if record_history:
                history["qpos"].append(hand_qpos.copy())  # (nq_hand,)
                history["anchor_positions"].append(anchor_positions.copy())  # (n_anchors,3)
                history["tip_positions"].append(anchor_positions.copy())  # backward compatibility
                history["pts_top_Mp"].append(all_top_Mp_pts.copy())  # (n_anchors*Mp, 3)
                history["pts_target"].append(step_targets.copy())  # (n_anchors, 3)

            # print(f"step_i: {step_i}")
            # print(f"hand_qpos: {hand_qpos}")
            # print(f"tip_positions: {tip_positions}")

            # construct desired anchor velocities (world frame)
            v_anchors = speed_gain * (step_targets - anchor_positions)  # (n_anchors, 3)
            # clip per-anchor speed
            norms = np.linalg.norm(v_anchors, axis=1)
            for ti in range(self.n_anchors):
                if norms[ti] > max_tip_speed:
                    v_anchors[ti] = (v_anchors[ti] / norms[ti]) * max_tip_speed

            # print(f"v_anchors: {v_anchors}")
            # print(f"v_anchors.shape: {v_anchors.shape}")

            # For each anchor body: compute jacp (3 x nv), then extract hand columns and solve.
            total_dq_hand = np.zeros(
                (self.nq_hand - 7), dtype=float
            )  # accumulate contributions
            # prepare jac arrays (width model.nv)
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            for ti, bid in enumerate(self.anchor_body_ids):
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(bid))
                # print(f"jacp: {jacp}")
                # print(f"jacp.shape: {jacp.shape}")

                # extract hand columns
                J_hand = jacp[:, 6 : 6 + self.nq_hand]  # (3, n_hand_dofs)
                # print(f"J_hand: {J_hand}")
                # print(f"J_hand.shape: {J_hand.shape}")

                # solve least squares J_hand @ dq = v_anchor
                v_des = v_anchors[ti]
                # If Jacobian is rank-deficient or rectangular, use lstsq
                # dq_sol, *_ = np.linalg.lstsq(J_hand, v_des, rcond=None)
                # dq_sol = np.asarray(dq_sol).reshape(-1)
                dq_sol = self.anchor_weights[ti] * np.linalg.pinv(J_hand) @ v_des

                # print(self.anchor_weights[ti])

                # accumulate (simple sum — you may weight by finger importance)
                total_dq_hand += dq_sol
                # print(f"dq_sol: {dq_sol}")

            # exit()

            # now add to hand_qpos
            mask = ~np.isin(np.arange(total_dq_hand.shape[0]), self.side_swing_indices)
            total_dq_hand[mask] = np.maximum(total_dq_hand[mask], 0)
            # for idx in special_indices:
            #     total_dq_hand[idx] /= 3

            for idx in self.thumb_relax_indices:
                if 0 <= idx < total_dq_hand.shape[0]:
                    total_dq_hand[idx] /= max(self.thumb_relax_divisor, 1e-6)
            # print(f"total_dq_hand: {total_dq_hand}")
            # print(f"total_dq_hand.shape: {total_dq_hand.shape}")

            hand_qpos[7 : 7 + self.nq_hand] += total_dq_hand

            # step the simulation by 1 (ctrl already set by set_hand_qpos, but ensure we step with it)
            ctrl = self.qpos2ctrl(hand_qpos)
            self.step(1, ctrl=ctrl)

            if visualize:
                self._render_viewer()
                # time.sleep(0.1)
        # if visualize:
        #     while self.viewer.is_alive:
        #         self.viewer.render()

        # after loop: collect final
        final_hand_qpos = self.get_hand_qpos(copy=True)
        return final_hand_qpos, history

    def get_contact_num(self, obj_margin=0.0) -> int:
        for geom_id in self.obj_collision_geom_indices:
            self.model.geom_margin[geom_id] = obj_margin
            self.model.geom_gap[geom_id] = obj_margin

        ho_contact_count = 0
        for contact_i in range(self.data.ncon):
            contact = self.data.contact[contact_i]
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            if (
                body1_id > self.world_body_id
                and body1_id < self.hand_body_upper_id
                and body2_id == self.object_body_id
            ) or (
                body2_id > self.world_body_id
                and body2_id < self.hand_body_upper_id
                and body1_id == self.object_body_id
            ):
                ho_contact_count += 1

        for geom_id in self.obj_collision_geom_indices:
            self.model.geom_margin[geom_id] = obj_margin
            self.model.geom_gap[geom_id] = obj_margin
        return ho_contact_count

    def get_contact_info(self, obj_margin=0.0):
        # for body_id in range(self.model.nbody):
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        #     print(f"body_id: {body_id} name: {body_name}")
        # print("-" * 40)

        # Set margin and gap
        for geom_id in self.obj_collision_geom_indices:
            self.model.geom_margin[geom_id] = self.model.geom_gap[geom_id] = obj_margin

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact_i in range(self.data.ncon):
            contact = self.data.contact[contact_i]
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (
                body1_id > self.world_body_id
                and body1_id < self.hand_body_upper_id
                and body2_id == self.object_body_id
            ) or (
                body2_id > self.world_body_id
                and body2_id < self.hand_body_upper_id
                and body1_id == self.object_body_id
            ):
                # keep body1=hand and body2=object
                if body2_id == self.object_body_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif (
                body1_id > self.world_body_id
                and body1_id < self.hand_body_upper_id
                and body2_id > self.world_body_id
                and body2_id < self.hand_body_upper_id
            ):
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )
            else:
                print(body1_name, body2_name, body1_id, body2_id)

        # Set margin and gap back
        for geom_id in self.obj_collision_geom_indices:
            self.model.geom_margin[geom_id] = self.model.geom_gap[geom_id] = obj_margin
        return ho_contact, hh_contact

    def is_contact(self):
        has_collision = len(self.data.contact) > 0
        return has_collision

    def get_pose_delta(self, pose1, pose2):
        # pose: [x, y, z, qw, qx, qy, qz]
        delta_pos = np.linalg.norm(pose1[:3] - pose2[:3])  # (1)
        q1_inv = tq.qinverse(pose1[3:])
        q_rel = tq.qmult(pose2[3:], q1_inv)
        if np.abs(q_rel[0]) > 1:
            q_rel[0] = 1
        angle = 2 * np.arccos(q_rel[0])
        angle_degrees = np.degrees(angle)
        return delta_pos, angle_degrees

    def sim_under_extforce(
        self,
        qpos_target: np.ndarray,
        qpos_prepared: np.ndarray,
        duration: float = 1.0,
        trans_thresh: float = 0.05,
        angle_thresh: float = 10.0,
        force_mag: float = 1.0,
        check_steps: int = 50,
        close_steps: int = 100,
        visualize: bool = False,
    ) -> bool:

        if self.object_fixed:
            raise RuntimeError(
                "Object is fixed in this MjHO instance; cannot run extforce validation."
            )
        qpos_target = np.asarray(qpos_target, dtype=float).reshape(-1).copy()
        if qpos_target.shape[0] == self.nq:
            qpos_target = qpos_target[: self.nq_hand].copy()
        elif qpos_target.shape[0] != self.nq_hand:
            raise ValueError(
                f"Unsupported qpos length {qpos_target.shape[0]} for sim_under_extforce."
            )

        qpos_prepared = np.asarray(qpos_prepared, dtype=float).reshape(-1).copy()
        if qpos_prepared.shape[0] == self.nq:
            qpos_prepared = qpos_prepared[: self.nq_hand].copy()
        elif qpos_prepared.shape[0] != self.nq_hand:
            raise ValueError(
                f"Unsupported prepared qpos length {qpos_prepared.shape[0]} for sim_under_extforce."
            )

        hand_ctrl = self.qpos2ctrl(qpos_target)

        # which body id is object?
        obj_body_id = int(self.model.nbody - 1)

        # six external force directions (force only; no moment)
        external_force_dirs = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # compute number of simulation steps for duration (use model timestep)
        dt = float(self.model.opt.timestep)
        n_steps = int(duration / dt)
        check_steps = max(int(check_steps), 1)
        n_chunks = n_steps // check_steps
        close_steps = max(int(close_steps), 1)

        # perform tests for each direction
        for dir_vec in external_force_dirs:
            self.reset()
            self.set_hand_qpos(qpos_prepared)
            # Close from prepared joints to target joints before applying external forces.
            initial_obj_pose = self.get_obj_pose().copy()  # [x,y,z,qw,qx,qy,qz]
            self.step(close_steps, ctrl=hand_ctrl)
            if visualize:
                self._render_viewer()
                time.sleep(0.003)

            if not self.is_contact():
                if visualize:
                    print("Object lost contact during settling phase.")
                return False, np.inf, np.inf
            settle_pos_delta, settle_angle_delta = self.get_pose_delta(
                initial_obj_pose, self.get_obj_pose()
            )
            if (settle_pos_delta >= (trans_thresh)) or (
                settle_angle_delta >= (angle_thresh)
            ):
                if visualize:
                    print(f"Object moved too much during settling phase: {settle_pos_delta:.6f}, {settle_angle_delta:.6f}")
                return False, settle_pos_delta, settle_angle_delta
            settled_obj_pose = self.get_obj_pose().copy()
            # print(f"Testing direction: {dir_vec}")
            # print(f"obj pos after reset: {self.get_obj_pose()}")

            # scale direction by force magnitude (only first 3 entries are force)
            applied = np.zeros(6, dtype=float)
            applied[:3] = dir_vec[:3] * force_mag

            # set external wrench on object body
            self.data.xfrc_applied[obj_body_id] = applied

            # apply force for n_steps
            for chunk_i in range(n_chunks):
                # step simulation forward by one mj_step
                self.step(check_steps, ctrl=hand_ctrl)

                if visualize:
                    self._render_viewer()
                    time.sleep(0.003)

                # check if object has moved
                if not self.is_contact():
                    if visualize:
                        print("Object lost contact during force application.")
                    return False, np.inf, np.inf
            
                pos_delta, angle_delta = self.get_pose_delta(
                    settled_obj_pose, self.get_obj_pose()
                )
                succ_flag = (pos_delta < trans_thresh) & (
                    angle_delta < angle_thresh
                )
                # print(f"Step {step_i}: {succ_flag}, {pos_delta}, {angle_delta}")
                if not succ_flag:
                    if visualize:
                        print(f"Object moved too much during force application: {pos_delta:.6f}, {angle_delta:.6f}")
                    return False, pos_delta, angle_delta

            self.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)

        # all directions passed
        self.data.xfrc_applied[obj_body_id] = np.zeros(6, dtype=float)

        # compute translation and rotation deltas
        pos_delta, angle_delta = self.get_pose_delta(
            settled_obj_pose, self.get_obj_pose()
        )
        succ_flag = (pos_delta < trans_thresh) & (angle_delta < angle_thresh)

        return succ_flag, pos_delta, angle_delta

    def viewer_loop(self):
        """Launch simple viewer loop with official MuJoCo viewer."""
        try:
            while self._viewer_alive():
                self.step()
                self._render_viewer()
        except KeyboardInterrupt:
            try:
                self.viewer.close()
            except Exception as e:
                print(e)
            return


class RobotKinematics:
    def __init__(self, xml_path: str):
        """
        Load MJCF, compile, collect visual and collision meshes.
        Simple rules:
          - geom.contype == 0 -> visual
          - geom.contype != 0 -> collision
            - if mesh descriptor has convex-graph info -> collision = convex_hull(original)
              and visual = original (store both)
        """
        self.spec = mujoco.MjSpec.from_file(xml_path)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        # store entries as dicts: key -> { vert, face, geom_id, body, mesh_name }
        self.visual_meshes: Dict[str, Dict] = {}
        self.collision_meshes: Dict[str, Dict] = {}

        for i in range(self.model.ngeom):
            geom = self.model.geom(i)
            mesh_id = geom.dataid[0] if hasattr(geom, "dataid") else -1
            if mesh_id == -1:
                # no mesh for this geom -> skip
                continue

            mjm = self.model.mesh(mesh_id)

            # robustly extract indices (support scalar or 1-element arrays)
            try:
                vertadr = (
                    int(mjm.vertadr[0])
                    if hasattr(mjm.vertadr, "__len__")
                    else int(mjm.vertadr)
                )
                vertnum = (
                    int(mjm.vertnum[0])
                    if hasattr(mjm.vertnum, "__len__")
                    else int(mjm.vertnum)
                )
                faceadr = (
                    int(mjm.faceadr[0])
                    if hasattr(mjm.faceadr, "__len__")
                    else int(mjm.faceadr)
                )
                facenum = (
                    int(mjm.facenum[0])
                    if hasattr(mjm.facenum, "__len__")
                    else int(mjm.facenum)
                )
            except Exception:
                # can't parse mesh buffers -> skip
                continue

            verts = np.array(self.model.mesh_vert[vertadr : vertadr + vertnum])
            faces = np.array(self.model.mesh_face[faceadr : faceadr + facenum]).astype(
                int
            )

            rgba = None
            try:
                # 优先使用 geom.rgba（如果 xml 中直接在 geom 上指定了 rgba）
                if hasattr(geom, "rgba"):
                    rgba_attr = geom.rgba
                    rgba = np.array(rgba_attr).flatten()[:4]
                else:
                    # 退回到 material rgba（如果 geom 指定了 matid）
                    matid = geom.matid[0] if hasattr(geom, "matid") else -1
                    if matid is not None and matid >= 0 and hasattr(self.model, "mat_rgba"):
                        rgba = np.array(self.model.mat_rgba[matid]).flatten()[:4]
            except Exception:
                rgba = None

            body_name = (
                self.model.body(geom.bodyid[0]).name
                if geom.bodyid[0] >= 0
                else f"body_{geom.bodyid[0]}"
            )
            mesh_name = mjm.name if hasattr(mjm, "name") else f"mesh_{mesh_id}"

            key_base = f"{body_name}_{mesh_name}"
            entry_orig = {
                "vert": verts,
                "face": faces,
                "color": rgba,
                "geom_id": i,
                "body": body_name,
                "mesh_name": mesh_name,
            }

            # classify: contype==0 -> visual ; else -> collision
            if int(geom.contype[0]) == 0:
                # visual geometry
                self.visual_meshes[f"{key_base}_{i}"] = entry_orig
            else:
                # collision geometry: check for convex-graph info on the mesh descriptor
                # Try several attribute names that may exist on mjm
                graph_attr = None
                if hasattr(mjm, "graphadr"):
                    graph_attr = mjm.graphadr
                elif hasattr(mjm, "mesh_graphadr"):
                    graph_attr = mjm.mesh_graphadr
                elif hasattr(mjm, "graph_adr"):
                    graph_attr = mjm.graph_adr

                has_convex_graph = False
                try:
                    if graph_attr is not None:
                        # if it's array-like, check first element
                        first = (
                            graph_attr[0]
                            if hasattr(graph_attr, "__len__")
                            else graph_attr
                        )
                        if int(first) != -1:
                            has_convex_graph = True
                except Exception:
                    has_convex_graph = False

                if has_convex_graph:
                    # collision should store convex hull; visual should keep original
                    # compute convex hull using trimesh (simple and robust)
                    try:
                        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                        hull = tm.convex_hull
                        if hull.vertices.shape[0] > 0 and hull.faces.shape[0] > 0:
                            entry_hull = {
                                "vert": np.asarray(hull.vertices),
                                "face": np.asarray(hull.faces),
                                "color": entry_orig.get("color"),
                                "geom_id": i,
                                "body": body_name,
                                "mesh_name": mesh_name + "_hull",
                            }
                            self.collision_meshes[f"{key_base}_hull_{i}"] = entry_hull
                        else:
                            # fallback to original if hull degenerate
                            self.collision_meshes[f"{key_base}_{i}"] = entry_orig
                    except Exception:
                        # on any failure, keep original as collision
                        self.collision_meshes[f"{key_base}_{i}"] = entry_orig

                    # also ensure visual stores original (so visual and collision differ)
                    self.visual_meshes.setdefault(f"{key_base}_orig_{i}", entry_orig)
                else:
                    # no convex-graph info -> use original as collision
                    self.collision_meshes[f"{key_base}_{i}"] = entry_orig

    def forward_kinematics(self, qpos: np.ndarray):
        """Set qpos and run mujoco kinematics (mj_kinematics)."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return

    def _pose_vertices(self, verts: np.ndarray, geom_id: int) -> np.ndarray:
        """Pose vertices (Vx3) into world coordinates using data.geom_xmat and data.geom_xpos."""
        if verts is None or verts.size == 0:
            return np.zeros((0, 3), dtype=float)
        xpos = np.array(self.data.geom_xpos[geom_id])
        xmat = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)
        posed = (verts @ xmat.T) + xpos
        return posed

    def get_posed_meshes(
        self, qpos: np.ndarray, kind: str = "visual"
    ) -> Optional[trimesh.Trimesh]:
        """
        Return a concatenated trimesh of posed geometries.
        kind: "visual" | "collision"
        """
        assert kind in ("visual", "collision")
        self.forward_kinematics(qpos)

        pieces = []

        if kind == "visual":
            for ent in self.visual_meshes.values():
                geom_id = int(ent["geom_id"])
                posed = self._pose_vertices(ent["vert"], geom_id)
                if posed.size == 0 or ent["face"].size == 0:
                    continue

                tm = trimesh.Trimesh(vertices=posed, faces=ent["face"], process=False)
                col = ent.get("color")
                if col is not None:
                    # 确保形状是 (1,4)，然后重复到每个顶点
                    col_arr = np.asarray(col, dtype=float).reshape(1, 4)
                    tm.visual.vertex_colors = np.tile(col_arr, (tm.vertices.shape[0], 1))
                pieces.append(tm)

        elif kind == "collision":
            for ent in self.collision_meshes.values():
                geom_id = int(ent["geom_id"])
                posed = self._pose_vertices(ent["vert"], geom_id)
                if posed.size == 0 or ent["face"].size == 0:
                    continue
                
                tm = trimesh.Trimesh(vertices=posed, faces=ent["face"], process=False)
                col = ent.get("color")
                if col is not None:
                    # 确保形状是 (1,4)，然后重复到每个顶点
                    col_arr = np.asarray(col, dtype=float).reshape(1, 4)
                    tm.visual.vertex_colors = np.tile(col_arr, (tm.vertices.shape[0], 1))
                pieces.append(tm)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if not pieces:
            return None

        combined = trimesh.util.concatenate(pieces)
        return combined


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../assets/hands/liberhand/liberhand_right.xml"
    )
    rk = RobotKinematics(xml_path)
    q = np.array(
        [
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.5,
            0.5,
            0.5,
            0,
            0.3,
            0.3,
            0.3,
            0,
            0.3,
            0.3,
            0.3,
            1.6,
            0.0,
            0.3,
            0.3,
        ]
    )
    # q = np.zeros((20))
    qpos = np.concatenate((np.array([0, 0, 0, 1, 0, 0, 0]), q))
    rk.forward_kinematics(qpos)
    vis = rk.get_posed_meshes(qpos, kind="visual")
    if vis is not None:
        vis.export("debug_hand_visual.obj")
    col = rk.get_posed_meshes(qpos, kind="collision")
    if col is not None:
        col.export("debug_hand_collision.obj")
