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
import mujoco_viewer
import transforms3d.quaternions as tq
from typing import Dict, List, Tuple, Optional, Sequence


class MjHO:
    def __init__(
        self,
        obj_info: Dict,
        hand_xml_path: str,
        target_body_params: Optional[Dict] = None,
        friction_coef: Sequence[float] = (0.2, 0.2),
        object_fixed: bool = True,
    ):
        """
        Args:
            obj_info: dict returned by DatasetObjects.get_info(name)
            hand_xml_path: path to hand MJCF (used as base)
            friction_coef: scalar or sequence (len 2 or 3) for friction
            object_fixed: if True, weld object to world (no DOFs)
        """
        self.obj_info = obj_info
        self.hand_xml_path = os.path.abspath(hand_xml_path)
        self.friction_coef = friction_coef
        self.object_fixed = object_fixed

        # load hand spec as base
        self.spec = self._add_hand(self.hand_xml_path)

        # add object assets/geoms
        self._add_object(self.obj_info, fixed=self.object_fixed)

        # apply friction to object collision geoms
        self._set_friction(self.friction_coef)

        # set solimp and solref
        self._set_sol()

        # compile into model/data
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        # self._set_margin_and_gap(0.001, 0.0005)

        # indices and sizes
        self.nq = self.model.nq
        self.nu = self.model.nu

        # hand qpos length = total nq minus 7 if object exists
        self.nq_hand = self.nq - 7 if not self.object_fixed else self.nq


        # build mapping: fingertip body names -> body ids
        self.body_name_to_id = {}
        for bi in range(self.model.nbody):
            bname = self.model.body(bi).name
            if bname is not None:
                self.body_name_to_id[str(bname)] = bi

        self.target_bodies = None
        self.target_body_weights = None
        if target_body_params is not None:

            self.fingertip_bodies = list(target_body_params.keys())
            self.target_body_weights = np.array(list(target_body_params.values()))[:,0]
            self.target_body_dis_weights = np.array(list(target_body_params.values()))[:,1]
            self.tip_body_ids = []
            for bname in self.fingertip_bodies:
                if bname not in self.body_name_to_id:
                    raise ValueError(f"Body name '{bname}' not found in model bodies")
                self.tip_body_ids.append(self.body_name_to_id[bname])
            self.n_tips = len(self.tip_body_ids)

            print(f"fingertip_bodies: {self.fingertip_bodies}")
            print(f"target_body_weights: {self.target_body_weights}")
            print(f"target_body_dis_weights: {self.target_body_dis_weights}")
            print(f"tip_body_ids: {self.tip_body_ids}")
            print(f"n_tips: {self.n_tips}")
        # exit()
        # reset to sensible default
        self.reset()

    # -----------------------
    # spec construction
    # -----------------------
    def _add_hand(self, hand_xml_path: str) -> mujoco.MjSpec:
        """Load hand MJCF and return its MjSpec to be used as base spec."""
        hand_spec = mujoco.MjSpec.from_file(hand_xml_path)
        # adjust meshdir so relative mesh paths resolve relative to hand xml
        hand_spec.meshdir = os.path.dirname(hand_xml_path)

        # for g in hand_spec.geoms:
        #     # This solimp and solref comes from the Shadow Hand xml
        #     # They can generate larger force with smaller penetration
        #     # The body will be more "rigid" and less "soft"
        #     g.solimp[:3] = [0.5, 0.99, 0.0001]
        #     g.solref[:2] = [0.005, 1]

        return hand_spec

    def _add_object(self, obj_info: Dict, fixed: bool = True):
        """
        Minimal: merge object's MjSpec assets (meshes) and worldbody into self.spec.

        - obj_info['xml_path'] should point to the object's MJCF/XML produced by preprocessing.
        - Only mesh assets and worldbody are merged (no texture/material copying).
        - If fixed==True, add a weld equality between 'world' and the object's root body.
        """
        obj_name = obj_info.get("name")
        obj_xml = obj_info.get("xml_abs")
        obj_dir = os.path.dirname(obj_xml)
        obj_spec = mujoco.MjSpec.from_file(obj_xml)
        self.obj_info = obj_info
        self.obj_name = obj_name
        print(f"loading object xml: {obj_xml}")
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

        attach_frame = self.spec.worldbody.add_frame()
        self.spec.attach(obj_spec, frame=attach_frame, prefix="")

        if fixed:
            # delete free joints from object
            for j in self.spec.joints:
                # print(f"joint {j.name} ")
                if j.name == obj_name + "_joint":
                    print(f"joint {j.name} is fixed")
                    # self.spec.joints.remove(j)
                    # j.type = mujoco.mjtJoint.mjJNT_WELD
                    self.spec.delete(j)
                    break

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return
    
    def _set_margin_and_gap(self, obj_margin, obj_gap):
        for i in range(self.model.ngeom):
                if f"{self.obj_name}_collision" in self.model.geom(i).name:
                    self.model.geom_margin[i] = obj_margin
                    self.model.geom_gap[i] = obj_gap

    def _set_sol(self):
        for g in self.spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

            # g.solimp[:3] = [0.9, 0.95, 0.001]
            # g.solref[:2] = [0.02, 1]

            # g.solimp[:3] = [0.5, 0.99, 0.001]
            # g.solref[:2] = [0.01, 1]


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
        self.data.qpos[:7] = np.array([-0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

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
        self.data.ctrl = self.qpos2ctrl(self.data.qpos)
        # print(f"hand_qpos: {hand_qpos}")
        # print(f"ctrl: {self.data.ctrl}")
        mujoco.mj_forward(self.model, self.data)

    def set_obj_pose(self, obj_pose: np.ndarray):
        """
        Set object's 7-dof pose in last 7 qpos entries.
        pose7: [x,y,z,qw,qx,qy,qz]
        """
        if self.object_fixed:
            print("This MjHO instance has no object to set pose for, because object is fixed.")
            return
        if obj_pose.shape[0] != 7:
            raise ValueError("pose7 must be length 7")
        self.data.qpos[-7:] = obj_pose
        mujoco.mj_forward(self.model, self.data)

    def get_hand_qpos(self) -> np.ndarray:
        """Return hand's qpos (first self.hand_qpos_len entries)."""
        return np.array(self.data.qpos[: self.nq_hand])

    def get_obj_pose(self) -> np.ndarray:
        """Return object's 7-dof qpos (last 7 entries)."""
        if self.object_fixed:
            return np.array([0, 0, 0, 1, 0, 0, 0])
        else:
            return np.array(self.data.qpos[-7:])

    def get_obj_mesh(self):
        # TODO
        pass

    def qpos2ctrl(self, qpos):
        # 7 + (0,1,2) + (4,5,6) + (8,9) + (12, 13) + (16, 17, 18)
        ctrl = np.zeros(self.nu)
        ctrl[0:3] = qpos[7+0:7+3]
        ctrl[3:6] = qpos[7+4:7+7]
        ctrl[6:8] = qpos[7+8:7+10]
        ctrl[8:10] = qpos[7+12:7+14]
        ctrl[10:13] = qpos[7+16:7+19]
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
        obj_pts: np.ndarray,
        obj_norms: np.ndarray,
        Mp: int = 5,
        Mn: int = 20,
        steps: int = 40,
        speed_gain: float = 1.0,
        max_tip_speed: float = 0.03,
        visualize: bool = True,
    ):
        # --- sanity checks ---
        if self.fingertip_bodies is None:
            raise ValueError("fingertip_bodies must be specified in constructor")

        history = {
            "qpos": [],
            "tip_positions": [],
            "pts_top_Mp": [],
            "pts_top_Mn": [],
            "pts_target": [],
        }
        if visualize:
            viewer = self.viewer_loop(blocking=False)

        # Main loop
        for step_i in range(int(steps)):
            # read current qpos and tip positions (no modification)
            hand_qpos = self.get_hand_qpos().copy()

            tip_positions = np.vstack([np.array(self.data.xpos[bid]).copy() for bid in self.tip_body_ids])  # (n_tips,3)

            # containers for flattening per step
            all_top_Mp_pts = []   # will collect shape (n_tips*Mp,3)
            all_top_Mn_pts = []   # will collect shape (n_tips*Mp,3)
            step_targets = []     # (n_tips,3)

            # per-target-body loop
            for ti, bid in enumerate(self.tip_body_ids):
                tip_pos = tip_positions[ti]

                # --- top Mp by global nearest ---
                d_near, idx_near = self.obj_tree.query(tip_pos, k=Mp)
                idx_near = np.asarray(idx_near, dtype=int)
                d_near = np.asarray(d_near, dtype=float)
                pts_top_Mp = self.obj_pts[idx_near].copy()  # (Mp,3)

                # --- alignment ranking then pick Mp nearest among top Mn ---
                # compute body y-axis using data.xmat
                xmat = np.array(self.data.xmat[bid]).reshape(3, 3)
                y_body = xmat[:, 1]
                y_body = y_body / (np.linalg.norm(y_body) + 1e-12)

                aligns = np.dot(obj_norms, (-y_body))  # (N,)
                aligns = np.clip(aligns, -1.0, 1.0)

                # top Mn indices by alignment descending
                topMn_idx_part = np.argpartition(-aligns, Mn - 1)[:Mn]
                # sort those Mn by alignment descending for determinism
                topMn_idx_sorted = topMn_idx_part[np.argsort(-aligns[topMn_idx_part])]

                # among these Mn, choose Mp nearest to tip_pos
                pts_candidates = self.obj_pts[topMn_idx_sorted].copy()  # (Mn, 3)
                d_cand = np.linalg.norm(pts_candidates - tip_pos[None, :], axis=1)
                take_k = min(Mp, pts_candidates.shape[0])
                order_k = np.argsort(d_cand)[:take_k]
                chosen_idx_from_Mn = topMn_idx_sorted[order_k]  # actual indices into obj_pts
                pts_top_Mn = obj_pts[chosen_idx_from_Mn]  # (Mp, 3) (or fewer if Mn < Mp)

                # Collect for flatten history
                all_top_Mp_pts.append(pts_top_Mp)
                all_top_Mn_pts.append(pts_top_Mn)

                # Compute uniform average of concatenated points -> target for this body
                # concat_pts = np.vstack([pts_top_Mp, pts_top_Mn])  # (2*Mp,3)
                # pts_target = np.mean(concat_pts, axis=0)          # (3,)
                lam = self.target_body_dis_weights[ti]
                pts_target = lam * np.mean(pts_top_Mp, axis=0) + (1-lam) * np.mean(pts_top_Mn, axis=0)

                step_targets.append(pts_target)
            
            # flatten and combine per-step arrays
            all_top_Mp_pts = np.vstack(all_top_Mp_pts)  # (n_tips*Mp, 3)
            all_top_Mn_pts = np.vstack(all_top_Mn_pts)  # (n_tips*Mp, 3)
            step_targets = np.vstack(step_targets)      # (n_tips, 3)

            # store history (flattened as requested)
            history["qpos"].append(hand_qpos.copy())                   # (nq_hand,)
            history["tip_positions"].append(tip_positions.copy())      # (n_tips,3)
            
            history["pts_top_Mp"].append(all_top_Mp_pts.copy())        # (n_tips*Mp, 3)
            history["pts_top_Mn"].append(all_top_Mn_pts.copy())        # (n_tips*Mp, 3)
            history["pts_target"].append(step_targets.copy())          # (n_tips, 3)

            # print(f"step_i: {step_i}")
            # print(f"hand_qpos: {hand_qpos}")
            # print(f"tip_positions: {tip_positions}")

            # construct desired tip velocities (world frame)
            v_tips = speed_gain * (step_targets - tip_positions)  # (n_tips, 3)
            # clip per-tip speed
            norms = np.linalg.norm(v_tips, axis=1)
            for ti in range(self.n_tips):
                if norms[ti] > max_tip_speed:
                    v_tips[ti] = (v_tips[ti] / norms[ti]) * max_tip_speed

            # print(f"v_tips: {v_tips}")
            # print(f"v_tips.shape: {v_tips.shape}")

            # For each tip: compute jacp (3 x nv) using mj_jacBody, then extract hand columns and solve
            total_dq_hand = np.zeros((self.nq_hand-7), dtype=float)  # accumulate contributions
            # prepare jac arrays (width model.nv)
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            for ti, bid in enumerate(self.tip_body_ids):
                # call mj_jacBody -> fills jacp/jacr for current state in self.data
                # signature: mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(bid))
                # print(f"jacp: {jacp}")
                # print(f"jacp.shape: {jacp.shape}")
                
                # extract hand columns
                J_hand = jacp[:, 6:6+self.nq_hand]  # (3, n_hand_dofs)
                # print(f"J_hand: {J_hand}")
                # print(f"J_hand.shape: {J_hand.shape}")

                # solve least squares J_hand @ dq = v_tip
                v_des = v_tips[ti]
                # If Jacobian is rank-deficient or rectangular, use lstsq
                # dq_sol, *_ = np.linalg.lstsq(J_hand, v_des, rcond=None)
                # dq_sol = np.asarray(dq_sol).reshape(-1)
                dq_sol = self.target_body_weights[ti] * np.linalg.pinv(J_hand) @ v_des

                # print(self.target_body_weights[ti])

                # accumulate (simple sum — you may weight by finger importance)
                total_dq_hand += dq_sol
                # print(f"dq_sol: {dq_sol}")
            
            # now add to hand_qpos
            special_indices = [0, 4, 8, 12, 16]
            mask = ~np.isin(np.arange(total_dq_hand.shape[0]), special_indices)
            total_dq_hand[mask] = np.maximum(total_dq_hand[mask], 0)
            for idx in special_indices:
                total_dq_hand[idx] /= 2
            # print(f"total_dq_hand: {total_dq_hand}")
            # print(f"total_dq_hand.shape: {total_dq_hand.shape}")

            hand_qpos[7:7+self.nq_hand] += total_dq_hand

            # step the simulation by 1 (ctrl already set by set_hand_qpos, but ensure we step with it)
            ctrl = self.qpos2ctrl(hand_qpos)
            self.step(1, ctrl=ctrl)

            if visualize:
                viewer.render()
        if visualize:
            while viewer.is_alive:
                viewer.render()

        # after loop: collect final
        final_hand_qpos = self.get_hand_qpos().copy()
        return final_hand_qpos, history


    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid
            body2_id = self.model.geom(contact.geom2).bodyid
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (
                body1_id > world_id and body1_id < hand_id and body2_id == object_id
            ) or (body2_id > world_id and body2_id < hand_id and body1_id == object_id):
                # keep body1=hand and body2=object
                if body2_id == object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
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
                body1_id > world_id
                and body1_id < hand_id
                and body2_id > world_id
                and body2_id < hand_id
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
            # else:
            #     print(body1_name, body2_name, body1_id, body2_id)

        # Set margin and gap back
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact
    
    def is_contact(self):
        has_collision = len(self.data.contact) > 0
        return has_collision
    

    def viewer_loop(self, blocking: bool = True):
        """Launch simple viewer (MujocoViewer). Blocking if specified."""
        viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        if not blocking:
            return viewer
        try:
            while viewer.is_alive:
                self.step()
                # hand_qpos = self.get_hand_qpos()
                # print(hand_qpos)
                viewer.render()
        except KeyboardInterrupt:
            try:
                viewer.close()
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
                pieces.append(
                    trimesh.Trimesh(vertices=posed, faces=ent["face"], process=False)
                )

        elif kind == "collision":
            for ent in self.collision_meshes.values():
                geom_id = int(ent["geom_id"])
                posed = self._pose_vertices(ent["vert"], geom_id)
                if posed.size == 0 or ent["face"].size == 0:
                    continue
                pieces.append(
                    trimesh.Trimesh(vertices=posed, faces=ent["face"], process=False)
                )
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
