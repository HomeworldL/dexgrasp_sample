from types import SimpleNamespace

import numpy as np

from src.mj_ho import MjHO


def _make_pose(x: float = 0.0, qw: float = 1.0, qz: float = 0.0) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, qw, 0.0, 0.0, qz], dtype=float)


def _make_fake_mjho(poses, contacts):
    mjho = MjHO.__new__(MjHO)
    mjho.object_fixed = False
    mjho.nq = 8
    mjho.nq_hand = 8
    mjho.nu = 1
    mjho.model = SimpleNamespace(nbody=2, opt=SimpleNamespace(timestep=1.0))
    mjho.data = SimpleNamespace(xfrc_applied=np.zeros((2, 6), dtype=float))
    mjho._poses = [np.asarray(p, dtype=float) for p in poses]
    mjho._contacts = [bool(v) for v in contacts]
    mjho._idx = 0

    def reset():
        mjho._idx = 0
        mjho.data.xfrc_applied[:] = 0.0

    def set_hand_qpos(hand_qpos):
        return None

    def qpos2ctrl(qpos):
        return np.zeros((1,), dtype=float)

    def step(n_steps=1, ctrl=None):
        if mjho._idx < (len(mjho._poses) - 1):
            mjho._idx += 1

    def get_obj_pose():
        return mjho._poses[mjho._idx].copy()

    def is_contact():
        return mjho._contacts[mjho._idx]

    mjho.reset = reset
    mjho.set_hand_qpos = set_hand_qpos
    mjho.qpos2ctrl = qpos2ctrl
    mjho.step = step
    mjho.get_obj_pose = get_obj_pose
    mjho.is_contact = is_contact
    return mjho


def test_sim_under_extforce_rejects_large_no_force_settle_drift():
    mjho = _make_fake_mjho(
        poses=[
            _make_pose(0.0),
            _make_pose(0.03),
            _make_pose(0.03),
        ],
        contacts=[True, True, True],
    )

    success, pos_delta, angle_delta = mjho.sim_under_extforce(
        np.zeros((8,), dtype=float),
        duration=1.0,
        trans_thresh=0.05,
        angle_thresh=10.0,
        check_step=1,
    )

    assert bool(success) is False
    assert np.isinf(pos_delta)
    assert np.isinf(angle_delta)


def test_sim_under_extforce_uses_settled_pose_as_force_baseline():
    mjho = _make_fake_mjho(
        poses=[
            _make_pose(0.0),
            _make_pose(0.02),
            _make_pose(0.06),
        ],
        contacts=[True, True, True],
    )

    success, pos_delta, angle_delta = mjho.sim_under_extforce(
        np.zeros((8,), dtype=float),
        duration=1.0,
        trans_thresh=0.05,
        angle_thresh=10.0,
        check_step=1,
    )

    assert bool(success) is True
    assert np.isclose(pos_delta, 0.04)
    assert np.isclose(angle_delta, 0.0)
