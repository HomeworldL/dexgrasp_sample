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
    mjho._ctrl_log = []

    def reset():
        mjho._idx = 0
        mjho.data.xfrc_applied[:] = 0.0

    def set_hand_qpos(hand_qpos):
        return None

    def qpos2ctrl(qpos):
        return np.asarray([float(np.asarray(qpos, dtype=float)[7])], dtype=float)

    def step(n_steps=1, ctrl=None):
        for _ in range(int(n_steps)):
            if ctrl is not None:
                mjho._ctrl_log.append(float(np.asarray(ctrl, dtype=float)[0]))
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
            _make_pose(0.06),
        ],
        contacts=[True, True, True],
    )

    success, pos_delta, angle_delta = mjho.sim_under_extforce(
        np.zeros((8,), dtype=float),
        np.zeros((8,), dtype=float),
        duration=1.0,
        trans_thresh=0.05,
        angle_thresh=10.0,
        check_steps=1,
        close_steps=2,
    )

    assert bool(success) is False
    assert np.isclose(pos_delta, 0.06)
    assert np.isclose(angle_delta, 0.0)


def test_sim_under_extforce_uses_settled_pose_as_force_baseline():
    mjho = _make_fake_mjho(
        poses=[
            _make_pose(0.0),
            _make_pose(0.01),
            _make_pose(0.02),
            _make_pose(0.06),
        ],
        contacts=[True, True, True, True],
    )

    success, pos_delta, angle_delta = mjho.sim_under_extforce(
        np.zeros((8,), dtype=float),
        np.zeros((8,), dtype=float),
        duration=1.0,
        trans_thresh=0.05,
        angle_thresh=10.0,
        check_steps=1,
        close_steps=2,
    )

    assert bool(success) is True
    assert np.isclose(pos_delta, 0.04)
    assert np.isclose(angle_delta, 0.0)


def test_sim_under_extforce_interpolates_close_ctrl_over_close_steps():
    mjho = _make_fake_mjho(
        poses=[_make_pose(0.0)] * 32,
        contacts=[True] * 32,
    )
    qpos_prepared = np.zeros((8,), dtype=float)
    qpos_target = np.zeros((8,), dtype=float)
    qpos_target[7] = 1.0

    success, _, _ = mjho.sim_under_extforce(
        qpos_target,
        qpos_prepared,
        duration=0.0,
        check_steps=1,
        close_steps=4,
    )

    assert bool(success) is True
    assert np.allclose(mjho._ctrl_log[:4], [0.25, 0.5, 0.75, 1.0])


def test_build_pregrasp_qpos_uses_target_pose_and_prepared_joints():
    mjho = MjHO.__new__(MjHO)
    mjho.nq = 10
    mjho.nq_hand = 10

    qpos_target = np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 9.0, 8.0, 7.0])
    prepared_joints = np.asarray([0.1, 0.2, 0.3])

    qpos_prepared = mjho.build_pregrasp_qpos(qpos_target, prepared_joints)

    assert np.allclose(qpos_prepared[:7], qpos_target[:7])
    assert np.allclose(qpos_prepared[7:], prepared_joints)
