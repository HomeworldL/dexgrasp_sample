import numpy as np

from run import _select_fail_samples


def test_select_fail_samples_is_deterministic_and_capped_by_valid_count():
    fail_qpos_rows = [np.full((3,), float(i), dtype=np.float32) for i in range(20)]
    fail_stages = [f"stage_{i}" for i in range(20)]

    selected_qpos_a, selected_stages_a = _select_fail_samples(
        fail_qpos_rows=fail_qpos_rows,
        fail_stages=fail_stages,
        valid_count=25,
        min_valid_count=10,
        fail_keep_ratio=1.2,
        seed=7,
    )
    selected_qpos_b, selected_stages_b = _select_fail_samples(
        fail_qpos_rows=fail_qpos_rows,
        fail_stages=fail_stages,
        valid_count=25,
        min_valid_count=10,
        fail_keep_ratio=1.2,
        seed=7,
    )

    assert len(selected_qpos_a) == 20
    assert len(selected_stages_a) == 20
    assert selected_stages_a == selected_stages_b
    assert all(np.allclose(a, b) for a, b in zip(selected_qpos_a, selected_qpos_b))


def test_select_fail_samples_requires_more_than_ten_valid_grasps():
    selected_qpos, selected_stages = _select_fail_samples(
        fail_qpos_rows=[np.zeros((3,), dtype=np.float32)],
        fail_stages=["prepared_contact"],
        valid_count=10,
        min_valid_count=11,
        fail_keep_ratio=1.2,
        seed=0,
    )

    assert selected_qpos == []
    assert selected_stages == []
