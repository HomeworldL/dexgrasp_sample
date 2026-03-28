from pathlib import Path

import numpy as np

from utils.utils_sample import global_pc_exists, global_pc_path, write_global_pc


def test_write_global_pc_writes_float32_points(tmp_path: Path):
    output_dir = tmp_path / "obj_a" / "scale080"
    points = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=np.float64)

    path = write_global_pc(points, str(output_dir), "pc_warp")

    assert path == global_pc_path(str(output_dir), "pc_warp")
    assert global_pc_exists(str(output_dir), "pc_warp")

    saved = np.load(path)
    assert saved.dtype == np.float32
    assert saved.shape == (2, 3)
    assert np.allclose(saved, points.astype(np.float32))
