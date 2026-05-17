from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from utils.utils_sample import (
    ARRAY_DTYPE,
    H5_DTYPE,
    as_array,
    encode_h5_str,
    write_fail_npy_from_h5,
    write_grasp_npy_from_h5,
)


class ObjectGraspWriter:
    """Per-object-scale writer for bucket MJWarp sampling outputs."""

    def __init__(
        self,
        output_dir: str | Path,
        object_name: str,
        scale: Optional[float],
        hand_name: str,
        qpos_dim: int,
        max_cap: int,
        fail_keep_ratio: float,
        seed: int,
        h5_name: str = "grasp.h5",
        npy_name: str = "grasp.npy",
        fail_h5_name: str = "grasp_fail.h5",
        fail_npy_name: str = "grasp_fail.npy",
        flush_every: int = 0,
    ):
        if int(max_cap) <= 0:
            raise ValueError(f"max_cap must be positive, got {max_cap}.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.object_name = str(object_name)
        self.scale = scale
        self.hand_name = str(hand_name)
        self.qpos_dim = int(qpos_dim)
        self.max_cap = int(max_cap)
        self.fail_keep_ratio = float(fail_keep_ratio)
        self.seed = int(seed)
        self.flush_every = int(flush_every)
        self.valid_count = 0
        self._flushed_valid = 0
        self.fail_qpos_rows: List[np.ndarray] = []
        self.fail_stages: List[str] = []

        self.h5_path = self.output_dir / str(h5_name)
        self.npy_path = self.output_dir / str(npy_name)
        self.fail_h5_path = self.output_dir / str(fail_h5_name)
        self.fail_npy_path = self.output_dir / str(fail_npy_name)
        self._hf = h5py.File(self.h5_path, "w")
        self._hf.create_dataset("object_name", data=encode_h5_str(self.object_name))
        self._hf.create_dataset(
            "scale", data=np.float32(self.scale if self.scale is not None else np.nan)
        )
        self._hf.create_dataset("hand_name", data=encode_h5_str(self.hand_name))
        self._hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))
        self._ds_init = self._hf.create_dataset(
            "qpos_init",
            shape=(self.max_cap, self.qpos_dim),
            maxshape=(None, self.qpos_dim),
            dtype=H5_DTYPE,
        )
        self._ds_approach = self._hf.create_dataset(
            "qpos_approach",
            shape=(self.max_cap, self.qpos_dim),
            maxshape=(None, self.qpos_dim),
            dtype=H5_DTYPE,
        )
        self._ds_prepared = self._hf.create_dataset(
            "qpos_prepared",
            shape=(self.max_cap, self.qpos_dim),
            maxshape=(None, self.qpos_dim),
            dtype=H5_DTYPE,
        )
        self._ds_grasp = self._hf.create_dataset(
            "qpos_grasp",
            shape=(self.max_cap, self.qpos_dim),
            maxshape=(None, self.qpos_dim),
            dtype=H5_DTYPE,
        )
        self._ds_squeeze = self._hf.create_dataset(
            "qpos_squeeze",
            shape=(self.max_cap, self.qpos_dim),
            maxshape=(None, self.qpos_dim),
            dtype=H5_DTYPE,
        )

    @property
    def done(self) -> bool:
        return self.valid_count >= self.max_cap

    def add_valid(
        self,
        qpos_init: np.ndarray,
        qpos_approach: np.ndarray,
        qpos_prepared: np.ndarray,
        qpos_grasp: np.ndarray,
        qpos_squeeze: np.ndarray,
    ) -> bool:
        if self.done:
            return False
        row = self.valid_count
        self._ds_init[row] = np.asarray(qpos_init, dtype=ARRAY_DTYPE)
        self._ds_approach[row] = np.asarray(qpos_approach, dtype=ARRAY_DTYPE)
        self._ds_prepared[row] = np.asarray(qpos_prepared, dtype=ARRAY_DTYPE)
        self._ds_grasp[row] = np.asarray(qpos_grasp, dtype=ARRAY_DTYPE)
        self._ds_squeeze[row] = np.asarray(qpos_squeeze, dtype=ARRAY_DTYPE)
        self.valid_count += 1
        if (
            self.flush_every > 0
            and (self.valid_count - self._flushed_valid) >= self.flush_every
        ):
            self._hf.flush()
            self._flushed_valid = self.valid_count
        return True

    def add_fail(self, qpos_fail: np.ndarray, failure_stage: str) -> None:
        self.fail_qpos_rows.append(as_array(qpos_fail))
        self.fail_stages.append(str(failure_stage))

    def _select_fail_samples(self) -> tuple[List[np.ndarray], List[str]]:
        keep_count = min(
            len(self.fail_qpos_rows),
            int(np.floor(self.fail_keep_ratio * float(self.valid_count))),
        )
        if keep_count <= 0:
            return [], []
        indices = np.random.default_rng(self.seed).permutation(
            len(self.fail_qpos_rows)
        )[:keep_count]
        return (
            [self.fail_qpos_rows[int(idx)] for idx in indices],
            [self.fail_stages[int(idx)] for idx in indices],
        )

    def _write_fail_h5(self, qpos_fail: List[np.ndarray], stages: List[str]) -> None:
        if qpos_fail:
            qpos_fail_np = np.asarray(qpos_fail, dtype=ARRAY_DTYPE)
            stage_np = np.asarray(stages, dtype=object)
        else:
            qpos_fail_np = np.zeros((0, self.qpos_dim), dtype=ARRAY_DTYPE)
            stage_np = np.asarray([], dtype=object)
        failure_stage_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(self.fail_h5_path, "w") as hf:
            hf.create_dataset("object_name", data=encode_h5_str(self.object_name))
            hf.create_dataset(
                "scale",
                data=np.float32(self.scale if self.scale is not None else np.nan),
            )
            hf.create_dataset("hand_name", data=encode_h5_str(self.hand_name))
            hf.create_dataset("rot_repr", data=encode_h5_str("wxyz+qpos"))
            hf.create_dataset("qpos_fail", data=qpos_fail_np, dtype=H5_DTYPE)
            hf.create_dataset(
                "failure_stage",
                data=stage_np,
                dtype=failure_stage_dtype,
            )

    def close(self) -> None:
        self._ds_init.resize((self.valid_count, self.qpos_dim))
        self._ds_approach.resize((self.valid_count, self.qpos_dim))
        self._ds_prepared.resize((self.valid_count, self.qpos_dim))
        self._ds_grasp.resize((self.valid_count, self.qpos_dim))
        self._ds_squeeze.resize((self.valid_count, self.qpos_dim))
        self._hf.flush()
        self._hf.close()
        kept_qpos, kept_stages = self._select_fail_samples()
        self._write_fail_h5(kept_qpos, kept_stages)
        write_grasp_npy_from_h5(self.h5_path, self.npy_path)
        write_fail_npy_from_h5(self.fail_h5_path, self.fail_npy_path)
