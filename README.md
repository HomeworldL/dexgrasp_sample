# dexgrasp_sample

MuJoCo-based offline dexterous grasp sampling for scaled 3D objects.

The mainline pipeline is:

1. build a manifest-driven object-scale index
2. sample surface points and grasp frames
3. run MuJoCo collision and stability filtering
4. save validated grasps to `grasp.h5` and `grasp.npy`
5. render multi-view partial point clouds with Warp
6. build dataset-level `train.json` and `test.json`

This repository supports both:
- CPU MuJoCo sampling via `run.py` / `run_multi.py`
- GPU MJWarp sampling via `run_mjw.py` / `run_multi_mjw.py`

## Contents
- [Repository Introduction](#repository-introduction)
- [Installation and Dependencies](#installation-and-dependencies)
- [Dataset Construction](#dataset-construction)
- [Run Commands](#run-commands)
- [Main Files](#main-files)
- [Notes](#notes)
- [Tuning Notes](#tuning-notes)
- [TODO](#todo)
- [License](#license)

## Repository Introduction

Mainline purpose:
- generate grasp candidates offline for object-scale entries
- validate them with MuJoCo physics
- save grasp states in a stable dataset format
- render post-sampling partial point clouds for downstream learning

Current mainline assumptions:
- config-first workflow
- manifest-gated dataset indexing
- one flattened global id space over object-scale entries
- `grasp.h5` is the source of truth, `grasp.npy` is derived from it

Default dataset/config conventions:
- default config: `configs/run_YCB_liberhand.json`
- default dataset root: `assets/objects/processed`
- common generated dataset tag rule: `run_<...>.json -> graspdata_<...>`

## Installation and Dependencies

Environment assumptions:
- Ubuntu Linux
- Python 3.10
- MuJoCo
- Open3D / trimesh / h5py / tqdm / transforms3d
- optional Warp stack for GPU rendering and MJWarp sampling

Recommended install flow:

```bash
conda create -n grasp python=3.10 -y
conda activate grasp
pip install -r requirements.txt
```

If you use Warp / MJWarp:
- ensure NVIDIA driver and CUDA runtime are available
- install `warp-lang`
- install `mujoco_warp`

Quick sanity checks:

```bash
python run.py --help
python run_multi.py --help
python run_warp_render.py --help
python run_mjw.py --help
```

## Dataset Construction

### Input Layout

`DatasetObjects` indexes processed mesh datasets under:

```text
assets/objects/processed/
  YCB/
  DGN/
  DGN2/
  HOPE/
```

This repo commonly uses a symlinked mesh source:

```text
assets/objects -> /home/ccs/repositories/mesh_process/assets/objects
```

### Manifest Gating

Only objects listed in `manifest.process_meshes.json` with:
- `process_status = success`

are eligible for indexing.

### Object-Scale Flattening

Every `(object, scale)` pair becomes one flat entry with:
- `global_id`
- `object_scale_key`
- `object_name`
- `scale`
- `coacd_abs`
- `mjcf_abs`
- `output_dir_abs`

Example scale list used by mainline liberhand configs:

```json
[0.08, 0.10, 0.12, 0.14, 0.16]
```

### Generated Output Layout

For a dataset tag like `graspdata_YCB_liberhand`, outputs are written under:

```text
datasets/graspdata_YCB_liberhand/<object>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/*.obj
  grasp.h5
  grasp.npy
  partial_pc_warp/
    cam_in.npy
    cam_ex_XX.npy
    partial_pc_XX.npy
    partial_pc_cam_XX.npy
```

## Run Commands

### CPU: Single Object-Scale

`run.py` samples one object-scale entry.

```bash
python run.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120 \
  --force -v
```

Required arguments:
- `--object-scale-key`
- `--coacd-path`
- `--mjcf-path`
- `--output-dir`

### CPU: Whole Dataset in Parallel

`run_multi.py` only launches parallel grasp sampling.

```bash
python run_multi.py -c configs/run_YCB_liberhand.json -j 16 --force
```

Useful flags:
- `-j/--max-parallel`
- `--script` to swap in another single-entry script
- `--force`
- `-v` to forward verbose mode to child `run.py`

Logs are stored under:

```text
logs/run/<dataset_tag>/
```

### GPU MJWarp: Single Object-Scale

```bash
python run_mjw.py \
  -c configs/run_YCB_liberhand.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand/YCB_002_master_chef_can/scale120 \
  --batch-size 512 \
  --nconmax 32 \
  --naconmax 16384 \
  --njmax 200 \
  --ccd-iterations 200 \
  --force -v
```

### GPU MJWarp: Whole Dataset in Parallel

```bash
python run_multi_mjw.py \
  -c configs/run_YCB_liberhand.json \
  -j 4 \
  --batch-size 512 \
  --njmax 200 \
  --ccd-iterations 200 \
  --force
```

Logs are stored under:

```text
logs/run_mjw/<dataset_tag>/
```

### Warp Partial Point Cloud Rendering

`run_warp_render.py` supports both:
- single entry mode via `-i` or `-k`
- full dataset mode when neither is passed

Whole dataset:

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -j 2
```

Single entry:

```bash
python run_warp_render.py -c configs/run_YCB_liberhand.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

### Dataset Split Manifest Build

`build_dataset_splits.py` scans existing grasp and render outputs and writes:
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

```bash
python build_dataset_splits.py -c configs/run_YCB_liberhand.json
```

### One-Command Pipeline

`scripts/run_pipeline.sh` runs:
1. page-cache drop
2. `run_multi.py`
3. `run_warp_render.py`
4. `build_dataset_splits.py`

```bash
bash scripts/run_pipeline.sh -c configs/run_YCB_liberhand.json
```

Overrides:

```bash
bash scripts/run_pipeline.sh \
  -c configs/run_YCB_liberhand.json \
  --cpu-set 0-23 \
  --run-j 24 \
  --render-j 2 \
  --no-force \
  --no-drop-caches
```

### Visualization

Object mesh:

```bash
python vis_obj.py -c configs/run_YCB_liberhand.json -i 0
python vis_obj.py -c configs/run_YCB_liberhand.json -k YCB_002_master_chef_can__scale080
```

Hand-object state:

```bash
python vis_ho.py -c configs/run_YCB_liberhand.json -i 0
```

Saved grasp states:

```bash
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

Saved partial point clouds:

```bash
python vis_partial_pc.py -c configs/run_YCB_liberhand.json -i 0 --show-cam-frames
```

Plot grasp pose distributions:

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand.json -i 0
```

## Main Files

### Entrypoints
- [run.py](/home/ccs/repositories/dexgrasp_sample/run.py)
  CPU single-entry grasp sampling
- [run_multi.py](/home/ccs/repositories/dexgrasp_sample/run_multi.py)
  CPU dataset-level parallel launcher
- [run_mjw.py](/home/ccs/repositories/dexgrasp_sample/run_mjw.py)
  GPU MJWarp single-entry sampling
- [run_multi_mjw.py](/home/ccs/repositories/dexgrasp_sample/run_multi_mjw.py)
  GPU MJWarp dataset-level parallel launcher
- [run_warp_render.py](/home/ccs/repositories/dexgrasp_sample/run_warp_render.py)
  Warp partial point cloud rendering
- [build_dataset_splits.py](/home/ccs/repositories/dexgrasp_sample/build_dataset_splits.py)
  Build `train.json` / `test.json`
- [scripts/run_pipeline.sh](/home/ccs/repositories/dexgrasp_sample/scripts/run_pipeline.sh)
  One-command full pipeline shell script

### Core Modules
- [src/dataset_objects.py](/home/ccs/repositories/dexgrasp_sample/src/dataset_objects.py)
  Manifest-driven object-scale indexing and asset lookup
- [src/mj_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mj_ho.py)
  CPU MuJoCo hand-object simulation helper
- [src/mjw_ho.py](/home/ccs/repositories/dexgrasp_sample/src/mjw_ho.py)
  MJWarp batched simulation helper
- [src/sample.py](/home/ccs/repositories/dexgrasp_sample/src/sample.py)
  Grasp frame sampling logic
- [utils/utils_sample.py](/home/ccs/repositories/dexgrasp_sample/utils/utils_sample.py)
  Shared sampling/output helpers
- [utils/utils_file.py](/home/ccs/repositories/dexgrasp_sample/utils/utils_file.py)
  Config loading, path utilities, logs-path helpers

### Visualization
- [vis_obj.py](/home/ccs/repositories/dexgrasp_sample/vis_obj.py)
- [vis_obj_scale.py](/home/ccs/repositories/dexgrasp_sample/vis_obj_scale.py)
- [vis_ho.py](/home/ccs/repositories/dexgrasp_sample/vis_ho.py)
- [vis_grasp.py](/home/ccs/repositories/dexgrasp_sample/vis_grasp.py)
- [vis_partial_pc.py](/home/ccs/repositories/dexgrasp_sample/vis_partial_pc.py)

## Notes

- Mainline is config-first. Do not expect hidden Python defaults to replace missing config fields.
- `run_multi.py` only performs parallel sampling. Dataset split export is now separate in `build_dataset_splits.py`.
- `run_warp_render.py` supports both single-entry mode and full-dataset mode.
- `grasp.h5` is the authoritative result file. `grasp.npy` is always derived from it.
- `docs/` is treated as local reference in this repo workflow and may be gitignored.
- CPU and GPU sampling share the same object-scale interface, but not the same runtime behavior.
- MJWarp GPU execution is not guaranteed to be deterministic across repeated runs.

## Tuning Notes

### CPU
- `output.flush_every = 25` is the current default and avoids flushing HDF5 on every accepted sample.
- `run.py` uses `output.max_time_sec` as a total wall-clock cap for the whole CPU sampling run.
- `sim_grasp(record_history=False)` is the default hot-path setting; enable history only for visualization/debugging.
- For cleaner CPU benchmarks:

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand.json -j 16
```

### MJWarp
- `output.max_cap = 100` is still the practical target for the current MJWarp path.
- `output.max_time_sec = 180.0` is the current extforce wall-clock cap in `run_mjw.py`.
- If the goal is to reach about `100` valid grasps quickly, prefer `batch_size` near the target, typically `128`, `256`, or `512`.
- Very large `batch_size` values such as `4096` are better suited to large-scale harvesting than to early-stop collection around `100` valid grasps.
- Current mainline default `sampling.n_points = 2048` is the compromise point for the `max_cap = 100` workflow.

## TODO

Current follow-up directions:
- Improve GPU sampling throughput, likely including cross-object batching instead of only one object per batch.
- Explore different grasp families beyond the current setting, for example 2-finger, 3-finger, and 4-finger grasps.

Active plan file:
- [TODO.md](/home/ccs/repositories/dexgrasp_sample/TODO.md)

## References

- BODex: <https://github.com/JYChen18/BODex>
- DexGraspBench: <https://github.com/JYChen18/DexGraspBench>

## License

This project is released under the MIT License.
See [LICENSE](/home/ccs/repositories/dexgrasp_sample/LICENSE).
