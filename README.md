# dexgrasp_sample

MuJoCo-based offline dexterous grasp sampling for scaled 3D objects.

The current repository workflow is:

1. prepare object assets under `datasets/objdata_*`
2. build a manifest-driven object-scale index from prepared assets
3. sample surface points and grasp frames
4. run MuJoCo collision and stability filtering
5. save validated grasp states under `datasets/graspdata_*`
6. render multi-view partial point clouds with Warp
7. optionally build shape clusters and RL-only metadata under `_meta/`

This repository supports both:
- CPU MuJoCo sampling via `run.py` / `run_multi.py`
- GPU MJWarp sampling via `run_mjw.py` / `run_multi_mjw.py`

## Contents
- [Repository Introduction](#repository-introduction)
- [Installation and Dependencies](#installation-and-dependencies)
- [Dataset Construction](#dataset-construction)
- [Clustering and RL Metadata](#clustering-and-rl-metadata)
- [Run Commands](#run-commands)
- [Utility Scripts](#utility-scripts)
- [Main Files](#main-files)
- [Notes](#notes)
- [Tuning Notes](#tuning-notes)
- [TODO](#todo)
- [License](#license)

## Repository Introduction

Mainline purpose:
- prepare reusable object assets before sampling
- generate grasp candidates offline for object-scale entries
- validate them with MuJoCo physics
- save grasp states in a stable dataset format
- render post-sampling partial point clouds for downstream learning
- optionally build object-level shape clusters and RL manifests without depending on grasp files

Current mainline assumptions:
- config-first workflow
- manifest-gated dataset indexing
- object assets (`objdata_*`) and grasp outputs (`graspdata_*`) are stored separately
- one flattened global id space over object-scale entries
- `grasp.h5` is the source of truth, `grasp.npy` is derived from it
- mainline grasp arrays are stored as `float32`
- dataset splits are grouped by `object_name` to avoid cross-scale leakage

Default dataset/config conventions:
- default config: `configs/run_YCB_liberhand_right.json`
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
```

This repo commonly uses a symlinked mesh source:

```text
assets/objects -> /home/ccs/repositories/mesh_process/assets/objects
```

### Manifest Gating

Only objects listed in `manifest.process_meshes.json` with:
- `process_status = success`

are eligible for indexing.

### Object Asset Preparation

`prepare_object_assets.py` is the required preparation entrypoint before sampling.

It:
- scans manifest-eligible objects from `assets/objects/processed/<dataset>/manifest.process_meshes.json`
- builds object assets under `datasets/objdata_<dataset>/`
- writes `coacd.obj`, `object.xml`, `convex_parts/*.obj`
- writes `pc_warp/global_pc.npy` and `pc_warp/global_normals.npy`
- optionally builds peer `native/` assets when enabled in asset config

Example:

```bash
python prepare_object_assets.py -c configs/assets_YCB.json --force --jobs 8
```

`DatasetObjects` is now a read-only indexer over prepared assets. It should not rebuild missing assets implicitly.

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

Object assets and grasp outputs are separated:

```text
datasets/objdata_YCB/<object>/scaleXXX/
  coacd.obj
  object.xml
  convex_parts/*.obj
  pc_warp/
    global_pc.npy
    global_normals.npy

datasets/graspdata_YCB_liberhand_right/<object>/scaleXXX/
  grasp.h5
  grasp.npy
  grasp_fail.h5
  grasp_fail.npy
  pc_warp/
    cam_in.npy
    cam_ex_XX.npy
    partial_pc_XX.npy
    partial_pc_cam_XX.npy
```

Inside `grasp.h5` / `grasp.npy`, the mainline grasp arrays are:
- `qpos_init`
- `qpos_approach`
- `qpos_prepared`
- `qpos_grasp`
- `qpos_squeeze`

All of them are stored as `float32` in the current mainline.

Current replay note:
- stored `qpos_prepared` remains the original candidate pregrasp state
- extforce replay rebuilds a validation pregrasp from `qpos_squeeze` pose plus the stored prepared joints
- `sim_dataset.py` keeps the stored `qpos_init` pre-check, then calls `sim_under_extforce(qpos_target, rebuilt_qpos_prepared, ...)`

Failure sample note:
- `run.py` also exports `grasp_fail.h5` and `grasp_fail.npy`
- it stores `qpos_fail` plus `failure_stage`
- current retained failure stages are `prepared_contact`, `insufficient_contact`, and `extforce_failure`
- `approach_contact` and `init_contact` are not exported into the failure dataset
- `qpos_fail` stores the stage-specific failed state:
  - `prepared_contact`: colliding prepared state
  - `insufficient_contact`: post-close `qpos_grasp`
  - `extforce_failure`: failed `qpos_squeeze`
- if `valid_count < data.min_valid_count`, both positive and failure grasp files are truncated to zero rows
- otherwise failure samples are deterministically shuffled with config `seed` and truncated to `floor(data.fail_keep_ratio * valid_count)`

Global point cloud note:
- `run.py` loads `global_pc.npy` / `global_normals.npy` from objdata assets
- `run_mjw.py` writes `pc_warp/global_pc.npy` under grasp output dir
- it stores the initial world-frame object points sampled for grasp generation, not a merge of `partial_pc_XX.npy`
- its current mainline shape is `(sampling.n_points, 3)` with `float32`

### Dataset Split Policy

`build_dataset_splits.py` writes dataset manifests under:
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

Current split rules:
- split by `object_name`, not by object-scale row
- keep all scales of the same object in the same split
- default ratio is approximately `80/20` over unique objects, shuffled with config `seed`
- require positive grasp outputs, failure grasp outputs, and required render outputs to exist
- each manifest row includes:
  - `grasp_h5_path`, `grasp_npy_path`
  - `grasp_h5_fail_path`, `grasp_fail_npy_path`
  - `global_pc_path`
- filter out empty grasp files before writing final manifests
- store paths relative to dataset root for portability

## Clustering and RL Metadata

Shape clustering and RL split are separate from imitation-learning dataset construction.

### Shape Clustering

Use `run_shape_cluster.py` to build object-level shape features and KMeans clusters from prepared objdata assets:

```bash
python run_shape_cluster.py -c configs/assets_YCB.json --force
```

Current assumptions:
- uses prepared `global_pc.npy`
- uses one configured scale tag, typically `scale120`
- clustering is object-level
- `native` is excluded
- outputs are written only under `_meta`

Main output directory:

```text
datasets/objdata_YCB/_meta/shape_cluster/<cluster_tag>/
```

Main files:
- `meta.json`
- `object_features.npy`
- `cluster_centers.npy`
- `object_cluster.json`
- `cluster_index.json`
- `curriculum_index.json`

### RL-Only Split

Use `build_dataset_splits_rl.py` to build train/test manifests for RL from objdata assets plus shape-cluster metadata:

```bash
python build_dataset_splits_rl.py -c configs/assets_YCB.json --force
```

Main output directory:

```text
datasets/objdata_YCB/_meta/rl_split/<split_tag>/
```

Current assumptions:
- split by `object_name`
- expand back to object-scale entries
- inherit object-level cluster metadata onto each object-scale row
- use standard `scaleXXX` assets only
- do not require `grasp.h5` or Warp partial render outputs

### Cluster Visualization

Use `vis_shape_cluster.py` to render per-cluster thumbnail pages:

```bash
python vis_shape_cluster.py -c configs/assets_YCB.json
```

This writes cluster overview images under:

```text
datasets/objdata_YCB/_meta/shape_cluster/<cluster_tag>/vis/
```

## Run Commands

### CPU: Single Object-Scale

`run.py` samples one object-scale entry.

```bash
python run.py \
  -c configs/run_YCB_liberhand_right.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
  --force -v
```

Required arguments:
- `--object-scale-key`
- `--coacd-path`
- `--mjcf-path`
- `--output-dir`

### Visualization Helpers

Online grasp-only visualization without extforce:

```bash
python demo_grasp.py \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand_right.json -v
```

Online visualization that stops at the first grasp passing extforce:

```bash
python demo.py \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  -c configs/run_YCB_liberhand_right.json -v
```

Replay all saved grasps for one object-scale under MuJoCo extforce:

```bash
python vis_grasp_mujoco.py \
  --object-scale-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
  -c configs/run_YCB_liberhand_right.json -v
```

## Utility Scripts

### Inspect Indexed Objects

Use `print_dataset_objects.py` to inspect `DatasetObjects` entries without starting sampling:

```bash
python print_dataset_objects.py -c configs/run_YCB_liberhand_right.json
python print_dataset_objects.py -c configs/run_YCB_liberhand_right_native.json --native-only
```

### CPU: Whole Dataset in Parallel

`run_multi.py` only launches parallel grasp sampling.

```bash
python run_multi.py -c configs/run_YCB_liberhand_right.json -j 16 --force
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
  -c configs/run_YCB_liberhand_right.json \
  --object-scale-key YCB_002_master_chef_can__scale120 \
  --coacd-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/coacd.obj \
  --mjcf-path datasets/objdata_YCB/YCB_002_master_chef_can/scale120/object.xml \
  --output-dir datasets/graspdata_YCB_liberhand_right/YCB_002_master_chef_can/scale120 \
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
  -c configs/run_YCB_liberhand_right.json \
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
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -j 2
```

Single entry:

```bash
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -i 0
python run_warp_render.py -c configs/run_YCB_liberhand_right.json -k YCB_002_master_chef_can__scale080
```

### Dataset Split Manifest Build

`build_dataset_splits.py` scans existing grasp and render outputs and writes:
- `datasets/<dataset_tag>/train.json`
- `datasets/<dataset_tag>/test.json`

```bash
python build_dataset_splits.py -c configs/run_YCB_liberhand_right.json
```

### Contact Parameter Ablation

`scripts/solimp_solref_experiment.py` runs four `solimp/solref` cases on one object-scale:
- `hand_soft_obj_hard`
- `hand_hard_obj_soft`
- `hand_hard_obj_hard`
- `hand_soft_obj_soft`

Default outputs:
- generated case configs: `scripts/configs/solimp_solref_cases/`
- experiment logs/results: `tmp/solimp_solref_experiment/`

```bash
python scripts/solimp_solref_experiment.py \
  --object-scale-key YCB_013_apple__scale080 \
  --asset-dir datasets/objdata_YCB/YCB_013_apple/scale080 \
  --work-dir tmp/solimp_solref_experiment_YCB_013_apple_scale080 \
  --parallel-cases --max-workers 4
```

### Dataset Simulation Check

`sim_dataset.py` replays saved dataset grasps from `train.json` / `test.json` with
`MjHO.sim_under_extforce`.

Current mainline validation:
- loads stored `qpos_squeeze`
- re-checks saved `qpos_init` / `qpos_prepared` collision gates before extforce
- defaults to `float32` qpos casting, matching the stored dataset dtype

Default validation uses `float32` qpos casting:

```bash
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train -v
```

To compare simulated success rates under `float32` and `float64` casting:

```bash
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train --dtype float32 -v
python sim_dataset.py -c configs/run_YCB_liberhand_right.json --split train --dtype float64 -v
```

### One-Command Pipeline

`scripts/run_pipeline.sh` runs:
1. page-cache drop
2. `run_multi.py`
3. `run_warp_render.py`
4. `build_dataset_splits.py`

```bash
bash scripts/run_pipeline.sh -c configs/run_YCB_liberhand_right.json
```

Overrides:

```bash
bash scripts/run_pipeline.sh \
  -c configs/run_YCB_liberhand_right.json \
  --cpu-set 0-23 \
  --run-j 24 \
  --render-j 2 \
  --no-force \
  --no-drop-caches
```

### Visualization

Object mesh:

```bash
python vis_obj.py -c configs/run_YCB_liberhand_right.json -i 0
python vis_obj.py -c configs/run_YCB_liberhand_right.json -k YCB_002_master_chef_can__scale080
```

Hand-object state:

```bash
python vis_ho.py -c configs/run_YCB_liberhand_right.json -i 0
```

Saved grasp states:

```bash
python vis_grasp.py -c configs/run_YCB_liberhand_right.json -i 0
python vis_grasp.py -c configs/run_YCB_liberhand_right.json -i 0 --vis-ids 0,10,-1 --frame-stage qpos_grasp
```

Saved partial point clouds:

```bash
python vis_pc.py -c configs/run_YCB_liberhand_right.json -i 0 --show-cam-frames
```

Plot grasp pose distributions:

```bash
PYTHONPATH=. python tools/visualization/plot_grasp_pose_plotly.py -c configs/run_YCB_liberhand_right.json -i 0
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
- [sim_dataset.py](/home/ccs/repositories/dexgrasp_sample/sim_dataset.py)
  Replay saved dataset grasps and report extforce success rates under `float32` / `float64`
- [scripts/run_pipeline.sh](/home/ccs/repositories/dexgrasp_sample/scripts/run_pipeline.sh)
  One-command full pipeline shell script
- [scripts/solimp_solref_experiment.py](/home/ccs/repositories/dexgrasp_sample/scripts/solimp_solref_experiment.py)
  Single-object contact parameter ablation with optional parallel case execution

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
- [vis_pc.py](/home/ccs/repositories/dexgrasp_sample/vis_pc.py)

## Notes

- Mainline is config-first. Do not expect hidden Python defaults to replace missing config fields.
- `run_multi.py` only performs parallel sampling. Dataset split export is now separate in `build_dataset_splits.py`.
- `run_warp_render.py` supports both single-entry mode and full-dataset mode.
- `grasp.h5` is the authoritative result file. `grasp.npy` is always derived from it.
- Mainline grasp arrays in `grasp.h5` / `grasp.npy` are stored as `float32`.
- Mainline dataset format stores `qpos_init`, `qpos_approach`, `qpos_prepared`, `qpos_grasp`, and `qpos_squeeze`.
- CPU extforce validation uses a two-stage check: no-force settling drift first, then six-direction force checks from the settled pose.
- `sim_dataset.py` validates stored `qpos_squeeze` and can cast saved qpos arrays to either `float32` or `float64` for stability comparison.
- `build_dataset_splits.py` splits by object, not by object-scale row, to avoid leakage across scales.
- `docs/` is treated as local reference in this repo workflow and may be gitignored.
- CPU and GPU sampling share the same object-scale interface, but not the same runtime behavior.
- MJWarp GPU execution is not guaranteed to be deterministic across repeated runs.

## Tuning Notes

### CPU
- `data.flush_every = 25` is the current default and avoids flushing HDF5 on every accepted sample.
- `run.py` uses `data.max_time_sec` as a total wall-clock cap for the whole CPU sampling run.
- `sim_grasp(record_history=False)` is the default hot-path setting; enable history only for visualization/debugging.
- For cleaner CPU benchmarks:

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0-15 python run_multi.py -c configs/run_YCB_liberhand_right.json -j 16
```

### MJWarp
- `data.max_cap = 100` is still the practical target for the current MJWarp path.
- `data.max_time_sec = 180.0` is the current extforce wall-clock cap in `run_mjw.py`.
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
